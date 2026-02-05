import base64
import hashlib
import html
import json
import logging
import os
import secrets
import string
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from logging.handlers import RotatingFileHandler

from relay.auth import AuthHandler
from relay.config import (
    ADMIN_AUTH_TOKEN,
    ADMIN_DASHBOARD_PASSWORD,
    ADMIN_DASHBOARD_REFRESH_SECONDS,
    ADMIN_DASHBOARD_USERNAME,
    CONCENSUS_BLACKLIST,
    CONTRACT_ABI,
    CONTRACT_ADDRESS,
    DEFAULT_SOTA_THRESHOLD,
    DUPLICATE_TIME_WINDOW,
    MAX_SUBMISSIONS_PER_HOUR,
    RPC_URL,
    SOTA_ACTIVATION_DELAY_INTERVALS,
    SOTA_ALIGNMENT_MOD,
    SOTA_CONSENSUS_VOTES,
    SOTA_MIN_T2_INTERVALS,
    SOTA_T2_BLOCKS,
    SOTA_T2_INTERVALS,
    SessionLocal,
    RELAY_UID0_HOTKEY,
    TEST_INVITE_CODE,
    TEST_MODE,
    engine,
)
from relay.models import (
    Base,
    BlacklistVote,
    InvitationCode,
    MiningResult,
    SOTACache,
    SOTAEvent,
    SOTAVote,
    TestSubmission,
)
from relay.schemas import (
    FrontendSOTAEvent,
    FrontendSOTAInfo,
    LinkInvitationCode,
    MiningResultResponse,
    PaginatedSOTAEvents,
    ResultSubmission,
    SOTAEventResponse,
    SOTAResponse,
    SOTAVoteRequest,
    SOTAVoteResponse,
    TestSubmissionResponse,
    UpdateColdkeyAddress,
)
from relay.sota import compute_sota_window

try:
    from common.contract_manager import ContractManager
except Exception:  # pragma: no cover
    ContractManager = None


def _setup_relay_logging() -> None:
    relay_logger = logging.getLogger("relay")
    if getattr(relay_logger, "_configured", False):
        return

    level_name = (os.getenv("RELAY_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    relay_logger.setLevel(level)
    relay_logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    relay_logger.addHandler(stream_handler)

    log_file = os.getenv("RELAY_LOG_FILE", "logs/relay.log").strip()
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=int(os.getenv("RELAY_LOG_MAX_BYTES", str(10 * 1024 * 1024))),
                backupCount=int(os.getenv("RELAY_LOG_BACKUP_COUNT", "5")),
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            relay_logger.addHandler(file_handler)
        except Exception:
            relay_logger.exception("Failed to configure RELAY_LOG_FILE=%r", log_file)

    relay_logger._configured = True


_setup_relay_logging()
logger = logging.getLogger(__name__)


class SubmissionTracker:
    def __init__(self):
        self.submission_times = defaultdict(list)
        self.submission_hashes = {}
        self.lock = threading.Lock()

    def clean_old_entries(self, current_time, miner_hotkey):
        with self.lock:
            self.submission_times[miner_hotkey] = [
                ts
                for ts in self.submission_times[miner_hotkey]
                if current_time - ts < timedelta(hours=1)
            ]
            self.submission_hashes = {
                h: ts
                for h, ts in self.submission_hashes.items()
                if current_time - ts < timedelta(seconds=DUPLICATE_TIME_WINDOW)
            }

    def check_rate_limit(self, miner_hotkey):
        return len(self.submission_times[miner_hotkey]) >= MAX_SUBMISSIONS_PER_HOUR

    def is_duplicate(self, solution_hash):
        return solution_hash in self.submission_hashes

    def add_submission(self, miner_hotkey, solution_hash, current_time):
        with self.lock:
            self.submission_times[miner_hotkey].append(current_time)
            self.submission_hashes[solution_hash] = current_time


class RequestMetrics:
    def __init__(self, *, retention_minutes: int = 24 * 60, retention_hours: int = 24 * 7):
        self.retention_minutes = int(retention_minutes)
        self.retention_hours = int(retention_hours)
        self.started_at = time.time()
        self.last_request_ts: float | None = None
        self.lock = threading.Lock()

        self.minute_counts = defaultdict(lambda: defaultdict(int))
        self.hour_counts = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)

        self.minute_5xx = defaultdict(int)
        self.hour_5xx = defaultdict(int)
        self.total_5xx = 0

    def _prune(self, current_minute: int, current_hour: int) -> None:
        min_cutoff = current_minute - self.retention_minutes
        hour_cutoff = current_hour - self.retention_hours

        for _, buckets in list(self.minute_counts.items()):
            for k in list(buckets.keys()):
                if k < min_cutoff:
                    del buckets[k]

        for _, buckets in list(self.hour_counts.items()):
            for k in list(buckets.keys()):
                if k < hour_cutoff:
                    del buckets[k]

        for k in list(self.minute_5xx.keys()):
            if k < min_cutoff:
                del self.minute_5xx[k]
        for k in list(self.hour_5xx.keys()):
            if k < hour_cutoff:
                del self.hour_5xx[k]

    def record(self, *, category: str, status_code: int) -> None:
        ts = time.time()
        minute = int(ts // 60)
        hour = int(ts // 3600)

        with self.lock:
            self.last_request_ts = ts
            self.total_counts["total"] += 1
            self.total_counts[category] += 1

            self.minute_counts["total"][minute] += 1
            self.minute_counts[category][minute] += 1
            self.hour_counts["total"][hour] += 1
            self.hour_counts[category][hour] += 1

            if int(status_code) >= 500:
                self.total_5xx += 1
                self.minute_5xx[minute] += 1
                self.hour_5xx[hour] += 1

            self._prune(minute, hour)

    def snapshot(self) -> dict:
        now = time.time()
        current_minute = int(now // 60)
        current_hour = int(now // 3600)

        def sum_minutes(category: str, minutes: int) -> int:
            buckets = self.minute_counts.get(category, {})
            start = current_minute - int(minutes) + 1
            return sum(int(buckets.get(m, 0)) for m in range(start, current_minute + 1))

        def sum_hours(category: str, hours: int) -> int:
            buckets = self.hour_counts.get(category, {})
            start = current_hour - int(hours) + 1
            return sum(int(buckets.get(h, 0)) for h in range(start, current_hour + 1))

        def sum_5xx_minutes(minutes: int) -> int:
            start = current_minute - int(minutes) + 1
            return sum(int(self.minute_5xx.get(m, 0)) for m in range(start, current_minute + 1))

        def sum_5xx_hours(hours: int) -> int:
            start = current_hour - int(hours) + 1
            return sum(int(self.hour_5xx.get(h, 0)) for h in range(start, current_hour + 1))

        with self.lock:
            return {
                "uptime_seconds": max(0.0, now - self.started_at),
                "last_request_ts": self.last_request_ts,
                "totals": dict(self.total_counts),
                "last_1m": {
                    "total": sum_minutes("total", 1),
                    "miner": sum_minutes("miner", 1),
                    "validator": sum_minutes("validator", 1),
                    "admin": sum_minutes("admin", 1),
                    "public": sum_minutes("public", 1),
                    "other": sum_minutes("other", 1),
                    "errors_5xx": sum_5xx_minutes(1),
                },
                "last_1h": {
                    "total": sum_minutes("total", 60),
                    "miner": sum_minutes("miner", 60),
                    "validator": sum_minutes("validator", 60),
                    "admin": sum_minutes("admin", 60),
                    "public": sum_minutes("public", 60),
                    "other": sum_minutes("other", 60),
                    "errors_5xx": sum_5xx_minutes(60),
                },
                "last_24h": {
                    "total": sum_hours("total", 24),
                    "miner": sum_hours("miner", 24),
                    "validator": sum_hours("validator", 24),
                    "admin": sum_hours("admin", 24),
                    "public": sum_hours("public", 24),
                    "other": sum_hours("other", 24),
                    "errors_5xx": sum_5xx_hours(24),
                },
                "errors_5xx_total": int(self.total_5xx),
            }


def _classify_request_path(path: str) -> str:
    if not path:
        return "other"

    if path.startswith("/admin"):
        return "admin"

    if path.startswith("/submit_solution") or path.startswith("/coldkey_address/update"):
        return "miner"
    if path.startswith("/invitation_code/linked") or path.startswith("/invitation_code/link"):
        return "miner"

    if path.startswith("/results") or path.startswith("/sota/vote") or path.startswith("/sota/events"):
        return "validator"
    if path.startswith("/blacklist/") or path.startswith("/test/"):
        return "validator"

    if path in {"/health", "/version.json", "/sota_threshold", "/sota-events"}:
        return "public"
    if path.startswith("/docs") or path.startswith("/openapi") or path.startswith("/redoc"):
        return "public"

    return "other"


request_metrics = RequestMetrics()


app = FastAPI(
    title="AutoML Validator Proxy",
    description="API to enable miners to submit results and validators to query results",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bitsota.com", "https://www.bitsota.com", "https://bitsota.ai", "https://www.bitsota.ai","https://bitsota.io", "https://www.bitsota.io","https://bitsota.xyz", "https://www.bitsota.xyz", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    category = _classify_request_path(request.url.path)

    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled exception method=%s path=%s request_id=%s x_key=%s",
            request.method,
            str(request.url),
            request_id,
            (request.headers.get("X-Key") or "")[:8] or None,
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "request_id": request_id},
        )
    duration_ms = (time.perf_counter() - start) * 1000.0

    status_code = getattr(response, "status_code", 0) or 0
    request_metrics.record(category=category, status_code=status_code)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "HTTP %s %s status=%s ms=%.1f request_id=%s x_key=%s",
        request.method,
        str(request.url),
        status_code,
        duration_ms,
        request_id,
        (request.headers.get("X-Key") or "")[:8] or None,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    logger.exception(
        "Unhandled exception (handler) method=%s path=%s request_id=%s",
        request.method,
        str(request.url),
        request_id,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": request_id},
    )

contract_manager = None
if ContractManager is not None and CONTRACT_ADDRESS:
    contract_manager = ContractManager(
        rpc_url=RPC_URL,
        contract_address=CONTRACT_ADDRESS,
        abi=CONTRACT_ABI,
        no_wallet=True,
    )

auth_handler = AuthHandler(
    contract_manager=contract_manager,
    netuid=int(os.getenv("RELAY_NETUID", "94")),
    network=os.getenv("RELAY_NETWORK", "finney"),
)
submission_tracker = SubmissionTracker()

logger.info("Initializing AutoML Relay Service")
Base.metadata.create_all(bind=engine)
logger.info("Database tables initialized successfully")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_submission_tracker():
    return submission_tracker


def verify_hotkey(hotkey: str) -> bool:
    if not hotkey or len(hotkey) != 48:
        return False
    return hotkey.isalnum()


def _resolve_uid0_hotkey() -> Optional[str]:
    if RELAY_UID0_HOTKEY:
        return RELAY_UID0_HOTKEY
    try:
        return auth_handler.get_uid0_hotkey()
    except Exception:
        return None


async def require_uid0_validator(
    validator_hotkey: str = Depends(auth_handler.auth_decorator),
) -> str:
    uid0_hotkey = _resolve_uid0_hotkey()
    if uid0_hotkey:
        if validator_hotkey != uid0_hotkey:
            raise HTTPException(status_code=403, detail="Only UID0 validator may access this endpoint")
        return validator_hotkey
    if TEST_MODE:
        # Local/test deployments may not have a metagraph; allow any authenticated validator.
        return validator_hotkey
    raise HTTPException(status_code=503, detail="UID0 validator not configured or unavailable")


def require_admin_dashboard_auth(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    auth_token: Optional[str] = Header(None, alias="X-Auth-Token"),
) -> bool:
    if auth_token and secrets.compare_digest(auth_token, ADMIN_AUTH_TOKEN):
        return True

    if not authorization or not authorization.startswith("Basic "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    raw = authorization.split(" ", 1)[1].strip()
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header",
            headers={"WWW-Authenticate": "Basic"},
        )

    if ":" not in decoded:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header",
            headers={"WWW-Authenticate": "Basic"},
        )

    username, password = decoded.split(":", 1)
    ok = secrets.compare_digest(username, ADMIN_DASHBOARD_USERNAME) and secrets.compare_digest(
        password, ADMIN_DASHBOARD_PASSWORD
    )
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


def _ceil_to_mod(block: int, mod: int) -> int:
    if mod <= 0:
        raise ValueError("mod must be > 0")
    r = block % mod
    return block if r == 0 else (block + (mod - r))


_sota_vote_lock = threading.Lock()


def _get_or_create_sota_cache(db: Session) -> SOTACache:
    cached_entry = db.query(SOTACache).filter(SOTACache.task_type == "global").first()
    if cached_entry:
        return cached_entry
    cached_entry = SOTACache(
        task_type="global",
        sota_value=DEFAULT_SOTA_THRESHOLD,
        updated_at=datetime.now(UTC),
    )
    db.add(cached_entry)
    db.commit()
    db.refresh(cached_entry)
    return cached_entry


def _attach_coldkeys(results_db, db: Session):
    if not results_db:
        return results_db

    miner_hotkeys = {result.miner_hotkey for result in results_db if result.miner_hotkey}
    if not miner_hotkeys:
        return results_db

    coldkey_rows = (
        db.query(InvitationCode.miner_hotkey, InvitationCode.coldkey_address)
        .filter(InvitationCode.miner_hotkey.in_(miner_hotkeys))
        .all()
    )
    coldkey_map = {miner_hotkey: coldkey for miner_hotkey, coldkey in coldkey_rows}
    for result in results_db:
        result.coldkey_address = coldkey_map.get(result.miner_hotkey)
    return results_db


def generate_invitation_code(length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(UTC)}


@app.get("/version.json")
async def get_version_info():
    from pathlib import Path

    version_file = Path(__file__).parent / "version.json"
    if version_file.exists():
        return json.loads(version_file.read_text())
    raise HTTPException(status_code=404, detail="Version info not found")


@app.get("/sota_threshold", response_model=SOTAResponse)
async def get_sota_threshold(db: Session = Depends(get_db)):
    cached_entry = _get_or_create_sota_cache(db)
    return SOTAResponse(
        task_type="global",
        sota_threshold=cached_entry.sota_value,
        updated_at=cached_entry.updated_at,
        cached=True,
    )


@app.post("/sota/vote", response_model=SOTAVoteResponse)
async def vote_sota(
    vote: SOTAVoteRequest,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(auth_handler.auth_decorator),
):
    """Validators vote to accept a new SOTA (capacitorless mode)."""
    if not verify_hotkey(vote.miner_hotkey):
        raise HTTPException(status_code=400, detail="Invalid miner hotkey format")
    if vote.seen_block <= 0:
        raise HTTPException(status_code=400, detail="seen_block must be > 0")

    with _sota_vote_lock:
        sota_entry = _get_or_create_sota_cache(db)
        current_sota = float(sota_entry.sota_value)
        if float(vote.score) <= current_sota:
            raise HTTPException(
                status_code=400,
                detail=f"Score {vote.score} is not above current SOTA {current_sota}",
            )

        score_int = int(round(float(vote.score) * 10**18))
        vote_round = _ceil_to_mod(int(vote.seen_block), int(SOTA_ALIGNMENT_MOD))

        existing_vote = (
            db.query(SOTAVote)
            .filter(SOTAVote.validator_hotkey == validator_hotkey)
            .first()
        )
        if existing_vote:
            existing_round = _ceil_to_mod(
                int(existing_vote.seen_block or 0), int(SOTA_ALIGNMENT_MOD)
            )
            # Auto-expire stale votes if the validator is now in a new alignment window.
            # This prevents a "stuck" round when consensus never finalizes.
            if existing_round != vote_round:
                db.delete(existing_vote)
                db.flush()
                existing_vote = None

        if existing_vote:
            if (
                existing_vote.miner_hotkey == vote.miner_hotkey
                and existing_vote.score_int == score_int
            ):
                votes_for_candidate = (
                    db.query(SOTAVote)
                    .filter(
                        SOTAVote.miner_hotkey == vote.miner_hotkey,
                        SOTAVote.score_int == score_int,
                    )
                    .count()
                )
                return SOTAVoteResponse(
                    status="already_voted",
                    votes_for_candidate=votes_for_candidate,
                    votes_needed=SOTA_CONSENSUS_VOTES,
                    current_sota=current_sota,
                    finalized_event=None,
                )
            # Allow updating a vote within the same round only if the candidate is not worse.
            # This lets validators move to a better SOTA candidate as new results arrive.
            if score_int > int(existing_vote.score_int) or (
                score_int == int(existing_vote.score_int)
                and existing_vote.miner_hotkey != vote.miner_hotkey
            ):
                existing_vote.miner_hotkey = vote.miner_hotkey
                existing_vote.score_int = score_int
                existing_vote.score = float(vote.score)
                existing_vote.result_id = vote.result_id
                existing_vote.seen_block = int(vote.seen_block)
                existing_vote.timestamp = datetime.now(UTC)
                db.flush()
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Validator has already voted this round (existing vote is better or equal)"
                    ),
                )
        else:
            db.add(
                SOTAVote(
                    validator_hotkey=validator_hotkey,
                    miner_hotkey=vote.miner_hotkey,
                    score_int=score_int,
                    score=float(vote.score),
                    result_id=vote.result_id,
                    seen_block=int(vote.seen_block),
                )
            )
            db.flush()

        votes_for_candidate = (
            db.query(SOTAVote)
            .filter(
                SOTAVote.miner_hotkey == vote.miner_hotkey,
                SOTAVote.score_int == score_int,
            )
            .count()
        )

        finalized_event = None
        if votes_for_candidate >= SOTA_CONSENSUS_VOTES:
            existing_event = (
                db.query(SOTAEvent)
                .filter(
                    SOTAEvent.miner_hotkey == vote.miner_hotkey,
                    SOTAEvent.score_int == score_int,
                )
                .first()
            )
            if existing_event:
                finalized_event = existing_event
            else:
                decision_block = (
                    db.query(func.max(SOTAVote.seen_block))
                    .filter(
                        SOTAVote.miner_hotkey == vote.miner_hotkey,
                        SOTAVote.score_int == score_int,
                    )
                    .scalar()
                )
                if decision_block is None:
                    decision_block = int(vote.seen_block)

                start_block, end_block, effective_t2_blocks = compute_sota_window(
                    decision_block=int(decision_block),
                    alignment_mod=int(SOTA_ALIGNMENT_MOD),
                    t2_blocks=int(SOTA_T2_BLOCKS),
                    activation_delay_intervals=int(SOTA_ACTIVATION_DELAY_INTERVALS),
                    t2_intervals=SOTA_T2_INTERVALS,
                    min_t2_intervals=int(SOTA_MIN_T2_INTERVALS),
                )

                finalized_event = SOTAEvent(
                    miner_hotkey=vote.miner_hotkey,
                    score_int=score_int,
                    score=float(vote.score),
                    result_id=vote.result_id,
                    decision_block=int(decision_block),
                    start_block=int(start_block),
                    end_block=int(end_block),
                    alignment_mod=int(SOTA_ALIGNMENT_MOD),
                    t2_blocks=int(effective_t2_blocks),
                )
                db.add(finalized_event)

                sota_entry.sota_value = float(vote.score)
                sota_entry.updated_at = datetime.now(UTC)

            db.query(SOTAVote).delete()

        db.commit()

        if finalized_event is not None:
            db.refresh(finalized_event)
            return SOTAVoteResponse(
                status="finalized",
                votes_for_candidate=votes_for_candidate,
                votes_needed=SOTA_CONSENSUS_VOTES,
                current_sota=float(_get_or_create_sota_cache(db).sota_value),
                finalized_event=SOTAEventResponse.model_validate(finalized_event),
            )

        return SOTAVoteResponse(
            status="vote_updated" if existing_vote else "vote_recorded",
            votes_for_candidate=votes_for_candidate,
            votes_needed=SOTA_CONSENSUS_VOTES,
            current_sota=current_sota,
            finalized_event=None,
        )


@app.get("/sota/events", response_model=List[SOTAEventResponse])
async def list_sota_events(
    limit: int = 20,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(auth_handler.auth_decorator),
):
    limit = max(1, min(int(limit), 200))
    events_db = db.query(SOTAEvent).order_by(SOTAEvent.start_block.desc()).limit(limit).all()
    return [SOTAEventResponse.model_validate(e) for e in events_db]


@app.post("/submit_solution", response_model=dict)
async def submit_result(
    submission: ResultSubmission,
    db: Session = Depends(get_db),
    tracker: SubmissionTracker = Depends(get_submission_tracker),
    miner_hotkey: str = Depends(auth_handler.auth_miner_decorator),
    signature: str = Header(..., alias="X-Signature"),
    timestamp_message: str = Header(..., alias="X-Timestamp"),
):
    logger.info(
        f"RELAY: Received submission from miner {miner_hotkey[:8]}... with score {submission.score}"
    )

    current_time = datetime.now(UTC)

    if submission.score is None:
        raise HTTPException(status_code=400, detail="score is required")

    if not TEST_MODE:
        has_invitation = (
            db.query(InvitationCode)
            .filter(InvitationCode.miner_hotkey == miner_hotkey)
            .count()
            > 0
        )
        if not has_invitation:
            raise HTTPException(
                status_code=403, detail="Please register your invitation code first"
            )

    tracker.clean_old_entries(current_time, miner_hotkey)

    if tracker.check_rate_limit(miner_hotkey):
        logger.warning(f"Rate limit exceeded for miner {miner_hotkey[:8]}...")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {MAX_SUBMISSIONS_PER_HOUR} submissions per hour",
        )

    solution_hash = hashlib.sha256(
        json.dumps(submission.algorithm_result, sort_keys=True).encode()
    ).hexdigest()
    if tracker.is_duplicate(solution_hash):
        logger.warning(f"Duplicate submission from miner {miner_hotkey[:8]}...")
        raise HTTPException(
            status_code=400, detail="Duplicate solution submitted within 5 minutes"
        )
    tracker.add_submission(miner_hotkey, solution_hash, current_time)

    blacklist_votes = (
        db.query(BlacklistVote).filter(BlacklistVote.miner_hotkey == miner_hotkey).count()
    )
    if blacklist_votes >= CONCENSUS_BLACKLIST:
        raise HTTPException(status_code=403, detail="Miner is blacklisted")

    cached_sota = db.query(SOTACache).filter(SOTACache.task_type == "global").first()
    if cached_sota and float(submission.score) < float(cached_sota.sota_value):
        raise HTTPException(
            status_code=400,
            detail=f"Score {submission.score} is below the SOTA threshold of {cached_sota.sota_value}",
        )

    new_score = float(submission.score)

    # Neutral policy: keep exactly one submission per miner, always overwrite with latest.
    existing = (
        db.query(MiningResult)
        .filter(MiningResult.miner_hotkey == miner_hotkey)
        .order_by(MiningResult.timestamp.desc())
        .first()
    )
    if existing is not None:
        # Best-effort cleanup: remove any accidental duplicates from prior versions.
        db.query(MiningResult).filter(
            MiningResult.miner_hotkey == miner_hotkey,
            MiningResult.id != existing.id,
        ).delete(synchronize_session=False)

        existing.task_id = submission.task_id
        existing.algorithm_result = json.dumps(submission.algorithm_result)
        existing.score = new_score
        existing.signature = signature
        existing.timestamp_message = timestamp_message
        existing.timestamp = current_time
        db.commit()

        logger.info(
            "RELAY: Updated latest result %s from miner %s (score %.6f)",
            existing.id,
            miner_hotkey[:8],
            new_score,
        )
        return {"result_id": existing.id, "status": "accepted"}

    result_id = hashlib.sha256(
        f"{miner_hotkey}:{submission.task_id}:{int(time.time())}".encode()
    ).hexdigest()[:16]

    mining_result = MiningResult(
        id=result_id,
        miner_hotkey=miner_hotkey,
        task_id=submission.task_id,
        algorithm_result=json.dumps(submission.algorithm_result),
        score=new_score,
        signature=signature,
        timestamp_message=timestamp_message,
        timestamp=current_time,
    )
    db.add(mining_result)
    db.commit()

    logger.info(
        "RELAY: Stored result %s from miner %s (score %.6f)",
        result_id,
        miner_hotkey[:8],
        new_score,
    )
    return {"result_id": result_id, "status": "accepted"}


@app.post("/test/submit_solution", response_model=dict)
async def submit_test_result(
    submission: ResultSubmission,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(require_uid0_validator),
    signature: str = Header(..., alias="X-Signature"),
    timestamp_message: str = Header(..., alias="X-Timestamp"),
):
    """
    Accept a test submission signed by UID0's hotkey only.

    These submissions are NOT exposed via /results and are only retrievable by UID0 for logging.
    """

    current_time = datetime.now(UTC)
    submission_id = hashlib.sha256(
        f"test:{validator_hotkey}:{submission.task_id}:{int(time.time() * 1000)}".encode()
    ).hexdigest()[:16]

    test_row = TestSubmission(
        id=submission_id,
        submitter_hotkey=validator_hotkey,
        task_id=submission.task_id,
        algorithm_result=json.dumps(submission.algorithm_result),
        score=float(submission.score) if submission.score is not None else None,
        signature=signature,
        timestamp_message=timestamp_message,
        created_at=current_time,
    )
    db.add(test_row)
    db.commit()

    logger.info(
        "RELAY: Stored TEST submission %s from uid0=%s task=%s score=%s",
        submission_id,
        validator_hotkey[:8],
        submission.task_id,
        submission.score,
    )
    return {"test_submission_id": submission_id, "status": "accepted"}


@app.get("/test/submissions", response_model=List[TestSubmissionResponse])
async def pull_test_submissions(
    limit: int = 50,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(require_uid0_validator),
):
    """
    UID0-only queue for test submissions (claim-on-read).
    """

    limit = max(1, min(int(limit), 200))
    now = datetime.now(UTC)

    rows = (
        db.query(TestSubmission)
        .filter(TestSubmission.claimed_at.is_(None))
        .order_by(TestSubmission.created_at.asc())
        .limit(limit)
        .all()
    )
    if not rows:
        return []

    for row in rows:
        row.claimed_at = now
        row.claimed_by = validator_hotkey
    db.commit()

    return [TestSubmissionResponse.model_validate(row) for row in rows]


@app.get("/results/{miner_hotkey}", response_model=List[MiningResultResponse])
async def get_miner_results(
    miner_hotkey: str,
    limit: int = 256,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(auth_handler.auth_decorator),
):
    if not verify_hotkey(miner_hotkey):
        raise HTTPException(status_code=400, detail="Invalid hotkey format")

    # Relay keeps at most one submission per miner; return at most one row.
    results_db = (
        db.query(MiningResult)
        .filter(MiningResult.miner_hotkey == miner_hotkey)
        .order_by(MiningResult.timestamp.desc())
        .limit(1)
        .all()
    )
    results_db = _attach_coldkeys(results_db, db)
    return [MiningResultResponse.model_validate(result) for result in results_db]


@app.get("/results", response_model=List[MiningResultResponse])
async def get_all_results(
    limit: int = 256,
    task_id: Optional[str] = None,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(auth_handler.auth_decorator),
):
    logger.info(f"RELAY: Validator {validator_hotkey[:8]}... requesting {limit} results")

    if not verify_hotkey(validator_hotkey):
        raise HTTPException(status_code=400, detail="Invalid validator hotkey format")

    query = db.query(MiningResult)
    if task_id:
        query = query.filter(MiningResult.task_id == task_id)
    # Neutral ordering: newest first (miners could lie about score, so don't sort by it).
    results_db = query.order_by(MiningResult.timestamp.desc()).limit(limit).all()
    results_db = _attach_coldkeys(results_db, db)
    return [MiningResultResponse.model_validate(result) for result in results_db]


@app.get("/sota-events", response_model=PaginatedSOTAEvents)
async def get_sota_events(
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
):
    page = max(1, page)
    page_size = max(1, min(page_size, 100))

    offset = (page - 1) * page_size

    total = db.query(SOTAEvent).count()
    events_db = (
        db.query(SOTAEvent)
        .order_by(SOTAEvent.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    total_pages = (total + page_size - 1) // page_size

    return PaginatedSOTAEvents(
        results=[FrontendSOTAEvent.model_validate(event) for event in events_db],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )

@app.post("/blacklist/{miner_hotkey}")
async def blacklist_miner(
    miner_hotkey: str,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(auth_handler.auth_decorator),
):
    if not verify_hotkey(miner_hotkey):
        raise HTTPException(status_code=400, detail="Invalid miner hotkey format")

    existing_vote = (
        db.query(BlacklistVote)
        .filter(
            BlacklistVote.miner_hotkey == miner_hotkey,
            BlacklistVote.validator_hotkey == validator_hotkey,
        )
        .first()
    )
    if existing_vote:
        raise HTTPException(
            status_code=400,
            detail="Validator has already voted to blacklist this miner",
        )

    db.add(BlacklistVote(miner_hotkey=miner_hotkey, validator_hotkey=validator_hotkey))
    db.commit()
    return {"status": "blacklist vote recorded"}


@app.post("/invitation_code/generate/{count}")
async def generate_invitation_code_post(
    count: int,
    db: Session = Depends(get_db),
    is_admin: bool = Depends(auth_handler.auth_admin_role),
):
    invitation_codes = []
    invitation_code_models = []
    for _ in range(count):
        code = generate_invitation_code()
        invitation_codes.append(code)
        invitation_code_models.append(InvitationCode(id=code))
    db.add_all(invitation_code_models)
    db.commit()
    return {"data": invitation_codes}


@app.get("/invitation_code/list/{page}/{size}")
async def list_invitation_code(
    page: int,
    size: int,
    db: Session = Depends(get_db),
    is_admin: bool = Depends(auth_handler.auth_admin_role),
):
    count = db.query(InvitationCode).count()
    invitation_codes = (
        db.query(InvitationCode).offset((page - 1) * size).limit(size).all()
    )
    return {"data": invitation_codes, "count": count}


@app.get("/invitation_code/linked", response_model=dict)
async def get_invitation_code_linked_for_miner(
    db: Session = Depends(get_db),
    miner_hotkey: str = Depends(auth_handler.auth_miner_decorator),
):
    invitation_code = (
        db.query(InvitationCode)
        .filter(InvitationCode.miner_hotkey == miner_hotkey)
        .one_or_none()
    )
    return {"data": invitation_code.id if invitation_code else None}


@app.post("/invitation_code/link", response_model=dict)
async def lin_invitation_code(
    code: LinkInvitationCode,
    db: Session = Depends(get_db),
    miner_hotkey: str = Depends(auth_handler.auth_miner_decorator),
):
    if TEST_MODE and code.code == TEST_INVITE_CODE:
        existing = (
            db.query(InvitationCode)
            .filter(InvitationCode.id == TEST_INVITE_CODE)
            .one_or_none()
        )
        if existing is None:
            db.add(InvitationCode(id=TEST_INVITE_CODE))
            db.commit()

    update_data = {InvitationCode.miner_hotkey: miner_hotkey}
    if code.coldkey_address:
        update_data[InvitationCode.coldkey_address] = code.coldkey_address

    result = (
        db.query(InvitationCode)
        .filter(InvitationCode.id == code.code, InvitationCode.miner_hotkey == None)
        .update(update_data, synchronize_session=False)
    )
    db.commit()
    return {"data": result}


@app.post("/coldkey_address/update", response_model=dict)
async def update_coldkey_address(
    data: UpdateColdkeyAddress,
    db: Session = Depends(get_db),
    miner_hotkey: str = Depends(auth_handler.auth_miner_decorator),
):
    invitation = (
        db.query(InvitationCode)
        .filter(InvitationCode.miner_hotkey == miner_hotkey)
        .first()
    )

    if not invitation:
        if not TEST_MODE:
            raise HTTPException(
                status_code=404, detail="No invitation code linked to this miner"
            )

        code = TEST_INVITE_CODE
        if db.query(InvitationCode).filter(InvitationCode.id == code).count() > 0:
            for _ in range(25):
                candidate = generate_invitation_code()
                if db.query(InvitationCode).filter(InvitationCode.id == candidate).count() == 0:
                    code = candidate
                    break
        invitation = InvitationCode(id=code, miner_hotkey=miner_hotkey)
        db.add(invitation)

    invitation.coldkey_address = data.coldkey_address
    db.commit()
    return {
        "status": "success",
        "coldkey_address": data.coldkey_address,
        "invite_code": invitation.id,
    }


def _format_ts(ts: Optional[float]) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC).isoformat()
    except Exception:
        return "-"


@app.get("/admin/status")
async def admin_status(
    db: Session = Depends(get_db),
    _is_admin: bool = Depends(require_admin_dashboard_auth),
):
    db_ok = True
    db_error = None
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = f"{type(e).__name__}: {e}"

    snapshot = request_metrics.snapshot()
    uid0_hotkey = _resolve_uid0_hotkey()

    return {
        "status": "healthy" if db_ok else "unhealthy",
        "db_ok": db_ok,
        "db_error": db_error,
        "uid0_hotkey": uid0_hotkey,
        "auth": {
            "validator_cache_count": len(getattr(auth_handler, "valis", []) or []),
            "validator_cache_last_sync_ts": getattr(auth_handler, "last_sync_time", 0) or 0,
            "validator_cache_last_error": getattr(auth_handler, "_last_sync_error", None),
        },
        "metrics": snapshot,
    }


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    db: Session = Depends(get_db),
    _is_admin: bool = Depends(require_admin_dashboard_auth),
):
    status = await admin_status(db=db, _is_admin=True)

    refresh_s = max(1, int(ADMIN_DASHBOARD_REFRESH_SECONDS))
    db_ok = bool(status.get("db_ok"))
    db_error = str(status.get("db_error") or "")
    uid0_hotkey = str(status.get("uid0_hotkey") or "-")
    metrics = status.get("metrics") or {}

    last_request_ts = _format_ts(metrics.get("last_request_ts"))
    uptime_s = float(metrics.get("uptime_seconds") or 0.0)

    def _row(label: str, value: str) -> str:
        return (
            "<tr>"
            f"<th style='text-align:left;padding:4px 10px 4px 0'>{html.escape(label)}</th>"
            f"<td style='padding:4px 0'>{html.escape(value)}</td>"
            "</tr>"
        )

    status_rows = "".join(
        [
            _row("Service", "healthy" if db_ok else "unhealthy"),
            _row("DB", "ok" if db_ok else f"error: {db_error or '-'}"),
            _row("UID0 hotkey", uid0_hotkey),
            _row("Uptime", f"{uptime_s/60.0:.1f} minutes"),
            _row("Last request", last_request_ts),
        ]
    )

    last_1m = metrics.get("last_1m") or {}
    last_1h = metrics.get("last_1h") or {}
    last_24h = metrics.get("last_24h") or {}

    def _fmt_counts(bucket: dict, key: str) -> str:
        try:
            return str(int(bucket.get(key, 0) or 0))
        except Exception:
            return "0"

    requests_rows = "".join(
        [
            _row("Miner last 1m", _fmt_counts(last_1m, "miner")),
            _row("Validator last 1m", _fmt_counts(last_1m, "validator")),
            _row("5xx last 1m", _fmt_counts(last_1m, "errors_5xx")),
            _row("Miner last 1h", _fmt_counts(last_1h, "miner")),
            _row("Validator last 1h", _fmt_counts(last_1h, "validator")),
            _row("5xx last 1h", _fmt_counts(last_1h, "errors_5xx")),
            _row("Miner last 24h", _fmt_counts(last_24h, "miner")),
            _row("Validator last 24h", _fmt_counts(last_24h, "validator")),
            _row("5xx last 24h", _fmt_counts(last_24h, "errors_5xx")),
        ]
    )

    html_body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_s}">
  <title>Relay Admin Dashboard</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #eee; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h2>Relay Admin Dashboard</h2>
  <p class="muted">Auto-refresh: {refresh_s}s</p>

  <div class="grid">
    <div class="card">
      <h3>Status</h3>
      <table>
        {status_rows}
      </table>
    </div>
    <div class="card">
      <h3>Requests</h3>
      <table>
        {requests_rows}
      </table>
    </div>
  </div>
</body>
</html>"""

    return HTMLResponse(content=html_body)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)

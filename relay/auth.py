import asyncio
import logging
import os
import time
from typing import Optional
from substrateinterface import Keypair, KeypairType
from fastapi import Depends, HTTPException, Header
from relay.config import ADMIN_AUTH_TOKEN, RELAY_UID0_HOTKEY, TEST_MODE

logger = logging.getLogger(__name__)

try:
    import bittensor as bt  # type: ignore
    _BT_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # pragma: no cover
    bt = None
    _BT_IMPORT_ERROR = f"{type(e).__name__}: {e}"

class AuthHandler:
    HARDCODED_ALLOWED_VALIDATORS = {
        "5Ef5EsPQoMVmJ8rYectQ26BEvscvATEGm365bcQjo1Y6bxGr",
        "5FCFAQuCwnNN5VR9mo2sjfE9aAhz5YWaimv8VNxypiJwdYjX",
        "5F2SaUVxK3mb2WnzZiEaHLA3CssR9ATXfPfdkWzcfgaR9s94",
        "5C7eJLN6Wk1rh9oQoUmX4AZrFm9BWU6tRRserToeJ7fPsn94",
        "5C5ebCrgSFgKjAMaRkqZFpsgXmucnxkJZntsEugT5S3uDnzL",
        "5FWZX4iGnGPMPeGwyPCn7bQX5qN8BLcyXERzmqY3n7c6GGKc"
    }

    def __init__(
        self, contract_manager, netuid=0, network="finney"
    ):  # HINT you can either set the stake to be on subnet or on root !
        self.valis = set()
        self.netuid = netuid
        self.network = network
        self.last_sync_time = 0
        self.sync_delta = 600
        self.min_stake = 50000
        self.contract_manager = contract_manager
        self.sota_score = None
        self.sota_score_last_update = 0
        self._sync_lock = asyncio.Lock()
        self._last_sync_error: Optional[str] = None
        self.uid0_hotkey: Optional[str] = RELAY_UID0_HOTKEY or None
        self._bt_import_error: Optional[str] = _BT_IMPORT_ERROR
        logger.info(
            "Auth: initialized netuid=%s network=%s min_stake=%s hardcoded_valis=%s",
            self.netuid,
            self.network,
            self.min_stake,
            len(self.HARDCODED_ALLOWED_VALIDATORS),
        )
        if self.uid0_hotkey:
            logger.info("Auth: UID0 hotkey override configured x_key=%s", self.uid0_hotkey[:8])
        if self._bt_import_error and not TEST_MODE:
            logger.error("Auth: bittensor import failed: %s", self._bt_import_error)

    def should_resync(self):
        current_delta = time.time() - self.last_sync_time

        if current_delta > self.sync_delta:
            return True
        else:
            return False

    def cache_valis(self):
        if bt is None:
            raise RuntimeError(self._bt_import_error or "bittensor import failed")
        start = time.perf_counter()
        meta = bt.metagraph(netuid=self.netuid, network=self.network)

        valis = set()
        for hotkey, stake, permit in zip(meta.hotkeys, meta.stake, meta.validator_permit):
            if stake > self.min_stake and permit:
                valis.add(hotkey)

        self.valis = valis
        if not self.uid0_hotkey and getattr(meta, "hotkeys", None):
            try:
                self.uid0_hotkey = str(meta.hotkeys[0])
            except Exception:
                self.uid0_hotkey = None
        self.last_sync_time = time.time()
        self._last_sync_error = None
        logger.info(
            "Auth: synced validators netuid=%s network=%s count=%s ms=%.1f",
            self.netuid,
            self.network,
            len(self.valis),
            (time.perf_counter() - start) * 1000.0,
        )
        if self.uid0_hotkey:
            logger.info("Auth: cached UID0 hotkey x_key=%s", self.uid0_hotkey[:8])

    def check_if_vali_ss58(self, ss58_addr):
        if TEST_MODE:
            logger.debug("Auth: TEST_MODE enabled; accepting validator %s", (ss58_addr or "")[:8] or None)
            return True
        is_hardcoded = ss58_addr in self.HARDCODED_ALLOWED_VALIDATORS
        is_cached = ss58_addr in self.valis
        if is_hardcoded:
            logger.debug("Auth: validator %s matched hardcoded whitelist", ss58_addr[:8])
        elif is_cached:
            logger.debug("Auth: validator %s matched cached list", ss58_addr[:8])
        else:
            cache_age = time.time() - self.last_sync_time
            logger.warning(
                "Auth: validator %s NOT found (cache_age=%.1fs cached_count=%s hardcoded_count=%s)",
                ss58_addr[:8],
                cache_age,
                len(self.valis),
                len(self.HARDCODED_ALLOWED_VALIDATORS),
            )
        return is_hardcoded or is_cached

    def check_auth(self, public_address, message, signature):
        if not self.check_if_vali_ss58(public_address):
            logger.warning("Auth: rejected - not a valid validator x_key=%s", public_address[:8])
            return False

        try:
            signature_bytes = (
                bytes.fromhex(signature) if isinstance(signature, str) else signature
            )
            keypair = Keypair(
                ss58_address=public_address,
                crypto_type=KeypairType.SR25519,
            )
            is_valid = keypair.verify(message.encode("utf-8"), signature_bytes)
            if is_valid:
                logger.debug("Auth: signature verified x_key=%s", public_address[:8])
            else:
                logger.warning("Auth: signature invalid x_key=%s", public_address[:8])
            return is_valid
        except Exception:
            logger.warning("Auth: signature verify failed x_key=%s", (public_address or "")[:8] or None)
            return False

    def check_miner_auth(self, public_address, message, signature):
        try:
            signature_bytes = (
                bytes.fromhex(signature) if isinstance(signature, str) else signature
            )
            keypair = Keypair(
                ss58_address=public_address,
                crypto_type=KeypairType.SR25519,
            )
            is_valid = keypair.verify(message.encode("utf-8"), signature_bytes)
            return is_valid
        except Exception:
            logger.exception("Miner auth error x_key=%s", (public_address or "")[:8] or None)
            return False

    async def refresh_valis_if_needed(self) -> None:
        if TEST_MODE:
            return
        if bt is None:
            self._last_sync_error = self._bt_import_error or "bittensor import failed"
            raise RuntimeError(self._last_sync_error)
        if not self.should_resync():
            return

        timeout_sec = float(os.getenv("RELAY_AUTH_SYNC_TIMEOUT_SEC", "30"))
        cache_age = time.time() - self.last_sync_time

        logger.info(
            "Auth: starting metagraph sync (cache_age=%.1fs timeout=%ss)",
            cache_age,
            timeout_sec,
        )

        async with self._sync_lock:
            if not self.should_resync():
                return
            try:
                await asyncio.wait_for(asyncio.to_thread(self.cache_valis), timeout=timeout_sec)
                logger.info("Auth: metagraph sync completed successfully")
            except asyncio.TimeoutError:
                self._last_sync_error = f"validator sync timeout after {timeout_sec}s"
                logger.error(
                    "Auth: metagraph sync TIMEOUT after %ss (cache_age=%.1fs cached_count=%s)",
                    timeout_sec,
                    cache_age,
                    len(self.valis),
                )
                raise
            except Exception as e:
                self._last_sync_error = f"validator sync failed: {type(e).__name__}"
                logger.exception(
                    "Auth: metagraph sync FAILED %s (cache_age=%.1fs cached_count=%s)",
                    type(e).__name__,
                    cache_age,
                    len(self.valis),
                )
                raise

    def get_sota_score(self):
        current_time = time.time()
        if self.sota_score is None or (
            current_time - self.sota_score_last_update > 600
        ):
            if self.contract_manager is None:
                raise RuntimeError("Contract manager not configured for SOTA fetch")
            self.sota_score = self.contract_manager.get_current_sota_threshold()
            self.sota_score_last_update = current_time
        return self.sota_score

    async def auth_decorator(
        self,
        public_address: str = Header(..., alias="X-Key"),
        message: str = Header(..., alias="X-Timestamp"),
        signature: str = Header(..., alias="X-Signature"),
    ):

        try:
            await self.refresh_valis_if_needed()
        except Exception:
            raise HTTPException(
                status_code=503,
                detail=self._last_sync_error or "Authentication backend unavailable",
            )

        try:
            is_valid = self.check_auth(public_address, message, signature)
        except Exception:
            logger.exception("Auth: internal error during auth x_key=%s", public_address[:8])
            raise HTTPException(status_code=503, detail="Authentication backend unavailable")

        if not is_valid:
            logger.warning("Auth: failed x_key=%s", public_address[:8])
            raise HTTPException(status_code=401, detail="Authentication failed")

        return (
            public_address  # Or whatever you want to return to the decorated function
        )

    async def auth_miner_decorator(
        self,
        public_address: str = Header(..., alias="X-Key"),
        message: str = Header(..., alias="X-Timestamp"),
        signature: str = Header(..., alias="X-Signature"),
    ):

        try:
            is_valid = self.check_miner_auth(public_address, message, signature)
        except Exception:
            logger.exception("Auth: internal error during miner auth x_key=%s", public_address[:8])
            raise HTTPException(status_code=503, detail="Authentication backend unavailable")

        if not is_valid:
            logger.warning("Auth: miner auth failed x_key=%s", public_address[:8])
            raise HTTPException(
                status_code=401, detail="Authentication failed for miner"
            )

        return public_address
    
    async def auth_admin_role(
        self,
        auth_token: str = Header(..., alias="X-Auth-Token"),
    ):

        is_valid = auth_token != "" and auth_token == ADMIN_AUTH_TOKEN

        if not is_valid:
            raise HTTPException(
                status_code=401, detail="Authentication failed for admin"
            )

        return is_valid

    def get_uid0_hotkey(self) -> Optional[str]:
        return self.uid0_hotkey

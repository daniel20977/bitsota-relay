from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

class ResultSubmission(BaseModel):
    task_id: str
    algorithm_result: Dict[str, Any]
    score: Optional[float] = None


class MiningResultResponse(BaseModel):
    id: str
    miner_hotkey: str
    task_id: str
    algorithm_result: str
    score: Optional[float]
    timestamp: datetime
    validator_verified: Optional[str]
    signature: Optional[str]
    timestamp_message: Optional[str]
    coldkey_address: Optional[str] = None

    class Config:
        from_attributes = True


class TestSubmissionResponse(BaseModel):
    id: str
    submitter_hotkey: str
    task_id: str
    algorithm_result: str
    score: Optional[float]
    created_at: datetime
    claimed_at: Optional[datetime] = None
    claimed_by: Optional[str] = None
    signature: Optional[str]
    timestamp_message: Optional[str]

    class Config:
        from_attributes = True

class LinkInvitationCode(BaseModel):
    code: str
    coldkey_address: Optional[str] = None

class UpdateColdkeyAddress(BaseModel):
    coldkey_address: str

class SOTAResponse(BaseModel):
    task_type: str
    sota_threshold: float
    updated_at: datetime
    cached: bool

    class Config:
        from_attributes = True


class SOTAVoteRequest(BaseModel):
    miner_hotkey: str
    score: float
    seen_block: int
    result_id: Optional[str] = None


class SOTAEventResponse(BaseModel):
    id: int
    miner_hotkey: str
    score: float
    score_int: int
    result_id: Optional[str]
    decision_block: int
    start_block: int
    end_block: int
    alignment_mod: int
    t2_blocks: int
    created_at: datetime

    class Config:
        from_attributes = True


class SOTAVoteResponse(BaseModel):
    status: str
    votes_for_candidate: int
    votes_needed: int
    current_sota: float
    finalized_event: Optional[SOTAEventResponse] = None

class FrontendSOTAInfo(BaseModel):
    task_type: str
    sota_value: float
    updated_at: datetime

    class Config:
        from_attributes = True

class FrontendSOTAEvent(BaseModel):
    miner_hotkey: str
    score: float
    created_at: datetime

    class Config:
        from_attributes = True

class PaginatedSOTAEvents(BaseModel):
    results: List[FrontendSOTAEvent]
    total: int
    page: int
    page_size: int
    total_pages: int

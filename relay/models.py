from sqlalchemy import Column, String, DateTime, Text, Float, Integer, UniqueConstraint
from datetime import datetime, UTC
from relay.config import Base

class MiningResult(Base):
    __tablename__ = "mining_results"

    id = Column(String, primary_key=True)
    miner_hotkey = Column(String(48), nullable=False, index=True)
    task_id = Column(String, nullable=False)
    algorithm_result = Column(Text, nullable=False)
    score = Column(Float)
    timestamp = Column(DateTime, default=datetime.now(UTC))
    validator_verified = Column(String(48))
    # Fields for signature verification
    signature = Column(String, nullable=True)
    timestamp_message = Column(String, nullable=True)


class TestSubmission(Base):
    __tablename__ = "test_submissions"

    id = Column(String, primary_key=True)
    submitter_hotkey = Column(String(48), nullable=False, index=True)
    task_id = Column(String, nullable=False)
    algorithm_result = Column(Text, nullable=False)
    score = Column(Float)
    created_at = Column(DateTime, default=datetime.now(UTC), index=True)
    claimed_at = Column(DateTime, nullable=True, index=True)
    claimed_by = Column(String(48), nullable=True, index=True)
    # Fields for signature verification / audit logging
    signature = Column(String, nullable=True)
    timestamp_message = Column(String, nullable=True)


class BlacklistVote(Base):
    __tablename__ = "blacklist_votes"

    id = Column(Integer, primary_key=True, index=True)
    miner_hotkey = Column(String(48), nullable=False, index=True)
    validator_hotkey = Column(String(48), nullable=False)
    timestamp = Column(DateTime, default=datetime.now(UTC))

class InvitationCode(Base):
    __tablename__ = "invitation_codes"

    id = Column(String(8), primary_key=True, index=True)
    miner_hotkey = Column(String(48), nullable=True, unique=True)
    coldkey_address = Column(String(48), nullable=True)
    timestamp = Column(DateTime, default=datetime.now(UTC))

class SOTACache(Base):
    __tablename__ = "sota_cache"

    id = Column(Integer, primary_key=True, index=True)
    task_type = Column(String(64), nullable=False, unique=True, index=True)
    sota_value = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=datetime.now(UTC), onupdate=datetime.now(UTC))


class SOTAVote(Base):
    __tablename__ = "sota_votes"

    id = Column(Integer, primary_key=True, index=True)
    validator_hotkey = Column(String(48), nullable=False, index=True)
    miner_hotkey = Column(String(48), nullable=False, index=True)
    score_int = Column(Integer, nullable=False, index=True)
    score = Column(Float, nullable=False)
    result_id = Column(String, nullable=True)
    seen_block = Column(Integer, nullable=False, default=0)
    timestamp = Column(DateTime, default=datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint("validator_hotkey", name="uq_sota_votes_validator_round"),
    )


class SOTAEvent(Base):
    __tablename__ = "sota_events"

    id = Column(Integer, primary_key=True, index=True)
    miner_hotkey = Column(String(48), nullable=False, index=True)
    score_int = Column(Integer, nullable=False, index=True)
    score = Column(Float, nullable=False)
    result_id = Column(String, nullable=True)

    decision_block = Column(Integer, nullable=False)
    start_block = Column(Integer, nullable=False, index=True)
    end_block = Column(Integer, nullable=False)
    alignment_mod = Column(Integer, nullable=False)
    t2_blocks = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint("miner_hotkey", "score_int", name="uq_sota_events_candidate"),
    )

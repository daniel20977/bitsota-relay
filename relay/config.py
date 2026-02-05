import os

from sqlalchemy.engine.create import create_engine
from sqlalchemy.orm.decl_api import declarative_base
from sqlalchemy.orm.session import sessionmaker


# Environment variable validation
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
ADMIN_AUTH_TOKEN = os.getenv("ADMIN_AUTH_TOKEN", "")
if not ADMIN_AUTH_TOKEN:
    raise ValueError("ADMIN_AUTH_TOKEN environment variable is required")

RPC_URL = os.getenv("RPC_URL", "https://test.chain.opentensor.ai")

# Contract configuration (optional; required only for capacitor-based deployments)
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
CONTRACT_ABI = os.getenv(
    "CONTRACT_ABI",
    [
        {
            "inputs": [
                {"internalType": "address", "name": "_t1", "type": "address"},
                {"internalType": "address", "name": "_t2", "type": "address"},
                {"internalType": "address", "name": "_t3", "type": "address"},
            ],
            "stateMutability": "payable",
            "type": "constructor",
        },
        {
            "anonymous": False,
            "inputs": [
                {
                    "indexed": False,
                    "internalType": "bytes32",
                    "name": "recipient",
                    "type": "bytes32",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "amount",
                    "type": "uint256",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "score",
                    "type": "uint256",
                },
            ],
            "name": "Approved",
            "type": "event",
        },
        {"anonymous": False, "inputs": [], "name": "ContractBurned", "type": "event"},
        {
            "anonymous": False,
            "inputs": [
                {
                    "indexed": False,
                    "internalType": "bool",
                    "name": "paused",
                    "type": "bool",
                }
            ],
            "name": "ContractPaused",
            "type": "event",
        },
        {
            "anonymous": False,
            "inputs": [
                {
                    "indexed": False,
                    "internalType": "bytes32",
                    "name": "recipient",
                    "type": "bytes32",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "amount",
                    "type": "uint256",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "score",
                    "type": "uint256",
                },
                {
                    "indexed": False,
                    "internalType": "bool",
                    "name": "ok",
                    "type": "bool",
                },
                {
                    "indexed": False,
                    "internalType": "bytes",
                    "name": "ret",
                    "type": "bytes",
                },
            ],
            "name": "PayoutAttempt",
            "type": "event",
        },
        {
            "anonymous": False,
            "inputs": [
                {
                    "indexed": True,
                    "internalType": "address",
                    "name": "trustee",
                    "type": "address",
                },
                {
                    "indexed": False,
                    "internalType": "bytes32",
                    "name": "recipient",
                    "type": "bytes32",
                },
                {
                    "indexed": False,
                    "internalType": "uint256",
                    "name": "score",
                    "type": "uint256",
                },
            ],
            "name": "VoteCast",
            "type": "event",
        },
        {"anonymous": False, "inputs": [], "name": "VoteReset", "type": "event"},
        {
            "inputs": [],
            "name": "REQUIRED_VOTES",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "attemptPayout",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "burn",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "burned",
            "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [{"internalType": "address", "name": "addr", "type": "address"}],
            "name": "checkAddressStatus",
            "outputs": [
                {"internalType": "bool", "name": "isTrusteeAddress", "type": "bool"},
                {
                    "internalType": "bool",
                    "name": "hasVotedInCurrentRound",
                    "type": "bool",
                },
                {"internalType": "address", "name": "a1", "type": "address"},
                {"internalType": "address", "name": "a2", "type": "address"},
                {"internalType": "address", "name": "a3", "type": "address"},
            ],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "deployer",
            "outputs": [{"internalType": "address", "name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "getBalance",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "getVotingStatus",
            "outputs": [
                {
                    "internalType": "bytes32",
                    "name": "currentRecipient",
                    "type": "bytes32",
                },
                {"internalType": "uint256", "name": "currentScore", "type": "uint256"},
                {
                    "internalType": "uint256",
                    "name": "currentVoteCount",
                    "type": "uint256",
                },
                {"internalType": "uint256", "name": "votesNeeded", "type": "uint256"},
            ],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [{"internalType": "address", "name": "", "type": "address"}],
            "name": "hasVoted",
            "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [{"internalType": "address", "name": "a", "type": "address"}],
            "name": "isTrustee",
            "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "latestScore",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "paused",
            "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "pendingRecipient",
            "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "pendingScore",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "queuedAmount",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "queuedRecipient",
            "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "queuedScore",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [
                {"internalType": "bytes32", "name": "recipient", "type": "bytes32"},
                {"internalType": "uint256", "name": "newScore", "type": "uint256"},
            ],
            "name": "releaseReward",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [{"internalType": "bool", "name": "p", "type": "bool"}],
            "name": "setPaused",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "trustee1",
            "outputs": [{"internalType": "address", "name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "trustee2",
            "outputs": [{"internalType": "address", "name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "trustee3",
            "outputs": [{"internalType": "address", "name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [],
            "name": "voteCount",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {"stateMutability": "payable", "type": "receive"},
    ],
)
CONCENSUS_BLACKLIST = int(os.getenv("CONCENSUS_BLACKLIST", "2"))

# SOTA voting / scheduling (capacitorless)
SOTA_CONSENSUS_VOTES = int(os.getenv("SOTA_CONSENSUS_VOTES", "2"))
SOTA_ALIGNMENT_MOD = int(os.getenv("SOTA_ALIGNMENT_MOD", "100"))
SOTA_T2_BLOCKS = int(os.getenv("SOTA_T2_BLOCKS", "1000"))
SOTA_ACTIVATION_DELAY_INTERVALS = int(
    os.getenv("SOTA_ACTIVATION_DELAY_INTERVALS", "1")
)
SOTA_MIN_T2_INTERVALS = int(os.getenv("SOTA_MIN_T2_INTERVALS", "2"))
_raw_t2_intervals = os.getenv("SOTA_T2_INTERVALS", "").strip()
SOTA_T2_INTERVALS = int(_raw_t2_intervals) if _raw_t2_intervals else None
DEFAULT_SOTA_THRESHOLD = float(os.getenv("DEFAULT_SOTA_THRESHOLD", "0.0"))
TEST_MODE = os.getenv("BITSOTA_TEST_MODE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
TEST_INVITE_CODE = os.getenv("BITSOTA_TEST_INVITE_CODE", "TESTTEST1")

# Optional: restrict test-submission endpoints to a specific validator hotkey.
# If unset, the relay will attempt to derive UID0 from the metagraph cache.
RELAY_UID0_HOTKEY = os.getenv("RELAY_UID0_HOTKEY", "").strip()

# Optional: admin dashboard basic auth (defaults to ADMIN_AUTH_TOKEN as password).
ADMIN_DASHBOARD_USERNAME = os.getenv("ADMIN_DASHBOARD_USERNAME", "admin").strip() or "admin"
ADMIN_DASHBOARD_PASSWORD = os.getenv("ADMIN_DASHBOARD_PASSWORD", "").strip() or ADMIN_AUTH_TOKEN
ADMIN_DASHBOARD_REFRESH_SECONDS = int(os.getenv("ADMIN_DASHBOARD_REFRESH_SECONDS", "5"))

MAX_SUBMISSIONS_PER_HOUR = 9999999999999999999999
DUPLICATE_TIME_WINDOW = 300

# Database setup
Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    pool_size=25,
    max_overflow=40,
    pool_timeout=120,
    pool_pre_ping=True,
    pool_recycle=1800,
    echo_pool=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

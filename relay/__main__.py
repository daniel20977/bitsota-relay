import argparse
import os
import re
import tempfile
from pathlib import Path


def _default_test_database_url() -> str:
    """
    Pick a writable SQLite database location for local `--test` runs.

    We avoid a shared fixed filename (e.g. /tmp/bitsota_relay_test.db) because it can
    become root-owned or otherwise unwritable and then break local development with:
    "sqlite3.OperationalError: attempt to write a readonly database".
    """

    ident: str
    try:
        ident = str(os.getuid())
    except Exception:
        ident = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    ident = re.sub(r"[^A-Za-z0-9_.-]+", "_", ident).strip("._-") or "user"

    candidates: list[Path] = []
    try:
        candidates.append(Path(tempfile.gettempdir()) / f"bitsota_relay_test_{ident}.db")
    except Exception:
        pass
    try:
        candidates.append(Path.home() / ".bitsota" / f"relay_test_{ident}.db")
    except Exception:
        pass
    candidates.append(Path.cwd() / "bitsota_relay_test.db")

    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not os.access(path, os.W_OK):
                continue
            if not path.exists() and not os.access(path.parent, os.W_OK):
                continue
            return f"sqlite:///{path.resolve().as_posix()}"
        except Exception:
            continue

    return "sqlite:///./bitsota_relay_test.db"


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8002, type=int)
    parser.add_argument("--test", action="store_true", help="Enable local test mode.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy DATABASE_URL (overrides env).",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help="ADMIN_AUTH_TOKEN (overrides env).",
    )
    parser.add_argument(
        "--test-invite-code",
        default=None,
        help="Invite code accepted in test mode (overrides env).",
    )
    args = parser.parse_args()

    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    if args.admin_token:
        os.environ["ADMIN_AUTH_TOKEN"] = args.admin_token

    if args.test:
        os.environ.setdefault("BITSOTA_TEST_MODE", "1")
        os.environ.setdefault("ADMIN_AUTH_TOKEN", "dev")
        os.environ.setdefault("DATABASE_URL", _default_test_database_url())
        if args.test_invite_code:
            os.environ["BITSOTA_TEST_INVITE_CODE"] = args.test_invite_code

    import uvicorn

    uvicorn.run("relay.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

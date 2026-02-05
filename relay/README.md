# Relay service

The relay is a FastAPI service that coordinates:

- miner submissions
- validator polling and re-evaluation
- validator voting to finalize SOTA events

## Run locally

Install dependencies from the repo root:

```bash
python3 -m pip install -r requirements.txt -r relay/requirements.txt
```

Start the relay in local test mode:

```bash
SOTA_CONSENSUS_VOTES=1 SOTA_ALIGNMENT_MOD=1 python3 -m relay --test --host 127.0.0.1 --port 8002
```

OpenAPI:

- `http://127.0.0.1:8002/docs`

Basic checks:

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/sota_threshold
```

Admin dashboard (basic auth in browser):

- URL: `http://127.0.0.1:8002/admin/dashboard`
- Default credentials in `--test` mode: `admin` / `dev`

Test submission queue (UID0-only in production):

- `POST /test/submit_solution`
- `GET /test/submissions`

## Configuration

Key environment variables:

- `DATABASE_URL` and `ADMIN_AUTH_TOKEN` are required in non-test mode
- `BITSOTA_TEST_MODE=1` enables test mode
- `SOTA_CONSENSUS_VOTES` sets how many validator votes are needed to finalize

## Auth model

- Miner endpoints: `X-Key`, `X-Timestamp`, `X-Signature`
- Validator endpoints: `X-Key`, `X-Timestamp`, `X-Signature` and validator allowlist checks
- Admin endpoints: `X-Auth-Token`

In test mode, validator allowlist checks are disabled.

#!/bin/sh
set -eu
# tmpfs is discarded on every container restart. Never reuse a user workspace here.
python -m app.public_demo_seed
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --no-proxy-headers

# Contributing

Use a branch and open a pull request against `main`. For a bug report, include the commit,
enabled services, reproduction steps and actual response. Use public samples or synthetic
fixtures; do not attach private footage, credentials or model checkpoints.

## CPU development environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.test.txt
cp .env.example .env
.venv/bin/python -m app.preflight
.venv/bin/python -m pytest -q -ra
npm --prefix frontend ci
npm --prefix frontend run test:search
npm --prefix frontend run test:observations
npm --prefix frontend run test:reid
npm --prefix frontend run test:api
npm --prefix frontend run build
```

Use Python 3.11 or 3.12 and Node.js 22.18+. The CPU suite uses SQLite and test doubles for
external services. The vendored ReID preprocessing tests skip without Torch; the real
Ultralytics import check skips without that optional runtime. A green CPU job does not verify
GPU inference, model quality or a live Milvus/embedding/ReID deployment.

`requirements.dev.txt` is the existing heavier AGX/model development environment; it is not
required for the default HOG workflow. Optional model setup and licenses are described in the
[deployment guide](docs/deployment.md) and [license scope](LICENSE_SCOPE.md).

## Validation and CI

Repository checks run CPU tests on Python 3.11/3.12, frontend tests and a build, API call
compatibility against the current checkout, and a sample-only demo container smoke check.
Model installation and downloads must not occur while handling an API request. Add a regression
at the affected boundary when changing behavior; use existing tests when they cover it.

To check API calls locally, export the spec without relying on an already running instance:

```bash
.venv/bin/python -c 'import json; from main import app; print(json.dumps(app.openapi()))' > /tmp/sightindex-openapi.json
node frontend/scripts/check-api.mjs /tmp/sightindex-openapi.json
```

Keep frontend changes consistent with the existing console and check both desktop and mobile.
Update the English and Chinese guides when changing installation, configuration or API behavior.
Check third-party notices before adding code, assets or model dependencies; the root Apache-2.0
license does not replace the separate licenses on vendored research code or checkpoints.

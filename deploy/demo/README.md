# Public sample demo

This is a separate, anonymous **read-only** deployment of the real Vue console and Python search
API. It loads a reviewed snapshot of one bundled NASA photograph: actual OpenCV HOG crops and
precomputed CV tone labels from a local pose model. Visitors search those stored labels and
inspect the records through the real API. No inference is run for a visitor. The sample
contains four people and this configuration detects two boxes; it is not a benchmark. See
[sample attribution](../../samples/README.md).

It does **not** demonstrate semantic retrieval, face recognition, ReID, live cameras or a paid
model. It accepts no user uploads, chat messages, camera URLs or data edits. It is not an
isolated workspace for every visitor: everyone browses the same public sample. Normal self-hosted
deployments keep their existing routes and authentication.

## Deploy

Use a reviewed source checkout, Docker Compose, a free loopback port and a TLS proxy. No keys,
model weights, GPUs or existing customer data are required. This CPU image omits the optional model
dependencies and builds the console with `VITE_PUBLIC_DEMO=true`.

```sh
SOURCE_REVISION=$(git rev-parse HEAD) docker compose -f deploy/demo/compose.yaml build
docker compose -f deploy/demo/compose.yaml up -d --wait --wait-timeout 120
curl --fail http://127.0.0.1:18081/health
```

The default public URL is `https://sightindex.otterview.ai`. Override `SIGHTINDEX_DEMO_URL` when
using another host. Add an A record for the host, then append the separate site in
[`Caddyfile`](Caddyfile) to the existing edge configuration. Back up, validate the **entire**
config and reload, preserving all other sites. Do not publish port 8000 directly.

The API is restricted by `PublicDemoGuard`, independently of hidden frontend buttons. Its
default-deny allowlist permits sample reads and exactly one bounded search POST. New API routes
stay blocked. Search bodies are limited to 4 KB, query strings to 2 KB, API/page traffic to a
shared budget of 600 requests/minute, and concurrent admitted API/page requests to eight.
This is a modest demonstration with shared capacity, not a load-tested service or a per-user
quota system. Static assets and health probes do not consume that budget.

The one-worker container runs as UID 1000 with a read-only image, dropped capabilities, 512 MB
memory limit, limited CPU, an internal Docker network without internet egress, and disposable
tmpfs storage. A separate 64 MB ingress proxy connects this internal network to the published
loopback port; the application itself has no external network. This is needed because Docker
does not publish ports on an internal-only network. Every restart reloads the sample snapshot
after checking its media hashes. No visitor files or prompts persist. Snapshot provenance is
in `samples/demo/manifest.json`.
Do not attach a production database or media directory. The seed command refuses an existing
database instead of replacing it.

## Verify and update

Run `pytest tests/test_public_demo.py` and the ordinary API regression tests. Build both the
normal frontend and the demo frontend. In a browser, verify initial crops, a `深色` search, an
empty `浅色上衣` search, observation records, original/crop images, desktop/mobile widths, and
links back to GitHub. Verify that direct upload/delete/stream/model requests return 403 and leave
counts unchanged. Health alone does not prove that the demonstration works.

Record the source SHA, image digest, proxy backup, URL, container health and browser results in
a private deployment manifest without secrets. Upgrade or roll back only this Compose project;
keep the edge site's upstream port stable and do not restart other applications. Demo data is
disposable, so no database migration is required.

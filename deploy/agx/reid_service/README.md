# ReID service (SapiensID)

Person re-identification over HTTP. Runs beside the API rather than inside it: the backbone is
412M parameters and 1.5GB resident.

## What it is

[SapiensID](https://arxiv.org/pdf/2504.04708) (CVPR 2025), a whole-body human recognition model
that holds up under clothing change — unlike a generic CLIP/VL embedding, which encodes *what a
person looks like* ("someone in a red jacket") rather than *who they are*.

The chain, per image:

| stage | weight | role |
| --- | --- | --- |
| YOLO-pose | `~/.cache/yolov8n-pose.pt`, 6.5MB | 17 body keypoints |
| DFA face detector | vendored, 28MB | face keypoints |
| SapiensID ViT | `data/models/sapiensid_*`, 1.5GB | 4096-d identity vector |

Vectors come back L2-normalised, so a dot product is cosine similarity.

## Weights

The 1.5GB checkpoints are not in git. Obtain them from the
[upstream project](https://github.com/mk-minchul/sapiensid) under the terms published there, then
place one under `data/models/`:

```
data/models/sapiensid_wb12m/
  model.yaml
  model.pth
```

The immutable revision covers the complete inference asset manifest, not only the backbone:

```text
model.pth
model.yaml
aligners/configs/yolo_dfa.yaml
aligners/.../dfa_mobilenetv4_medium/mobilenetv4_Final.pth
~/.cache/yolov8n-pose.pt
```

`wb12m` is trained on ~3x the data of `wb4m` and is the default. Point `REID_CHECKPOINT_DIR`
elsewhere to switch. Pre-seed the exact YOLO pose asset at `~/.cache/yolov8n-pose.pt` before
starting the service. Startup intentionally does not auto-download it: that file participates in
the immutable pipeline revision, so readiness must never depend on an unpinned first-use fetch.

The 28MB face-detector weight *is* committed. It ships inside the upstream source tree with no
separate download URL, so vendoring it keeps the service runnable the moment the big checkpoint
lands.

## Vendored source

`sapiensid/` is an inference subset of
[`mk-minchul/sapiensid`](https://github.com/mk-minchul/sapiensid), kept at the upstream directory layout
because the code resolves its bundled weights relative to its own root. Dropped: the WebBody
dataset tooling and the validation-set harness (26MB), neither of which inference touches.

The vendored directory retains the upstream CC BY-NC 4.0 license. See the repository-level
`THIRD_PARTY_NOTICES.md` before using or redistributing it. The license is non-commercial and does
not grant unrestricted production use.

One upstream line is patched: `aligners/keypoint_predictor/__init__.py` guarded its
no-detection fallback with `len(_keypoints[0]) > 0`, which indexes the very tensor that is empty
when YOLO finds no person. The fallback was therefore unreachable and a single person-less crop
raised `IndexError` for its entire batch. The guard now tests `len(_keypoints)` first.

Upstream's `requirements.txt` lists 25 packages including mxnet 1.9.1, lightning and deepspeed.
Inference needs none of those — only `torch torchvision ultralytics timm transformers omegaconf`
plus numpy/pandas/PIL. mxnet appears solely in the dataset builder.

## Running

```bash
bash deploy/agx/start_reid_service.sh
curl localhost:18031/health
curl localhost:18031/ready
```

The service hashes the backbone, model config, aligner config, DFA face aligner, and YOLO pose
weights during warmup and reports a composite
`checkpoint_revision`. Configure the API's `REID_CHECKPOINT_REVISION` to that exact value. A
different weight file, model config, dimension, or preprocessing version makes readiness fail and
selects a separate Milvus collection instead of mixing incompatible vectors.

`REID_WARMUP_ON_START=true` (the default) loads and hashes the model in a background thread. During
warmup `/health` remains a cheap liveness endpoint while `/ready` returns 503. If startup warmup is
disabled, the first `/ready` request starts the same one-shot background load and returns 503 until
it completes; readiness therefore cannot remain permanently cold. `REID_DEVICE` overrides device
selection (cuda → mps → cpu). Note the bundled aligner checkpoints
were serialised on CUDA, so the server maps them onto the resolved device explicitly; without that
a CPU-only host cannot load them at all.

## API

```
POST /embed        {image_base64 | image_url}  -> {embedding: [4096], dim, device, elapsed_seconds}
POST /embed-batch  {images: [...]}             -> {embeddings: [[4096], ...]}
GET  /health                                   -> liveness + warmup state
GET  /ready                                    -> loaded model + complete immutable identity
```

`REID_SERVICE_API_KEY` enables `Authorization: Bearer` checking when set.

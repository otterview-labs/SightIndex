# First run: upload, process and inspect

[English](first-run.md) · [简体中文](first-run.zh-CN.md) · [Project README](../README.md)

This guide uses the default local configuration. It verifies ingestion and processing without
enabling external model services. It does not demonstrate semantic search quality or person identity.

## Start the service

Follow the [README quick start](../README.md#quick-start), then open `http://127.0.0.1:8000/`.
The default `.env.example` uses SQLite and the OpenCV HOG person detector. Milvus, VLM,
semantic search and ReID are disabled.

Check the process:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/api/media/counts
```

The media counts endpoint reports images with crops and person crops, not every uploaded image.
A fresh installation can return zero counts.

## Upload, process and inspect

Use a JPEG or PNG you are allowed to process. An image with clearly visible people is useful
for trying the default detector. The repository does not bundle a public sample dataset.

1. Upload an image using the console or the request below.
2. Copy the returned `id`; uploading creates a media record, while person detection is a separate step.
3. Process that image and inspect the returned crop list.
4. Return to the console or list the crops to compare them with the original image.

```bash
curl --fail -X POST http://127.0.0.1:8000/api/images/upload \
  -F 'file=@/path/to/your-image.jpg'
```

Replace `IMAGE_ID` with the UUID returned by upload:

```bash
curl --fail -X POST http://127.0.0.1:8000/api/images/IMAGE_ID/process
curl --fail 'http://127.0.0.1:8000/api/person-crops?image_id=IMAGE_ID'
curl --fail http://127.0.0.1:8000/api/images/IMAGE_ID
```

The paths and ID above are placeholders. If you enable Basic Auth, authenticate these requests
and image downloads using your own configured account.

An empty crop list means the detector produced no person crops. It does not indicate whether
the uploaded image record exists. Check the image endpoint separately. HOG is a basic CPU
detector; do not use this workflow as a production detection benchmark.

## Enable a search workflow

| Workflow | Additional setup |
| --- | --- |
| Parsed-attribute search | Configure a VLM and analyze crops. Strict matching cannot match missing attributes. |
| Semantic text search | Configure Qwen embeddings, a separate Milvus visual collection and its index. Verify coverage before enabling `SEMANTIC_SEARCH_ENABLED`. |
| Cross-camera candidates | Configure ReID and its index, then evaluate on a labeled dataset. |

See the [container deployment guide](../deploy/containers/README.md) for semantic search setup,
and the [multimodal retrieval notes](multimodal-retrieval/README.md) for retrieval behavior.
Scores rank candidates; they do not establish an identity or a verified path across cameras.

## Report a problem

Use [Issues](https://github.com/otterview-labs/SightIndex/issues). Include the commit, relevant
configuration names, workflow, status code and reproduction steps. State which model services
were enabled. Remove credentials and private images from the report.

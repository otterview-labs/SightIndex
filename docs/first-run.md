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

For example, an installation with no person crops can return:

```json
{"image_with_crops_count": 0, "person_crop_count": 0}
```

This illustrates the response format, not bundled demo data. Query the image record to check
whether an upload succeeded.

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

## Check the responses

| Operation | What to inspect |
| --- | --- |
| Upload an image | The returned image record; save its `id` and `image_url`. |
| Query the image | The same `id`. Open `image_url` in the browser to inspect the source. Media requests also require authentication when enabled. |
| Process the image | A crop list. It can be `[]` when no person is detected. |
| List crops by `image_id` | Each result includes `crop_url`, `bbox` and the source `image_id`; compare it with the original image. |
| Check media counts | Images with crops contribute to `image_with_crops_count`; individual crops contribute to `person_crop_count`. |

Calling `/process` again on a processed image reuses its result, including an empty result.
Changing the detector and repeating the request with the same image ID does not rerun detection.
Reprocessing existing media requires a separate data-reprocessing workflow.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Upload succeeds but counts stay at zero | Query the image record and confirm that `/process` ran. Images without crops do not contribute to these counts. |
| People are visible but the crop list is empty | Inspect the source image, selected detector and parameters. The default HOG detector can miss people; try a clear image before evaluating a different model. |
| Attribute search returns no matches | The default configuration does not parse attributes. Check the VLM setup, crop analysis and whether the relevant fields exist. |
| Semantic search returns no matches | Check the embedding service, Milvus collection, historical index and coverage. Enabling the flag does not build an index. |
| Results do not change after switching models | Processed images reuse their results. Vector-model or preprocessing changes also require a separate collection; plan reprocessing and indexing using the deployment guide. |
| API or media requests return 401 | Check whether Basic Auth is enabled and use your configured account. Keep passwords out of issue reports. |

## Enable a search workflow

| Workflow | Additional setup |
| --- | --- |
| Parsed-attribute search | Configure a VLM and analyze crops. Strict matching cannot match missing attributes. |
| Semantic text search | Configure Qwen embeddings, a separate Milvus visual collection and its index. Verify coverage before enabling `SEMANTIC_SEARCH_ENABLED`. |
| Cross-camera candidates | Configure ReID and its index, then evaluate on a labeled dataset. |

See the [container deployment guide](../deploy/containers/README.md) for semantic search setup,
and the [multimodal retrieval notes](multimodal-retrieval/README.md) for retrieval behavior.
Scores rank candidates; they do not establish an identity or a verified path across cameras.

For deployment, read the [deployment guide](deployment.md). The API process currently owns
stream-capture threads and index queues; use one API worker per deployment. Prepare model weights
according to each service's instructions. SapiensID checkpoints are not bundled; see the
[licensing section](../README.md#licensing-and-third-party-code) for applicable notices.

## Report a problem

Use [Issues](https://github.com/otterview-labs/SightIndex/issues). Include the commit, relevant
configuration names, workflow, status code and reproduction steps. State which model services
were enabled. Remove credentials and private images from the report.

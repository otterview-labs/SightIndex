# Visual AI APIs

SightIndex exposes two explicit HTTP APIs for model capabilities that are used by search and
offline indexing:

- Image vector generation: convert one image into a visual embedding vector.
- VLM structured analysis: parse a person or vehicle image into normalized JSON attributes.

The existing `/api/embeddings/visual` endpoint remains available for internal compatibility. New
integrations should use the more specific endpoints below.

## Authentication

Both APIs can be protected with the existing model-service tokens:

- Image vector API uses `VISUAL_EMBEDDING_SERVICE_API_KEY`.
- VLM structured analysis API uses `VLM_SERVICE_API_KEY`.

Send either header form:

```http
Authorization: Bearer <token>
X-API-Key: <token>
```

If the corresponding environment variable is empty, the endpoint does not require a token.

## Text/Image Vector API

Create a single visual retrieval embedding vector from either text or an image. Use this endpoint
for text-to-vector and image-to-vector calls that should share the same vector space.

```http
POST /api/embeddings/image-vector
Content-Type: application/json
```

Text request body:

```json
{
  "text": "黑衣背包的人",
  "instruction": "Retrieve images that match the user query."
}
```

Image request body:

```json
{
  "image_base64": "base64 encoded image bytes, or a data:image/...;base64,... URL",
  "image_filename": "crop.jpg"
}
```

Provide exactly one of `text` or `image_base64`.

Text response body:

```json
{
  "embedding": [0.0123, -0.0456],
  "dim": 2048,
  "model": "Qwen3-VL-Embedding-2B",
  "provider": "qwen3_vl_http",
  "input_type": "text",
  "image_filename": null
}
```

Image response body:

```json
{
  "embedding": [0.0123, -0.0456],
  "dim": 2048,
  "model": "Qwen3-VL-Embedding-2B",
  "provider": "qwen3_vl_http",
  "input_type": "image",
  "image_filename": "crop.jpg"
}
```

Text example:

```bash
curl -X POST http://localhost:8000/api/embeddings/image-vector \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "黑衣背包的人"
  }'
```

Image example:

```bash
IMAGE_B64="$(base64 -i crop.jpg)"

curl -X POST http://localhost:8000/api/embeddings/image-vector \
  -H 'Content-Type: application/json' \
  -d "{
    \"image_base64\": \"${IMAGE_B64}\",
    \"image_filename\": \"crop.jpg\"
  }"
```

Runtime configuration:

```bash
VISUAL_EMBEDDING_PROVIDER=qwen3_vl_http
VISUAL_EMBEDDING_MODEL=Qwen3-VL-Embedding-2B
VISUAL_EMBEDDING_DIM=2048
VISUAL_EMBEDDING_SERVICE_URL=http://embedding.example.test:18021
VISUAL_EMBEDDING_SERVICE_TIMEOUT_SECONDS=15
VISUAL_EMBEDDING_SERVICE_FAILURE_COOLDOWN_SECONDS=30
VISUAL_EMBEDDING_MAX_CONCURRENCY=1
VISUAL_EMBEDDING_QUEUE_TIMEOUT_SECONDS=2
```

Notes:

- `text` uses `VISUAL_EMBEDDING_INSTRUCTION` unless request `instruction` is provided.
- `image_base64` may include a data URL prefix.
- The response vector dimension must match `VISUAL_EMBEDDING_DIM`.
- This endpoint only returns the vector. It does not write Milvus.
- Milvus writes still happen through index rebuild/backfill or ingest when
  `VECTOR_INDEX_ON_INGEST=true`.

## VLM Structured Analysis API

Parse one image into structured person or vehicle attributes.

```http
POST /api/vlm/structured-analysis
Content-Type: application/json
```

Request body:

```json
{
  "image_base64": "base64 encoded image bytes, or a data:image/...;base64,... URL",
  "image_filename": "person.jpg",
  "object_type": "person",
  "label_language": "zh",
  "bbox": {
    "box_id": "det-1",
    "x1": 10,
    "y1": 20,
    "x2": 220,
    "y2": 520,
    "score": 0.91,
    "label": "person"
  }
}
```

`object_type` supports:

- `person`
- `vehicle`

`label_language` controls the response `labels` field:

- `zh`: Chinese labels, default.
- `en`: English labels.

`bbox` is optional. When provided, the VLM prompt asks the model to prioritize that region.

Person response example:

```json
{
  "object_type": "person",
  "bbox": {
    "box_id": "det-1",
    "x1": 10,
    "y1": 20,
    "x2": 220,
    "y2": 520,
    "score": 0.91,
    "label": "person"
  },
  "attributes": {
    "object_type": "person",
    "appearance": {
      "hair": "short_hair",
      "hat": false,
      "glasses": false,
      "gender": "unknown",
      "age_group": "adult"
    },
    "clothing": {
      "upper_color": "black",
      "lower_color": "gray"
    },
    "objects": {
      "backpack": true,
      "holding_phone": false,
      "cigarette": false
    },
    "behavior": {
      "smoking": false,
      "looking_at_phone": false,
      "falling": false,
      "lying_on_ground": false,
      "fighting": false,
      "physical_conflict": false
    },
    "confidence": 0.82,
    "notes": ""
  },
  "labels": {
    "对象类型": "人员",
    "上衣颜色": "黑色",
    "下装颜色": "灰色",
    "背包": "是",
    "帽子": "否",
    "眼镜": "否"
  },
  "label_language": "zh",
  "model": "Qwen3.6-27B",
  "provider": "openai_compatible"
}
```

Vehicle response example:

```json
{
  "object_type": "vehicle",
  "bbox": null,
  "attributes": {
    "object_type": "vehicle",
    "vehicle_color": "white",
    "vehicle_type": "suv",
    "vehicle_brand": null,
    "plate_color": null,
    "confidence": 0.76,
    "notes": ""
  },
  "labels": {
    "object_type": "vehicle",
    "vehicle_color": "white",
    "vehicle_type": "suv"
  },
  "label_language": "en",
  "model": "Qwen3.6-27B",
  "provider": "openai_compatible"
}
```

Example:

```bash
IMAGE_B64="$(base64 -i person.jpg)"

curl -X POST http://localhost:8000/api/vlm/structured-analysis \
  -H 'Content-Type: application/json' \
  -d "{
    \"image_base64\": \"${IMAGE_B64}\",
    \"image_filename\": \"person.jpg\",
    \"object_type\": \"person\",
    \"label_language\": \"zh\",
    \"bbox\": {
      \"x1\": 10,
      \"y1\": 20,
      \"x2\": 220,
      \"y2\": 520,
      \"label\": \"person\"
    }
  }"
```

Runtime configuration:

```bash
VLM_PROVIDER=openai_compatible
VLM_BASE_URL=http://127.0.0.1:8000/v1
VLM_MODEL=your-vlm-model
VLM_API_KEY=your-token
VLM_SERVICE_API_KEY=
VLM_TIMEOUT_SECONDS=120
VLM_STRUCTURED_MAX_TOKENS=1200
```

Notes:

- This endpoint only returns parsed attributes. It does not persist to `person_crops.attributes`.
- To parse and persist an existing crop, use
  `POST /api/attributes/person-crops/{crop_id}/analyze`.
- To backfill recent crops, use `POST /api/attributes/person-crops/backfill?limit=50`.
- Keep `VLM_STRUCTURED_ON_INGEST=false` for stability unless ingest latency is acceptable.

## Recommended Stable AGX Setup

For demo stability, keep the online search path lightweight:

```bash
VECTOR_INDEX_ON_INGEST=false
PERSON_TRAJECTORY_VECTOR_ENABLED=false
VISUAL_EMBEDDING_MAX_CONCURRENCY=1
VISUAL_EMBEDDING_QUEUE_TIMEOUT_SECONDS=2
```

Use the image vector API for explicit vector generation and use rebuild/backfill jobs to write
vectors into Milvus when the system is not under demo load.

# 图片和视频多模态检索实施框架

## 实现归属

本功能的唯一实现源是 `SightIndex`，不再新建独立的图片检索后端。

`SightIndex` 已经具备下列核心能力：

| 能力 | 当前入口 | 现状 |
| --- | --- | --- |
| 图片上传和媒体记录 | `app/api/media.py`、`app/services/media.py` | 已实现 |
| 视频上传和抽帧 | `app/api/media.py`、`app/services/video_processing.py` | 已实现，同步处理 |
| 图片/人员裁剪向量 | `app/services/embeddings.py`、`app/services/vector_index.py` | 已实现，可用 CLIP/Qwen3-VL/Milvus |
| 文本检索 | `app/api/search.py`、`app/services/search.py` | 已实现图片、人员裁剪检索 |
| 以图搜图 | `POST /api/search/by-image` | 已实现，查询对象是已入库图片 |
| 人脸库和按姓名查找 | `app/api/face.py`、`app/services/faces.py`、`app/services/search.py` | 已实现基础链路 |
| 视频检索结果聚合 | `POST /api/search/videos` | 当前为空实现，需要补齐 |

因此，后续的代码直接落在现有的 `app/models`、`app/services`、`app/api`、`app/schemas` 和 `tests` 中。不要复制一套存储、模型加载、Milvus 连接或人脸库。

## Integration boundary

SightIndex owns visual-media ingestion, model calls, metadata, and vector indexes. An upstream
application can call its HTTP APIs and compose the returned evidence into a wider workflow, but it
must not read SightIndex's PostgreSQL tables, filesystem, or Milvus collections directly.

Images, videos, and RTSP/HTTP streams should enter SightIndex directly. Keep document stores,
conversation state, and unrelated retrieval indexes in the application that owns them. Even when
services share a physical PostgreSQL or Milvus cluster, use separate databases, collections, and
credentials.

## 目标架构

```text
图片上传
  -> Image
  -> 可选 PersonCrop
  -> VisualEmbeddingService
  -> VLEmbedding + Milvus image/person_crop collection

视频上传
  -> VideoAsset
  -> 异步视频处理任务
  -> VideoSegment (时间片段)
  -> Image (关键帧，关联 video_asset_id 和 offset_ms)
  -> 可选 PersonCrop
  -> VisualEmbeddingService
  -> VLEmbedding + Milvus collection

文本 / 查询图片 / 姓名
  -> VisualSearchService 或 FaceRecognitionService
  -> Frame / Crop 命中
  -> VideoSearchService 按 VideoSegment 聚合
  -> 返回原视频 URL、命中时间点、预览帧、相似度
```

通用图文向量、人脸向量和人体 ReID 向量必须保持独立：

- 图文向量用于场景、衣着、物体和自然语言语义检索。
- 人脸向量用于身份确认和按姓名定位。
- 人体 ReID 向量用于跨画面轨迹关联。

三类向量不能放进同一个 Milvus collection，也不能用同一个阈值解释结果。

## 代码落点

| 目标 | 应修改的现有位置 | 新增内容 |
| --- | --- | --- |
| 视频元数据 | `app/models/media.py` | `VideoAsset`、`VideoSegment`，以及 `Image` 对原视频和时间偏移的外键 |
| 视频抽帧 | `app/services/video_processing.py` | 保存视频资产、关键帧对应 `segment_id`、`offset_ms`、开始/结束时间 |
| 视频检索 | `app/services/video_search.py` | 从图片/裁剪命中聚合为视频片段，合并相邻命中 |
| 搜索 API | `app/api/search.py` | 用真实 `VideoSearchService` 替换空的 `/api/search/videos` |
| 请求响应 | `app/schemas/media.py` | `VideoSearchRequest`、`VideoSearchItem`、`VideoSearchResponse` |
| 持久化初始化 | `app/db/session.py` | 新表和索引的兼容迁移；正式环境应随后接 Alembic |
| 测试 | `tests/test_video_search.py` | 聚合、过滤、时间定位和 API 端到端测试 |

不新建第二个 FastAPI `main.py`，不复制 `VisualEmbeddingService`，也不复制 `MilvusVectorIndex`。

## 实施顺序

### 1. 先建立视频资产和片段模型

先让每个关键帧能回答三个问题：属于哪个视频、在视频的哪一毫秒、属于哪个可播放片段。没有这层关系，命中帧只能返回图片，不能稳定定位视频。

最低字段建议：

```text
video_assets
  id, video_url, original_filename, content_type, byte_size,
  duration_ms, camera_id, location_id, captured_at, created_at

video_segments
  id, video_asset_id, start_ms, end_ms,
  representative_image_id, created_at

images
  ... existing fields ...,
  video_asset_id, video_segment_id, video_offset_ms
```

### 2. 让视频处理持久化这层关系

`VideoProcessingService` 当前已能保存原视频、抽帧、生成 `Image` 和 `PersonCrop`。下一步在抽帧循环中创建 `VideoAsset`/`VideoSegment`，并把 `CAP_PROP_POS_MSEC` 写入 `Image.video_offset_ms`。初期可以按抽帧间隔直接生成片段；后续可以改为场景切分。

### 3. 复用现有向量检索

关键帧仍按 `image` 进入现有 `MilvusVectorIndex`；人员目标按 `person_crop` 进入现有裁剪 collection。不要为视频复制向量 collection。视频检索只是对命中图片/裁剪的第二层聚合。

### 4. 实现视频结果聚合

`VideoSearchService` 应按以下规则工作：

1. 调用现有 `VisualSearchService` 得到图片或人员裁剪命中。
2. 将裁剪命中回溯到所属 `Image`，再回溯到 `VideoSegment`。
3. 按 `video_asset_id` 分组，并合并时间距离小于阈值的相邻片段。
4. 每组保留最高分、最佳预览帧、最早/最晚时间点和关联人员信息。
5. 返回视频 URL、`start_ms`、`end_ms`、预览帧 URL 与相似度。

完整请求与返回字段见 [video-search-contract.md](video-search-contract.md)。

### 5. 改为异步任务

现有视频上传是同步调用，短视频可用，长视频会占用 HTTP Worker。持久化模型稳定后再增加任务表和 Worker：上传接口先返回 `202 + task_id`，处理器独立完成抽帧、检测、向量入库和任务状态更新。

## 模型选择

保持 `VisualEmbeddingService` 为唯一通用图文嵌入入口。第一轮 POC 建议在真实业务查询集上比较：

- `Qwen3-VL-Embedding-2B`：项目已有本地和 HTTP provider 适配，适合中文图文检索。
- `sentence-transformers/clip-ViT-B-32`：轻量基线，便于验证吞吐与 Milvus 链路。
- 需要外部 GPU 部署时，继续使用已有 `qwen3_vl_http` provider，不在 API 进程重复加载大模型。

人脸身份继续使用现有 InsightFace 链路；它不是图文检索模型的替代品。

## 验收标准

1. 上传一个视频后，每个入库关键帧可追溯到 `video_asset_id` 和 `video_offset_ms`。
2. 文本检索和以图搜图都能返回视频结果，且可跳转到命中的片段。
3. 按姓名检索通过人脸库得到同一人的视频片段，不依赖通用 CLIP 向量猜测身份。
4. 删除视频会级联清理片段、帧、裁剪和 Milvus 向量。
5. 使用 300-1000 条真实查询评估 Recall@10、视频时间点准确率、P95 延迟和 GPU 吞吐量。

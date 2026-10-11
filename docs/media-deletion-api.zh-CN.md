# 图片与向量删除接口设计

本文给出 `DELETE /api/images/{image_id}` 的完整设计方案：删一张图片时，数据库、Milvus 向量、
磁盘文件和索引队列各自要清理什么、按什么顺序、失败怎么办、怎么验收。

文中标注为 **[实测]** 的内容来自北京测试机上的真实库与真实 Milvus 集合，其余为设计建议。

---

## 1. 目标与范围

**要解决的问题**：现在没有删除接口。老的 `scripts/cleanup_media_before.py` 只能按时间批量清理，
既不能按单个素材删除，也**没有跟上后来新增的表**（见 §5.3）。

**本期范围**：

- 按 `image_id` 删除一张图片，以及它派生的全部人员裁剪、向量、事件与文件；
- 支持 `dry_run`，先看清会删掉什么；
- 幂等：重复删除返回 404 而不是报错。

**不在本期范围**：删除人员（`persons`）、删除视频源文件、删除人脸库条目（涉及身份，需单独设计，见 §9.3）。

---

## 2. 一张图片牵出多少东西

删除的对象不是一行记录，而是一棵子图。

```
images (1 行)
├── person_crops                    该图检出的所有人员裁剪（1:N）
│   ├── person_observation_index    观察表（检索用，1:1）
│   ├── crop_face_extractions       人脸提取缓存（1:1，以 crop_id 为主键）
│   ├── reid_match_feedback         人工反馈（FK 声明了 CASCADE）
│   ├── vl_embeddings               PostgreSQL 侧向量副本（1:N）
│   ├── face_embeddings             人脸向量（可能属于已登记人员）
│   ├── recognition_events          识别事件
│   └── counting_events             计数事件
├── vl_embeddings                   image 自身的向量副本
├── recognition_events              以 image_id 直接关联的事件
├── counting_events                 以 image_id 直接关联的事件
├── video_streams.last_frame_image_id   若该图是某路流的最新帧，需置空
└── vector_index_jobs               未完成的索引任务（会引用上述 object_id）
```

Milvus 侧（**[实测]** 北京机器的真实集合）：

| 集合 | 内容 | 实测行数 |
| --- | --- | ---: |
| `sightindex_qwen3vl2b_v1_vl_images` | 图片视觉向量 | 22 |
| `sightindex_qwen3vl2b_v1_vl_person_crops` | 人员裁剪视觉向量 | 22 |
| `sightindex_reid_person_crops_<namespace>` | **ReID 人体向量**（另一套模型/维度） | 40 |

磁盘文件（`DATA_DIR` 下）：

| 字段 | 路径示例 | 说明 |
| --- | --- | --- |
| `images.image_url` | `/data/uploads/<uuid>.png`、`/data/frames/...` | 素材本体 |
| `images.thumbnail_url` | `/data/thumbnails/<uuid>.png` | 标注缩略图 |
| `person_crops.crop_url` | `/data/crops/<uuid>.png` | 裁剪图 |

> ⚠️ `images.video_url` 指向**源视频文件**，会被同一视频的所有关键帧共享。删除单帧时
> **绝不能删这个文件**，只有删到该视频的最后一帧（或走视频级删除）时才处理。

---

## 3. 为什么不能直接 `DELETE FROM images WHERE id = ...`

三个独立的坑，缺一个都会留下脏数据。

### 3.1 外键没有级联，会直接违约

**[实测]** 北京机器 `information_schema.referential_constraints` 的真实结果：

```
reid_match_feedback.query_crop_id      -> person_crops  CASCADE   ← 唯一会自动清理的
reid_match_feedback.candidate_crop_id  -> person_crops  CASCADE
crop_face_extractions.crop_id          -> person_crops  NO ACTION ← 会挡住裁剪删除
person_observation_index.crop_id       -> person_crops  NO ACTION ← 会挡住
person_observation_index.image_id      -> images        NO ACTION ← 会挡住
counting_events / recognition_events / face_embeddings / person_crops.image_id
video_streams.last_frame_image_id                       NO ACTION
```

也就是说，直接删 `person_crops` 会被 `crop_face_extractions` 和 `person_observation_index`
以 `ForeignKeyViolation` 拒绝。

### 3.2 向量会变成"幽灵"

DB 行删掉、Milvus 里的向量还在，检索就会返回指向已删除素材的命中，前端点开是 404。
而且 ReID 集合是**独立的一套**（`vector_index.py` 里 `reid_object_types`），很容易漏删。

### 3.3 队列会把删除的东西"复活"

`vector_index_jobs` 是持久化队列（`target` + `object_id` 唯一约束），worker 每轮会 claim 任务。
如果删除时不清掉相关任务，worker 可能对已删除的 `object_id` 反复重试并写入 `last_error`，
甚至在 Milvus 里重新建出向量。

---

## 4. API 契约

### 4.1 删除单张图片

```
DELETE /api/images/{image_id}
```

查询参数：

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `dry_run` | bool | `false` | 只统计、不落任何改动 |
| `delete_files` | bool | `true` | 是否删除磁盘文件；设为 `false` 便于人工复核后再清 |
| `reason` | string | `null` | 写入删除账本/备份表，便于审计 |

响应 `200`：

```json
{
  "image_id": "…",
  "dry_run": false,
  "deleted": {
    "person_crops": 3,
    "person_observation_index": 3,
    "crop_face_extractions": 3,
    "reid_match_feedback": 1,
    "recognition_events": 4,
    "counting_events": 0,
    "face_embeddings": 2,
    "vl_embeddings": 4,
    "vector_index_jobs": 2,
    "milvus": {"vl_images": 1, "vl_person_crops": 3, "reid_person_crops": 3},
    "files": 7
  },
  "warnings": [
    "face_embeddings 中有 2 条属于已登记人员，人物轨迹会因此缺一段"
  ]
}
```

`dry_run=true` 时 `deleted` 全为**预计值**，且不产生任何写入。

状态码：

| 码 | 场景 |
| --- | --- |
| `200` | 删除完成（或 `dry_run` 统计完成） |
| `404` | 图片不存在——**幂等**，重复删除得到同样结果，不报错 |
| `409` | 该图仍被引用且未指定处理方式（预留，例如未来 `video_streams.last_frame_image_id` 不允许清空时） |
| `503` | Milvus 不可用且本次删除需要它（见 §6.2），此时**不做任何破坏性操作** |

### 4.2 批量删除（第二阶段）

```
POST /api/images/delete
{
  "image_ids": ["…", "…"],          // 或
  "before": "2026-05-01T00:00:00+08:00",
  "source_type": "stream_frame",
  "limit": 500,
  "dry_run": true
}
```

约束与理由：

- **必须**设置 `limit` 上限（建议 500/次），避免一次请求删掉整库；
- `dry_run` 默认 `true`——批量删是不可逆操作，强制"先看后删"；
- 生产建议走异步：接口返回 `202 + task_id`，进度写 `vector_index_jobs` 同构的任务表。

### 4.3 鉴权与安全

- 沿用 `main.py` 的 Basic Auth（`APP_BASIC_AUTH_USERNAME/PASSWORD`）：删除是写操作，
  **不能**放进 `AUTH_EXEMPT_PATHS`；
- 建议再加一层：仅当 `settings.media_deletion_enabled`（新增开关）为真时才注册该路由，
  防止误部署到生产后被人一键清库；
- 记录审计：谁（用户名/IP）、何时、删了什么、`reason`。

---

## 5. 删除顺序

顺序不是风格问题——`NO ACTION` 的外键会强制你按特定次序删，而向量和文件必须在 DB 行消失
**之前**把关键信息（id、url）取出来。

### 5.1 阶段 0：预检与快照（同一事务内）

```python
image = db.get(Image, image_id)
if image is None:
    return 404

crop_ids      = db.scalars(select(PersonCrop.id).where(PersonCrop.image_id == image_id)).all()
crop_urls     = db.scalars(select(PersonCrop.crop_url).where(PersonCrop.image_id == image_id)).all()
file_urls     = [image.image_url, image.thumbnail_url, *crop_urls]   # 不含 image.video_url
vl_embedding_ids   = ...   # object_type='image' 且 object_id=image_id，或 person_crop 且 object_id in crop_ids
face_embedding_ids = ...
job_ids            = ...   # vector_index_jobs 中相关的行
referenced_streams = ...   # video_streams.last_frame_image_id == image_id
```

`dry_run` 在此处统计完直接返回。

### 5.2 阶段 1：数据库（单个事务，顺序不可乱）

```sql
-- 1) 解除流的最新帧引用，否则删 images 会被外键挡住
UPDATE video_streams SET last_frame_image_id = NULL WHERE last_frame_image_id = :image_id;

-- 2) 以 crop_id 为键的子表（必须先于 person_crops）
DELETE FROM reid_match_feedback    WHERE query_crop_id = ANY(:crop_ids)
                                      OR candidate_crop_id = ANY(:crop_ids);  -- 有 CASCADE，显式删更稳妥
DELETE FROM crop_face_extractions  WHERE crop_id = ANY(:crop_ids);            -- 无级联，必须显式删
DELETE FROM person_observation_index
       WHERE crop_id = ANY(:crop_ids) OR image_id = :image_id;                -- 无级联，必须显式删
DELETE FROM counting_events        WHERE crop_id = ANY(:crop_ids) OR image_id = :image_id;
DELETE FROM recognition_events     WHERE crop_id = ANY(:crop_ids) OR image_id = :image_id;

-- 3) 向量与任务
DELETE FROM face_embeddings WHERE crop_id = ANY(:crop_ids) OR image_id = :image_id;
DELETE FROM vl_embeddings
       WHERE (object_type = 'image'       AND object_id = :image_id)
          OR (object_type = 'person_crop' AND object_id = ANY(:crop_ids));
DELETE FROM vector_index_jobs
       WHERE object_id = :image_id
          OR (target IN ('person_crop','reid_person_crop','person_attributes')
              AND object_id = ANY(:crop_ids));

-- 4) 最后才是主体
DELETE FROM person_crops WHERE image_id = :image_id;
DELETE FROM images       WHERE id = :image_id;
```

**为什么先删子表**：`NO ACTION` 意味着约束在语句结束时就检查，父行不能先走。

**为什么先删 `vector_index_jobs`**：worker 是独立线程，先删任务能立刻止血；否则它可能在你删
DB 行之后仍去读 `person_crops` 并写回向量。

### 5.3 阶段 2：Milvus 向量

复用现成能力（`scripts/cleanup_media_before.py` 已经这么写了）：

```python
from app.services.vector_index import MilvusVectorIndex
index = MilvusVectorIndex(settings)

for object_type, ids in (
    ("image", [image_id]),
    ("person_crop", crop_ids),
    ("reid_person_crop", crop_ids),      # ← 老脚本漏了这一类
):
    if not ids:
        continue
    collection = index._collection(object_type)
    for batch in batched([str(i) for i in ids], 200):
        quoted = ", ".join(f'"{oid}"' for oid in batch)
        collection.delete(expr=f"object_id in [{quoted}]")
    collection.flush()
```

`MilvusVectorIndex.collection_suffixes` 已给出映射：

```python
{"image": "vl_images", "person_crop": "vl_person_crops",
 "face_embedding": "face_embeddings", "reid_person_crop": "reid_person_crops"}
```

> 老脚本 `scripts/cleanup_media_before.py` 只删 `image` 和 `person_crop` 两类，
> **不删 ReID 集合**；同时它的表清单里也没有 `person_observation_index`、
> `crop_face_extractions`、`reid_match_feedback`、`vector_index_jobs`。
> 新接口要按 §5.2 的清单来，不要直接照抄老脚本。

### 5.4 阶段 3：磁盘文件

```python
for url in file_urls:                       # 明确不含 image.video_url
    if not url or not url.startswith("/data/"):
        continue
    path = (data_dir / url.removeprefix("/data/")).resolve()
    if path.is_relative_to(data_dir.resolve()) and path.is_file():
        path.unlink(missing_ok=True)
```

路径必须做 `is_relative_to` 校验（`semantic_search.py:_fingerprint` 里已有同样的写法），
防止历史数据里出现 `../` 导致越权删除。

### 5.5 可行性验证（已在测试库上跑过，事务内回滚）

2026-10-10 在北京测试库上用 `BEGIN … ROLLBACK` 实测了 §5.2 的顺序（未产生任何真实改动）。

**反例：直接删 `person_crops`**

```
ERROR: update or delete on table "person_crops" violates foreign key constraint
       "person_observation_index_crop_id_fkey" on table "person_observation_index"
DETAIL: Key (id)=(ed9249a7-…) is still referenced from table "person_observation_index".
```

证实了 §3.1 的判断：`NO ACTION` 外键确实会拦住朴素的删除。

**正例：按 §5.2 的顺序**

```
UPDATE video_streams …                        （无匹配）
DELETE reid_match_feedback                  0
DELETE crop_face_extractions                0
DELETE person_observation_index             2
DELETE counting_events                      0
DELETE recognition_events                   0
DELETE face_embeddings                      0
DELETE vl_embeddings                        0
DELETE vector_index_jobs                    5
DELETE person_crops                         2
DELETE images                               1
ROLLBACK                                     ← 数据原封不动
```

全部语句成功，无外键报错；回滚后 `person_crops` / `images` / `person_observation_index`
行数均未变化。

> **实测结论（部署相关）**：这台机器上 PostgreSQL 的 `vl_embeddings` 表**是空的**，
> 视觉向量只存在于 Milvus。因此在这类部署里，**Milvus 的删除是唯一真正生效的向量清理**，
> §5.2 中的 `DELETE FROM vl_embeddings` 只是兜底（保留它，以兼容向量会落库的部署方式）。
> 反过来也说明：**漏删 Milvus 就等于完全没删向量**，这一点务必写进验收清单。

---

## 6. 失败与一致性

### 6.1 三个阶段的事务边界

DB 事务只能覆盖阶段 1。Milvus 和文件系统不支持回滚，所以只能**先做不可回滚的、后做可重试的**：

推荐顺序：**阶段 1（DB）→ 阶段 2（Milvus）→ 阶段 3（文件）**。理由是：

- 先删 DB 行，检索接口立刻不会再返回该素材（用户可见的"已删除"最先达成）；
- 向量残留只会导致检索命中一个 404，比"行还在但向量没了"更容易发现和补偿；
- 文件删不掉不影响接口正确性，可重试。

### 6.2 Milvus 不可用时必须"要么不做，要么记账"

`MilvusVectorIndex._require_available()` 的设计意图很明确（源码注释）：

> Reject correctness-sensitive operations instead of silently succeeding.
> A no-op during the failure cooldown would otherwise delete the durable outbox job and commit
> an index marker for a vector that was never written.

所以**不能**在 Milvus 挂了的时候静默跳过删除——那会留下永久性的幽灵向量。两种可选策略：

| 策略 | 行为 | 适用 |
| --- | --- | --- |
| **A. 快速失败（推荐先做）** | 预检 Milvus 可用性（`MilvusVectorIndex.probe()`），不可用直接 `503`，一个字节都不删 | 单张删除、人工操作 |
| **B. 记账补偿** | 先删 DB 行，同时写入 `media_deletion_ledger`（待删向量清单），后台 worker 重试直到 Milvus 删成功 | 批量删除、Milvus 不稳定环境 |

策略 B 建议新增一张账本表（与 `vector_index_jobs` 同构，复用同一套 lease 机制）：

```sql
CREATE TABLE media_deletion_ledger (
  id              uuid PRIMARY KEY,
  object_type     text NOT NULL,        -- image / person_crop / reid_person_crop
  object_id       uuid NOT NULL,
  status          text NOT NULL,        -- pending / done / failed
  attempts        int  NOT NULL DEFAULT 0,
  next_run_at     timestamptz,
  last_error      text,
  created_at      timestamptz NOT NULL DEFAULT now()
);
```

### 6.3 文件删除失败

收集失败清单，随响应返回 `warnings`，或写进账本延后重试。文件残留**不影响检索正确性**，
优先级最低。

---

## 7. 并发与竞态

| 竞态 | 场景 | 处理建议 |
| --- | --- | --- |
| 删除 vs 入库 | 图片刚上传、`process` 正在跑，同时被删 | 阶段 1 事务内先 `SELECT ... FOR UPDATE` 锁 `images` 行；`process` 侧对 `Image` 行加同样的锁 |
| 删除 vs 索引队列 | worker 正持有该对象的 lease | 先删 `vector_index_jobs` 并提交，再删 DB 行；worker 的 claim 会因行消失而跳过 |
| 删除 vs 观察表重建 | `ObservationIndexService.rebuild()` 正在全量重写 | 该服务已有 `_lock_writes_in_session()`，删除路径应使用同一把锁 |
| 重复删除 | 两个请求同时删同一张图 | 阶段 1 用 `DELETE ... WHERE id = :id RETURNING id` 判断是否真的删到；返回 404 而非 500 |

---

## 8. 验收与可观测性

### 8.1 删除前用 dry-run 核对

```bash
curl -s -X DELETE 'http://127.0.0.1:2059/api/images/<IMAGE_ID>?dry_run=true' | jq
```

### 8.2 删除后自检（DB）

```sql
SELECT 'crops'     , count(*) FROM person_crops              WHERE image_id = :image_id
UNION ALL SELECT 'observation', count(*) FROM person_observation_index WHERE image_id = :image_id OR crop_id = ANY(:crop_ids)
UNION ALL SELECT 'face_cache' , count(*) FROM crop_face_extractions   WHERE crop_id = ANY(:crop_ids)
UNION ALL SELECT 'reid_fb'    , count(*) FROM reid_match_feedback     WHERE query_crop_id = ANY(:crop_ids);
-- 全部应为 0
```

### 8.3 删除后自检（Milvus）

```python
for object_type, ids in (("image", [image_id]), ("person_crop", crop_ids),
                         ("reid_person_crop", crop_ids)):
    collection = MilvusVectorIndex(settings)._collection(object_type)
    res = collection.query(expr=f'object_id in ["…"]', output_fields=["object_id"])
    assert not res, (object_type, res)
```

注意 Milvus 删除是**标记删除**，`num_entities` 不会立刻下降；要用 `query` 而不是计数来判断。

### 8.4 回归检查

删除后应当：

- `POST /api/search/person-crops`、`/api/search/semantic/person-crops` 不再返回该素材；
- `GET /api/search/observations` 不再出现对应行；
- `/api/person-crops?image_id=…` 返回空；
- `/api/media/counts` 的计数相应减少。

---

## 9. 风险与注意事项

### 9.1 `face_embeddings` 可能属于已登记人员

`face_embeddings.image_id` 指向登记照。删掉这张图会**同时删掉该人员的人脸库条目**，
此后 `mode=face` 的人物轨迹会缺一段。建议：

- 删除前统计并返回 `warnings`；
- 若 `face_embeddings.person_id IS NOT NULL`，要求显式传 `?confirm_face_removal=true`。

### 9.2 删除是不可逆的

建议保留 §5.1 的"快照"能力：像 `cleanup_media_before.py` 那样把待删行写入
`cleanup_before_<时间戳>` 备份表，再由 DBA 决定何时真正清理。老脚本已有
`backup_tables()` / `insert_backup()` 可复用。

### 9.3 不做的事

- 不删 `persons`（人物档案独立于素材）；
- 不删 `images.video_url` 指向的源视频（多帧共享，需视频级删除接口）；
- 不删 `video_streams`（有独立的 `DELETE /api/streams/{id}`）。

---

## 10. 建议的实施顺序

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| **P0** | 单张 `DELETE /api/images/{id}`，策略 A（Milvus 不可用即 503），`dry_run` 支持 | §8.2/8.3 自检全为 0；§8.4 四项回归通过 |
| **P1** | 批量删除（`limit` 上限 + `dry_run` 默认开） | 500 张一批，无外键报错、无幽灵向量 |
| **P2** | 账本表 + 后台补偿 worker，改为 `202 + task_id` 异步 | 拔掉 Milvus 后仍能最终一致 |
| **P3** | 视频级删除（连带帧、片段、源文件） | 按 `video_url` 聚合后整体清理 |

---

## 11. 测试清单

- [ ] 删除只有图片、没有裁剪的素材（最小路径）
- [ ] 删除带 3 个裁剪、1 条 ReID 反馈的素材（覆盖全部子表）
- [ ] 删除作为 `video_streams.last_frame_image_id` 的帧（验证置空逻辑）
- [ ] 删除已登记人脸的素材（验证 warning / 二次确认）
- [ ] 重复删除同一 id（幂等，第二次 404）
- [ ] `dry_run=true` 不产生任何写入（对比删除前后的行数与 Milvus 查询）
- [ ] Milvus 停止时删除（策略 A 返回 503 且 DB 未改动）
- [ ] 删除过程中并发调用 `process`（无悬挂裁剪）
- [ ] 历史脏数据：`attributes` 为空、`vl_embeddings` 缺失的素材也能正常删

---

## 12. 附：可直接复用的现成能力

| 能力 | 位置 |
| --- | --- |
| 三类向量的集合访问与删除 | `app/services/vector_index.py` → `MilvusVectorIndex._collection()` |
| Milvus 可用性预检 | 同文件 → `probe()` / `_require_available()` |
| 表备份与按 id 删除的参考实现 | `scripts/cleanup_media_before.py` → `backup_tables()` / `delete_rows()` / `delete_milvus_entries()` / `remove_files()` |
| 路径越权校验写法 | `app/services/semantic_search.py` → `_fingerprint()` 里的 `is_relative_to` |
| 观察表写入锁 | `app/services/observation_index.py` → `_lock_writes_in_session()` |
| 持久化任务队列（账本可照此实现） | `app/models/vectors.py` → `VectorIndexJob`；`app/services/vector_index_queue.py` |
| 现有 DELETE 端点写法参考 | `app/api/media.py` → `delete_stream()` |
| 依赖注入 | `app/api/deps.py` → `DBSession` / `AppSettings` |

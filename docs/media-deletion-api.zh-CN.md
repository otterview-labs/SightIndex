# 图片与向量删除接口设计

现在没有删除接口。老脚本 `scripts/cleanup_media_before.py` 只能按时间批量清理，不能按单个素材
删；而且它写的时候还没有 `person_observation_index`、`crop_face_extractions` 这些表，清单已经
过期，直接抄会漏东西。

这里说清楚 `DELETE /api/images/{image_id}` 该怎么做：删一张图要动哪些数据、按什么顺序、失败
怎么收场。

下面出现的表名、外键规则、Milvus 集合名，都是在北京测试机的实际库和实际集合上查出来的。

## 一张图底下挂着什么

删的不是一行记录，是一棵子树。

```text
images
├── person_crops
│   ├── person_observation_index     观察表，检索用
│   ├── crop_face_extractions        人脸提取缓存，主键就是 crop_id
│   ├── reid_match_feedback          人工反馈
│   ├── vl_embeddings                PG 里的向量副本
│   ├── face_embeddings              人脸向量，可能属于已登记人员
│   ├── recognition_events
│   └── counting_events
├── vl_embeddings                    图片自身的向量副本
├── recognition_events / counting_events    按 image_id 直接关联的
├── video_streams.last_frame_image_id       这张图如果是某路流的最新帧
└── vector_index_jobs                还没跑完的索引任务
```

Milvus 侧有三个集合涉及：

| 集合 | 内容 | 实测行数 |
| --- | --- | ---: |
| `sightindex_qwen3vl2b_v1_vl_images` | 图片视觉向量 | 22 |
| `sightindex_qwen3vl2b_v1_vl_person_crops` | 裁剪视觉向量 | 22 |
| `sightindex_reid_person_crops_<namespace>` | ReID 向量，另一套模型和维度 | 40 |

文件路径在三个字段里：`images.image_url`、`images.thumbnail_url`、`person_crops.crop_url`。

`images.video_url` 指向源视频，同一个视频的所有关键帧共用它。删单帧不能碰这个文件。

## 直接删会坏在哪

### 外键

应用实际连的是 `milvus_log` 库的 `test` schema（`.env` 里生效的那条 `DATABASE_URL`）。那里的
规则是：

| 外键 | 引用 | 删除规则 |
| --- | --- | --- |
| `reid_match_feedback.query_crop_id` | `person_crops` | CASCADE |
| `reid_match_feedback.candidate_crop_id` | `person_crops` | CASCADE |
| `person_observation_index.crop_id` | `person_crops` | NO ACTION |
| `person_observation_index.image_id` | `images` | NO ACTION |
| `person_crops.image_id` | `images` | NO ACTION |
| `recognition_events.crop_id` / `.image_id` | `person_crops` / `images` | NO ACTION |
| `video_streams.last_frame_image_id` | `images` | NO ACTION |

只有 `reid_match_feedback` 会跟着删。删 `person_crops` 之前不动 `person_observation_index`，
就会被它的外键直接拒绝。

注意 `crop_face_extractions`、`counting_events`、`face_embeddings` 在这台机器的这个库里
**没有外键**。同机的另一个库 `sightindex`（`.env` 里被注释掉的那条）是从同一批模型建的，
这三张表都有 `NO ACTION` 外键。差异来自建库时机：`create_all` 只建缺失的表，不会给已有表补
约束，所以老库会缺约束。删除逻辑对这几张表一律显式清理，两种库上都成立。

实测这个库里 `DELETE FROM person_crops` 的报错是：

```
ERROR: update or delete on table "person_crops" violates foreign key constraint
       "person_observation_index_crop_id_fkey" on table "person_observation_index"
```

### 向量

DB 行删了、Milvus 里的向量还在，检索就会返回已经删掉的素材，前端点开是 404。ReID 用的是独立
集合（`MilvusVectorIndex.reid_object_types`），最容易漏。

### 队列

`vector_index_jobs` 是持久化队列，worker 会定期 claim。删除时不清掉相关任务，worker 会拿着已经
不存在的 `object_id` 反复重试，Milvus 里的向量也可能被重新写回来。

## 接口

### 删单张

```
DELETE /api/images/{image_id}
```

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `dry_run` | bool | `false` | 只统计不落改动 |
| `delete_files` | bool | `true` | 关掉就先只删记录，文件留给人复核 |
| `reason` | string | `null` | 记进备份表和日志，便于追责 |

返回 200：

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
    "face_embeddings 里有 2 条属于已登记人员，人物轨迹会缺一段"
  ]
}
```

`dry_run=true` 时这些数字是预估值，不产生任何写入。

状态码：

| 码 | 什么时候返回 |
| --- | --- |
| 200 | 删完，或者 dry_run 统计完 |
| 404 | 图片不存在。重复删同一个 id 也是 404，不要报 500 |
| 409 | 还有引用没处理干净（预留） |
| 503 | 这次删除需要 Milvus 但它用不了，此时一个字节都不删 |

### 批量

放到第二阶段，形态大概是这样：

```
POST /api/images/delete
{
  "image_ids": ["…", "…"],
  "before": "2026-05-01T00:00:00+08:00",
  "source_type": "stream_frame",
  "limit": 500,
  "dry_run": true
}
```

三个限制：必须带 `limit`（建议 500 封顶）；`dry_run` 默认开，批量删不可逆；真的删建议走异步，
返回 `202 + task_id`，进度沿用 `vector_index_jobs` 那套 lease 机制。

### 鉴权

沿用 `main.py` 的 Basic Auth（`APP_BASIC_AUTH_USERNAME` / `APP_BASIC_AUTH_PASSWORD`）。删除是
写操作，不能放进 `AUTH_EXEMPT_PATHS`。建议再加一个 `MEDIA_DELETION_ENABLED` 开关，不开就不注册
这条路由，免得测试代码带到生产上被人一键清库。日志里记下操作者、时间、对象和 `reason`。

## 删除顺序

顺序不是喜好问题。`NO ACTION` 的外键逼着你按固定次序删，而向量和文件必须在 DB 行消失之前把
id 和 url 取出来。

### 第一步：预检和快照

在一个事务里读完所有需要的东西：

```python
image = db.get(Image, image_id)
if image is None:
    return 404

crop_ids       = db.scalars(select(PersonCrop.id).where(PersonCrop.image_id == image_id)).all()
crop_urls      = db.scalars(select(PersonCrop.crop_url).where(PersonCrop.image_id == image_id)).all()
file_urls      = [image.image_url, image.thumbnail_url, *crop_urls]   # 不含 image.video_url
vl_embedding_ids   = ...
face_embedding_ids = ...
job_ids            = ...
referenced_streams = ...   # last_frame_image_id == image_id
```

`dry_run` 到这里就能算完返回了。

### 第二步：数据库

一个事务里按这个次序走：

```sql
-- 先解开流的最新帧引用，不然删 images 会被挡
UPDATE video_streams SET last_frame_image_id = NULL WHERE last_frame_image_id = :image_id;

-- crop_id 相关的子表，必须排在 person_crops 前面
DELETE FROM reid_match_feedback    WHERE query_crop_id = ANY(:crop_ids)
                                      OR candidate_crop_id = ANY(:crop_ids);
DELETE FROM crop_face_extractions  WHERE crop_id = ANY(:crop_ids);
DELETE FROM person_observation_index
       WHERE crop_id = ANY(:crop_ids) OR image_id = :image_id;
DELETE FROM counting_events        WHERE crop_id = ANY(:crop_ids) OR image_id = :image_id;
DELETE FROM recognition_events     WHERE crop_id = ANY(:crop_ids) OR image_id = :image_id;

-- 向量和任务
DELETE FROM face_embeddings WHERE crop_id = ANY(:crop_ids) OR image_id = :image_id;
DELETE FROM vl_embeddings
       WHERE (object_type = 'image'       AND object_id = :image_id)
          OR (object_type = 'person_crop' AND object_id = ANY(:crop_ids));
DELETE FROM vector_index_jobs
       WHERE object_id = :image_id
          OR (target IN ('person_crop','reid_person_crop','person_attributes')
              AND object_id = ANY(:crop_ids));

-- 最后才是主体
DELETE FROM person_crops WHERE image_id = :image_id;
DELETE FROM images       WHERE id = :image_id;
```

`reid_match_feedback` 虽然声明了 CASCADE，还是显式删一遍，省得依赖 DB 建表时的实际行为。

先删子表的原因很直接：`NO ACTION` 的约束在语句结束时检查，父行不能先走。

先删 `vector_index_jobs` 是为了止血。worker 是独立线程，任务留着，它可能在你删完之后仍去读
`person_crops` 并写回向量。

### 第三步：Milvus

现成的能力直接够用：

```python
from app.services.vector_index import MilvusVectorIndex
index = MilvusVectorIndex(settings)

for object_type, ids in (
    ("image", [image_id]),
    ("person_crop", crop_ids),
    ("reid_person_crop", crop_ids),      # 老脚本漏了这一类
):
    if not ids:
        continue
    collection = index._collection(object_type)
    for batch in batched([str(i) for i in ids], 200):
        quoted = ", ".join(f'"{oid}"' for oid in batch)
        collection.delete(expr=f"object_id in [{quoted}]")
    collection.flush()
```

`MilvusVectorIndex.collection_suffixes` 里已经有映射：

```python
{"image": "vl_images", "person_crop": "vl_person_crops",
 "face_embedding": "face_embeddings", "reid_person_crop": "reid_person_crops"}
```

`scripts/cleanup_media_before.py` 只删了 `image` 和 `person_crop` 两类，没碰 ReID 集合；它的表清单
也缺 `person_observation_index`、`crop_face_extractions`、`reid_match_feedback`、
`vector_index_jobs`。新接口按上面的清单来。

### 第四步：文件

```python
for url in file_urls:                       # 前面确认过不含 video_url
    if not url or not url.startswith("/data/"):
        continue
    path = (data_dir / url.removeprefix("/data/")).resolve()
    if path.is_relative_to(data_dir.resolve()) and path.is_file():
        path.unlink(missing_ok=True)
```

`is_relative_to` 这层校验不能省。历史数据里出现过奇怪的路径，直接拼字符串删文件会越权。
`semantic_search.py` 的 `_fingerprint` 里已经有同样的写法。

### 顺序验证过

2026-10-13 在测试机上拿 `BEGIN … ROLLBACK` 把上面的 SQL 跑了一遍（应用真正使用的
`milvus_log.test`），没有产生真实改动。

先试错误顺序，直接删 `person_crops`：

```
ERROR: update or delete on table "person_crops" violates foreign key constraint
       "person_observation_index_crop_id_fkey" on table "person_observation_index"
DETAIL: Key (id)=(977b0788-…) is still referenced from table "person_observation_index".
```

再按上面第二步的顺序：

```
UPDATE video_streams                        无匹配
DELETE reid_match_feedback                  0
DELETE person_observation_index             2
DELETE vector_index_jobs                    5
DELETE crop_face_extractions                0
DELETE counting_events                      0
DELETE face_embeddings                      0
DELETE vl_embeddings                        0
DELETE recognition_events                   0
DELETE person_crops                         2
DELETE images                               1
ROLLBACK
```

没有外键报错，回滚后 `person_crops`、`images` 的行数都没变。

有一件事顺便确认了：这台机器上 `vl_embeddings` 表是空的，向量只在 Milvus 里。所以在这种部署
下，Milvus 那一步才是唯一真正生效的向量清理，`DELETE FROM vl_embeddings` 只是兜底。写验收脚本
的时候别只看 PG 的行数。

## 失败与一致性

DB 事务只覆盖得到第二步。Milvus 和文件系统没有回滚，所以顺序安排成：先做不可回滚的，后做可
重试的。

具体是 DB → Milvus → 文件。这样用户看到的"已删除"最先达成；向量残留最多让检索命中一个 404，
比"行还在但向量没了"好查；文件删不掉不影响正确性，事后重试就行。

Milvus 用不了的时候不能静默跳过。`MilvusVectorIndex._require_available()` 的注释写得很清楚：

> Reject correctness-sensitive operations instead of silently succeeding.
> A no-op during the failure cooldown would otherwise delete the durable outbox job and commit
> an index marker for a vector that was never written.

跳过等于制造永久性的幽灵向量。两种做法：

| 做法 | 行为 | 适合 |
| --- | --- | --- |
| 快速失败 | 先 `MilvusVectorIndex.probe()`，不可用就返回 503，什么都不删 | 单张删除，人工操作 |
| 记账补偿 | 先删 DB 行，同时把待删向量写进账本，后台重试 | 批量删除，Milvus 不稳定 |

记账的话加一张表，字段和 lease 机制照抄 `vector_index_jobs`：

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

文件删除失败优先级最低，收集失败清单返回给调用方，或者写进账本稍后重试。

## 并发

| 情况 | 处理 |
| --- | --- |
| 删除和入库撞上 | 第二步事务里 `SELECT ... FOR UPDATE` 锁住 `images` 行；`process` 那边用同样的锁 |
| 删除和索引 worker 撞上 | 先删 `vector_index_jobs` 并提交，再删 DB 行；worker claim 时行已经没了，会跳过 |
| 删除和观察表重建撞上 | `ObservationIndexService` 有 `_lock_writes_in_session()`，删除路径用同一把锁 |
| 两个请求同时删同一张图 | 用 `DELETE ... WHERE id = :id RETURNING id` 看有没有真删到，没删到返回 404 |

## 验收

删之前先看一遍：

```bash
curl -s -X DELETE 'http://127.0.0.1:2059/api/images/<IMAGE_ID>?dry_run=true' | jq
```

删完在 PG 里对一遍，下面几个数都应该是 0：

```sql
SELECT 'crops'      , count(*) FROM person_crops              WHERE image_id = :image_id
UNION ALL SELECT 'observation', count(*) FROM person_observation_index WHERE image_id = :image_id OR crop_id = ANY(:crop_ids)
UNION ALL SELECT 'face_cache' , count(*) FROM crop_face_extractions   WHERE crop_id = ANY(:crop_ids)
UNION ALL SELECT 'reid_fb'    , count(*) FROM reid_match_feedback     WHERE query_crop_id = ANY(:crop_ids);
```

Milvus 那边用 `query` 查，不要用计数：

```python
collection = MilvusVectorIndex(settings)._collection(object_type)
res = collection.query(expr=f'object_id in ["…"]', output_fields=["object_id"])
assert not res, (object_type, res)
```

Milvus 的删除是标记删除，`num_entities` 不会马上降下来，用计数判断会误判成没删掉。

接口层面还要确认这几处不再返回被删的素材：

- `POST /api/search/person-crops`
- `POST /api/search/semantic/person-crops`
- `GET /api/search/observations`
- `/api/person-crops?image_id=…`
- `/api/media/counts` 的计数相应减少

## 几个坑

`face_embeddings.image_id` 指向的是登记照。删掉这张图会连带删掉那个人的人脸库条目，之后
`mode=face` 的人物轨迹会缺一段。建议删除前统计出来放进 `warnings`，如果这些行的 `person_id`
不为空，要求调用方显式传 `?confirm_face_removal=true`。

删除不可逆。想留退路就学 `cleanup_media_before.py`，把待删的行先写进
`cleanup_before_<时间戳>` 备份表，什么时候真清理由 DBA 定。那里已经有 `backup_tables()` 和
`insert_backup()`。

有几个东西这个接口不该碰：`persons`（人物档案和素材是两回事）、`images.video_url` 指向的源视频
（多帧共用，得走视频级删除）、`video_streams`（已经有 `DELETE /api/streams/{id}`）。

## 落地顺序

| 阶段 | 内容 | 怎么算做完 |
| --- | --- | --- |
| P0 | 单张 `DELETE /api/images/{id}`，Milvus 不可用就 503，支持 `dry_run` | 上面的 PG 和 Milvus 自检全为 0，五个接口回归通过 |
| P1 | 批量删除，带 `limit` 和默认开启的 `dry_run` | 一批 500 张，无外键报错，无残留向量 |
| P2 | 账本表 + 后台补偿，改成 `202 + task_id` | 把 Milvus 停掉，数据仍能最终一致 |
| P3 | 视频级删除，连带帧、片段和源文件 | 按 `video_url` 聚合后整体清理 |

## 测试清单

- 只有图片、没有裁剪的素材
- 带 3 个裁剪、1 条 ReID 反馈的素材
- 作为 `video_streams.last_frame_image_id` 的帧
- 已登记人脸的素材，确认 warning 和二次确认都生效
- 同一个 id 删两遍，第二次返回 404
- `dry_run=true` 前后对比行数和 Milvus 查询，必须完全一致
- Milvus 停掉时删除，应返回 503 且 DB 未改动
- 删除过程中并发调用 `process`，不应留下悬挂裁剪
- 脏数据：`attributes` 为空、`vl_embeddings` 缺行的素材也要能删

## 可以直接用的东西

| 能力 | 位置 |
| --- | --- |
| 三个集合的访问与删除 | `app/services/vector_index.py` 的 `MilvusVectorIndex._collection()` |
| Milvus 可用性预检 | 同文件，`probe()` / `_require_available()` |
| 备份表、按 id 删除、Milvus 清理、文件删除的参考实现 | `scripts/cleanup_media_before.py` |
| 路径越权校验 | `app/services/semantic_search.py` 的 `_fingerprint()` |
| 观察表写入锁 | `app/services/observation_index.py` 的 `_lock_writes_in_session()` |
| 持久化队列（账本照这个写） | `app/models/vectors.py` 的 `VectorIndexJob`、`app/services/vector_index_queue.py` |
| DELETE 端点写法参考 | `app/api/media.py` 的 `delete_stream()` |
| 依赖注入 | `app/api/deps.py` 的 `DBSession` / `AppSettings` |

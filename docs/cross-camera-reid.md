# 跨摄像头 ReID：阶段二架构设计

阶段一（已实现）提供的是可靠的单 crop 身份检索：SapiensID 向量持续入库、以图找人、按人员底库聚合的
`mode=reid` 轨迹。它给出的是**逐 crop 的概率性匹配列表**，不是跨摄像头的连续轨迹。本文档定义阶段二
——tracklet 与持久 global identity——的数据模型、算法边界与验收标准。阶段二动工前，本文档是唯一
契约；实现偏离时先改这里。

一条必须贯穿产品文案的原则：**ReID 输出的是"摄像头出现轨迹"（某人在哪些镜头、哪些时段出现过，
带置信度），不是盲区内的连续物理路径。** 两个摄像头之间发生了什么，系统不知道，也不应该假装知道。
阶段一只返回同时具备摄像头来源和真实 `captured_at` 的 crop；缺少采集时间的数据会被排除并返回警告，
不能用数据库 `created_at` 冒充摄像头出现时间。

## 数据模型

四张新表。全部软删除（`deleted_at`），删除某人时级联匿名化其 tracklet 而不是物理删除，保证计数
统计可复算。

### CameraTracklet —— 单摄像头内的一段连续出现

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | UUID PK | |
| camera_id / stream_id | UUID, 索引 | 来源镜头 |
| started_at / ended_at | datetime, 复合索引 (camera_id, started_at) | 段的时间区间 |
| track_key | str | 单镜追踪器给的短时 id（现有 line-crossing 的 track_id 可作雏形） |
| sample_count | int | 归属的 crop 数 |
| representative_crop_id | UUID FK | 代表帧（质量评分最高） |
| gallery_vector | JSON/BLOB, 可空 | 聚合后的 tracklet 向量（见下），同时写入 Milvus `reid_tracklets` |
| quality_score | float | 段级质量（样本数、清晰度、遮挡比例的加权） |
| global_identity_id | UUID FK, 可空, 索引 | 关联结果；NULL = 尚未关联 |
| model / model_version | str | 生成 gallery_vector 的 ReID 模型，跨版本不可比 |

生命周期：单镜追踪器闭合一个 track（超时或出画）即写入；`ended_at` 超过保留期（对齐现有
`MEDIA_RETENTION_DAYS`）后随 crop 一起清理。

### TrackletSample —— tracklet 与 crop 的关联

| 字段 | 类型 |
| --- | --- |
| tracklet_id / crop_id | UUID, 复合唯一 |
| embedding_weight | float（质量评分归一化后作聚合权重） |

### GlobalIdentity —— 跨镜身份

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | UUID PK | |
| person_id | UUID FK, 可空, 唯一 | 已知人员锚定；NULL = 匿名身份 |
| anchor_source | enum: face / manual / none | 身份是怎么锚定的 |
| label | str | 匿名身份的展示名（如「目标 A」） |
| created_at / last_seen_at | datetime | |
| status | enum: active / merged / dissolved | merged 时 `merged_into_id` 指向存活方 |

### TrackletAssociation —— 关联判决（证据可回放）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| tracklet_id | UUID FK | |
| global_identity_id | UUID FK | |
| score | float | 校准后的关联置信度 |
| evidence | JSON | 各因子得分：reid 相似度、时间窗、拓扑、人脸命中、冲突检查结果 |
| decided_by | enum: auto / human | |
| created_at | datetime | |

CameraTransition（镜头拓扑）单独一张配置表：`from_camera_id, to_camera_id, min_seconds,
max_seconds, enabled`。初期人工维护；有数据后可从已确认关联中统计学习。

## 算法边界

### Tracklet 聚合与代表帧

- tracklet 向量 = 各样本 ReID 向量按质量权重加权平均后重新 L2 归一化。样本 ≥3 时丢弃与均值余弦
  最低的 10% 再聚合（去掉遮挡/截断帧）。
- 质量评分：检测置信度 × 分辨率因子（短边/256 封顶 1.0）× 完整度（bbox 是否贴边截断）。
  阶段一已知教训直接沿用：纯人脸/头像不得作为人体 gallery 成分。
- 代表帧 = 质量评分最高的样本，供 UI 展示与人工复核。

### 候选召回与关联决策

按顺序过滤，每步产出写进 evidence：

1. **召回**：tracklet 向量查 Milvus `reid_tracklets`，top-k（同镜头段另行处理）。
2. **时间窗**：若存在 CameraTransition(from, to)，要求时间差落在 [min, max]；无拓扑记录时退化为
   全局宽窗（可配，默认 24h）并降权。
3. **冲突排斥**：候选身份在重叠时间段内已出现在第三个镜头 → 直接否决（一个人不能同时在两处）。
4. **同镜聚合**：同一摄像头内时间相邻（gap < 可配阈值）且互相匹配的 tracklet 先合并成段，
   避免把一次逗留拆成多条关联。
5. **人脸锚点**：段内任一 crop 有 `known` 人脸事件 → 直接锚定该 person 的 GlobalIdentity，
   ReID 分数只作佐证；人脸与 ReID 指向不同人时标记冲突进人工队列，不自动合并。
6. **置信度校准**：原始余弦经 Platt scaling（用验收集拟合）映射为概率；低于自动阈值高于展示
   阈值的进「疑似」而非自动关联。

已知人员流：人脸锚定（anchor_source=face）。未知人员流：首个无法关联到既有身份的高质量 tracklet
生成匿名 GlobalIdentity（label 自动编号），后续 tracklet 按同一决策链关联；匿名身份日后被人脸
锚定时升级为已知，历史关联保留。

### API 草案

```
POST /api/tracking/targets              以图创建追踪目标（生成匿名 identity + 立即回溯检索）
GET  /api/identities/{id}               身份详情 + 锚点来源
GET  /api/identities/{id}/timeline      按摄像头聚合的出现区间：
                                        [{camera, entered_at, left_at, confidence, evidence_refs}]
POST /api/identities/{a}/merge/{b}      人工合并（b merged_into a，关联记录保留 decided_by=human）
POST /api/tracklets/{id}/detach         人工拆分：tracklet 脱离身份，回到未关联池
```

timeline 是产品层唯一入口：返回 camera A -> B -> C 的**出现区间序列**，每段带置信度与证据引用，
绝不插值补盲区。

## 验收指标（真实多摄像头数据，不是公开集数字）

| 指标 | 门槛（初版建议） |
| --- | --- |
| Rank-1 / mAP（crop 级检索，标注对） | 相对阶段一基线 +10pp Rank-1 或同召回下假匹配率显著下降 |
| IDF1（tracklet 关联后） | ≥ 0.7 起步，随拓扑数据完善提升 |
| 错误合并率（不同人并成一个身份） | < 2%，这是最伤信任的错误，宁可拆分保守 |
| 错误拆分率（同人裂成多身份） | < 15% 起步 |
| 端到端延迟（crop 落库 → 可检索） | p95 < 30s（含队列） |
| 积压恢复 | 1h 停机积压在 2h 内消化完 |

验收集：至少 2 个真实摄像头、含同衣近似负样本、遮挡、低分辨率、跨时段。没有这个集合就没有
调阈值的依据，阶段二不应开始自动关联。

## 阶段一遗留的进入条件

1. 真实 Milvus 上跑通 `reid_person_crops` 建库/检索（本仓库测试用 mock，未验证真实实例）。
2. AGX 上 ReID 服务实测吞吐与显存（CPU 本机 ~0.4s/张，Jetson 数字未知）。
3. square-pad 修复前入库的向量全部重建。
4. 单镜追踪器产出稳定 track_key（现有 line-crossing track_id 需评估是否够用）。

阶段一索引的向量空间身份固定为：`model + checkpoint_revision + embedding_dim +
preprocess_version + milvus_namespace + collection`。服务 `/ready` 必须在模型实际加载并对
完整推理资产清单（backbone/config、YOLO pose、DFA face aligner 与 aligner config）做复合 SHA-256
后才返回成功；缺字段或任一字段不一致都拒绝检索/入库。ReID 相似度固定使用 COSINE；其他
`MILVUS_METRIC_TYPE` 会被明确拒绝，不能沿用余弦阈值解释 L2 距离。
`MILVUS_NAMESPACE_ID` 是数据库的逻辑身份，迁移同一 Milvus 数据库时保持不变，切换到另一套库时必须
更换。这样 SQL coverage marker 不会误把另一台 Milvus 中不存在的向量当作“已完成”。
coverage marker 在数据库中按 `(object_type, object_id)` 唯一，并通过短事务锁串行写入；并发重建可以
重复执行幂等的外部向量 upsert，但不能重复计算 SQL 覆盖率。

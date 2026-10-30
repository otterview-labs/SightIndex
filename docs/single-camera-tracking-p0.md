# 单摄像头人员跟踪：P0 方案与改动契约

本文档是 P0（跟踪与计数改造）的唯一契约。实现偏离时先改这里。

阶段一的计数链路已经上线，但跟踪质量不达标：过线计数漏计、串号、快走的人断轨。本文档先给出
可验证的归因，再把 P0 拆成「已落地」和「待评审」两部分——**跟踪输入与裁剪存储解耦已经实现，
ByteTrack 与高帧率计数线还没动手**。

一条贯穿的原则：**跟踪可以宽，存储必须严。** 一个人走进门口时大部分时间是被画面边缘切掉的，
而保护裁剪质量的那条规则恰好会删掉这些框。跟踪断在这一刻，计数就丢在这一个人身上。

---

## 1. 问题归因（按影响排序）

| # | 根因 | 证据 | 影响 |
| --- | --- | --- | --- |
| 1 | **跟踪输入继承了裁剪质检规则** | `stream_runtime.py` 先 `quality_filter_detections` 再 `_line_crossings`（改前）；规则见 `frame_processing.py:_is_quality_detection` | 走进/走出的人被提前删除，轨迹断在边缘 → 漏计 |
| 2 | **低帧率**（默认档 2 秒；**线上在跑的这路是 0.2 秒，不成立**） | `app/models/media.py:137` 默认 `frame_interval_seconds=2.0`；线上 `0b736ad1`（摄像头 33.156）= `0.2` | 5 路 `error` 的万科流是 2.0，但它们没有上游；**当前唯一在跑的那路不受此影响** |
| 3 | **手写贪心匹配** | `video_processing.py:_match_track`，半径 `line_crossing_match_distance=0.32`，无运动模型、无外观、无匈牙利 | 两人靠近即串号 |
| 4 | **计数状态机过简** | `PersonTrack.counted` 一次性锁死 | 进-出-再进只算一次 |
| 5 | **回滚语义与外部跟踪器冲突** | `stream_runtime.py` 后压时回滚 `tracks` + `next_track_id` | 跟踪搬到服务端后无法回滚，后压期间过线会静默丢失 |

### 1.1 根因 1 的真相：线上删人的是置信度门槛，不是贴边规则

这一条最初推错了，2026-10-16 在 `192.168.33.140` 上实测后修正。

**线上跑的构建根本没有贴边规则。** `/data/SightIndex` 的 git HEAD 是 `fd163a3`（2026-05-21），
它的 `_is_quality_detection(self, detection)` 只做三件事，签名里连 `frame_width / frame_height` 都没有：

```python
if detection.confidence < person_crop_min_confidence: return False   # 线上 = 0.10
if width  < person_crop_min_bbox_width:                return False   # 线上 = 60
if height < person_crop_min_bbox_height:               return False   # 线上 = 120
```

没有 `_touches_frame_edge`，也没有 `person_crop_require_whole_body`。仓库里的贴边规则是后来加的，
只存在于本地版本。上一版文档说的「默认档里唯一在删人的规则是贴边即丢」，只在本地默认档成立，线上不成立。

线上实测（`stream_diag` 诊断日志；journal 只保留最近约 24.5 小时，共 38,471 条）：

| 指标 | 数值 |
| --- | --- |
| 探测器原始检出 `sum(raw)` | 6,930 |
| 裁剪过滤后 `sum(quality)` | 6,338 |
| **被裁图规则丢掉** | **592（8.5%）** |
| `reason=quality_filtered`（整帧有人、却全被删光） | **203 帧** |
| `reason=no_line_crossing` | 3,970 帧 |
| `reason=no_raw_detections` | 34,280 帧（89.1%） |
| 同期计数事件 | 18 次 |

**203 帧有人经过、但计数链路看到的是空** —— 这是漏计的上界，因为「全部被过滤」的帧连 crop 都没有，
事后从库里查不到，只有诊断日志记得。

抽样拆解（50 帧重跑 `/predict`，与线上同参数 `conf=0.05 / iou=0.5 / imgsz=1920 / max_det=100`）：
16 个被丢的检出里 **16 个都是 `conf<0.10`**，宽度/高度规则 **一次都没触发**。其中一个是
`bbox 153×370`、底边中心正落在计数线上的大目标 —— 一次被直接吞掉的过线。

结论：

- 漏计的真身是 `PERSON_CROP_MIN_CONFIDENCE=0.10`。因为 `YOLO_SERVICE_CONFIDENCE=0.05` 让探测器
  交出 `[0.05, 0.10)` 这一档，而裁剪规则恰好把这一档整段删掉。
- 60×120 的最小尺寸门槛在这次样本里没拦住任何人（1920 输入下中景人物轻松过线），
  它仍然可能在小目标/远景上咬人，但不是当前的主要损失。
- **`tracking_min_confidence` 不能等于 `0.10`。** 那等于把同一个门槛抄两遍，新拆分一个都救不回来
  （实测：0.10 救回 0 个，0.05 救回 16/16）。默认值已改为 `0.0`，理由见 2.1。

**教训：先量再改。** 上面这张表花了不到一小时就拿到了，而它推翻了我此前基于本地代码写下的判断。

---

## 2. P0-1：跟踪输入与裁剪存储解耦 —— 已落地

### 2.1 改了什么

跟踪读**探测器原始输出**，裁剪存储仍走原来的质检规则。两条链路的判定互相独立：

```text
detector ─┬─ tracking_filter_detections ──→ _line_crossings ──→ 计数事件（可无 crop）
          └─ quality_filter_detections  ──→ process_image   ──→ person_crops（严格）

过线判定只看跟踪线；能不能存 crop 由 storable_crossings 再拆一次。
一个人被遮挡着过线，照样计一次；只是这一次没有裁剪图可存。
```

新增配置（`app/config/settings.py`，紧跟 `person_crop_*`，便于对照）：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `TRACKING_MIN_CONFIDENCE` | `0.0` | 探测器的下限就是下限。跟踪链路不再就置信度发表第二次意见 |
| `TRACKING_REQUIRE_WHOLE_BODY` | `false` | 跟踪默认不因贴边丢人 |

`tracking_min_confidence` 默认取 `0.0` 而不是 `0.10`，是被上面的实测逼出来的：探测器已经用自己的
下限（线上 `YOLO_SERVICE_CONFIDENCE=0.05`）筛过一遍，跟踪链路再叠一个更高的门槛，等于把探测器
刚认下来的人悄悄删掉。线上那 592 个被删的检出全部落在 `[0.05, 0.10)` 这一档——正是被第二道门槛吃掉的。
裁剪链路保留它自己的高门槛（仓库默认 0.70 / 线上 0.10）用于存储质量，两条链路各管各的。

新增接口（`app/services/frame_processing.py`）：

| 方法 | 作用 |
| --- | --- |
| `is_quality_detection(...)` | 裁剪规则的单框判定（公开化，供脚本与测试复用） |
| `tracking_filter_detections(...)` | 跟踪输入；宽于裁剪 |
| `is_trackable_detection(...)` | 跟踪规则的单框判定 |
| `storable_crossings(...)` | 把过线框拆成「可存」与「不可存」，并**保留原始索引** |

调用点改动：

- `app/services/stream_runtime.py`：直播循环，跟踪用 `tracking_filter_detections`，存 crop 用 `storable_crossings`
- `app/services/video_processing.py`：上传视频，同一处理

**索引对齐是这次改动里最容易写错的地方。** 旧代码用 `zip(crossings, crops)` 建立
`crossing.detection_index → crop` 的映射——那是因为旧链路里每个过线框必然通过裁剪质检，才能一一对应。
一旦跟踪输入变宽，这个假设就不成立了：`process_image` 会再过滤一次，`zip` 会错位，计数事件就会挂到
别人的 crop 上。所以改为由 `storable_crossings` 显式返回索引，再用 `dict(zip(indices, crops))`。

### 2.2 实测证据

**合成回归**：同一段两帧、同一个人在左边缘从 `h=40` 走到 `h=60` 越过画面中线（100×100 帧）：

```text
OLD  crop filter       detections kept per frame: 0  crossings counted: 0
NEW  tracking filter   detections kept per frame: 1  crossings counted: 1
```

**线上真实数据**（AGX `192.168.33.140`，2026-10-16）：把 50 帧真实画面按线上同参数重跑探测器，
再分别套用两套规则，用 `tracking_min_confidence` 扫了两个取值：

```text
                            tracking=0.10   tracking=0.05
被裁图规则丢掉的检出              16              16
  其中能救回                     0              16      <- 含 1 个正落在计数线上的
```

`0.10` 一个都救不回来，因为丢的原因就是「`conf < 0.10`」，而旧默认值恰好也是 `0.10`。
这就是默认值改成 `0.0` 的直接依据。

回归测试：`tests/test_tracking_filter_split.py`（10 个用例），其中
`test_process_video_path_counts_an_edge_clipped_person_without_storing_a_crop` 走完整视频链路，
断言 `counting_events_created == 1` 且 `crops_created == 0`；
`test_tracking_accepts_everything_the_detector_reports_by_default` 钉住新默认值，
并复现线上那一种 `bbox 153×370 / conf 0.072` 的丢人场景。

### 2.3 验收与边界

- 全量测试：`652 passed, 7 failed`，7 个失败为既有的本地 `.env` 泄漏问题（`FACE_EMBEDDING_PROVIDER=opencv`
  进入 `Settings()` 默认值），与本次改动无关——可用 `FACE_EMBEDDING_PROVIDER=insightface` 复跑验证。
- 这个泄漏比想象中更脏：某些用例会把 `.env` 的值写进 `os.environ`，而 `pydantic-settings`
  即使传了 `_env_file=None` 也照样读 `os.environ`。**所以凡是依赖 `Settings()` 默认值的用例都不可靠**，
  必须像本文件其它用例那样把阈值显式钉住。写入者的位置还没定位到（`os._Environ.__setitem__`
  的类级插桩没抓到，说明是别名或 C 层写入），作为遗留问题记在 §10 之后。
- `ruff check` 在改动文件上干净。
- **边界**：过线但无可存 crop 时，仍然会创建一帧 `images` 记录（证据帧），该图不计入
  `image_with_crops_count`。这是有意的：计数是业务结果，裁剪图是便利。

---

## 3. 度量工具：先量，再决定后面要不要做

`.env` 里 `STREAM_DIAGNOSTICS_ENABLED` 默认是 `false`，所以线上默认**没有**跟踪诊断日志可查；
本地 `logs/` 里那两个文件是旧的、不含 `stream_diag` 的残留。

新增 `scripts/measure_tracking_gap.py`，两条测量路径：

```bash
# 读取线上日志（需要先开启诊断并重启）
STREAM_DIAGNOSTICS_ENABLED=true
.venv/bin/python scripts/measure_tracking_gap.py --log logs/api.log

# 或者直接重跑真实帧（需要 GPU，但不用重启）
.venv/bin/python scripts/measure_tracking_gap.py --frames data/frames --max 200
.venv/bin/python scripts/measure_tracking_gap.py --video data/videos/door.mp4 --interval 0.5
```

关键读数是 `only_tracked`——被跟踪保留、被裁剪规则丢掉的那部分：

- 占 raw 比例**高** → 根因 1 成立，P0-1 立竿见影
- 接近 **0** → 根因 1 不是瓶颈，瓶颈在采样率（根因 2），直接进 P0-2

**P0-2 之前必须先跑一次这个工具。** 否则是在猜。

---

## 4. P0-2：高帧率计数线与入库线解耦 —— 待评审

### 4.1 方案

```text
计数/跟踪线   5~15 fps   轻量检测 + ByteTrack → 只产出轨迹与过线事件
入库/索引线   现状低频   只对「新出现的轨迹」截代表帧 → SapiensID / Milvus
```

两线分离后计数精度上升，而 Milvus 索引成本不变（避免一个人存 127 张 crop 的老问题——
见 `app/services/appearance_tracker.py` 的 docstring）。

计数线有两种落地形态，**这是第一个需要拍板的决策点**：

| 形态 | 做法 | 优点 | 代价 |
| --- | --- | --- | --- |
| A. API 内独立采集循环 | API 再拉一路流，高频采样 | 无新组件 | API 进程再多一路解码；与现有「一进程一线程」约束叠加 |
| B. YOLO 服务自己拉流 | 服务端持流、跟踪、只把过线事件回吐 | 帧不再落盘，带宽与磁盘都省 | 需要改服务契约（见下），运维需给更高权限 |

倾向 **B**：帧不必落盘，且这是唯一能让跟踪真正跑在帧流上的方式。

### 4.2 计数线参数（新增，均需评审）

| 配置 | 建议默认 | 说明 |
| --- | --- | --- |
| `COUNTING_PIPELINE_FPS` | `8.0` | 计数线采样率；低于 5 帧间关联质量骤降 |
| `TRACKING_TRACKER` | `bytetrack` | `bytetrack` / `botsort` |
| `TRACKING_TRACK_BUFFER` | `30` | 轨迹丢失后保留帧数 |
| `TRACKING_TRACK_HIGH_THRESH` | `0.50` | 第一阶段关联阈值 |
| `TRACKING_TRACK_LOW_THRESH` | `0.10` | 第二阶段用的低分框下限——**这正是 P0-1 要保住的那批框** |
| `TRACKING_NEW_TRACK_THRESH` | `0.60` | 新轨迹确认阈值 |

---

## 5. P0-3：跟踪服务契约扩展 —— 待评审

现有契约是**无状态、按图片路径**的（`frame_processing.py:YoloServicePersonDetector`）：

```jsonc
POST /predict
{"task":"det","image_path":"/abs/frame.jpg","conf":0.25,"iou":0.5,"imgsz":1280,"max_det":100}
```

ByteTrack 的 `persist=True` 需要稳定会话，所以协议必须扩，且**保持向后兼容**（现有解析器已经
容忍多余字段，见 `_is_person_item` / `_parse_bbox` / `_parse_confidence`）。

### 5.1 建议契约

```jsonc
POST /predict
{
  "task": "track",
  "session": "<stream_id>",      // 会话键；同一路流必须稳定
  "frame_index": 12345,          // 单调递增；幂等去重的依据
  "image_path": "/abs/frame.jpg",
  "conf": 0.25, "iou": 0.5, "imgsz": 1280, "max_det": 100,
  "tracker": "bytetrack"
}
→
{
  "width": 1920, "height": 1080,
  "detections": [
    {"track_id": 7, "label": "person", "score": 0.81, "bbox": [x1,y1,x2,y2]},
    {"track_id": 8, "label": "person", "score": 0.22, "bbox": [x1,y1,x2,y2]}
  ]
}
```

- 字段名继续用 `detections`/`boxes`，只在每个框上**增加** `track_id`，旧客户端不受影响。
- `Detection` 增加 `track_id: int | None = None`（`@dataclass(frozen=True)`，带默认值即向后兼容）。
- 新增 `POST /track/close {"session": "..."}`，释放会话内的跟踪器状态。

### 5.2 会话生命周期

- 服务端 `{session: tracker}`，空闲超过 `idle_seconds` 回收（对齐 `LINE_CROSSING_TRACK_IDLE_SECONDS`，建议 30~60s）。
- 现有代码在计数线变更时会重置 `tracks = {}` / `next_track_id = 1`
  （`stream_runtime.py` 的 `next_line_value != counting_line_value` 分支）——**这正是调用 `/track/close` 的钩子**。
- 流停止 / 重连时同样要 close，否则会话泄漏。

---

## 6. P0-5：计数状态机 —— 待评审

替换 `PersonTrack.counted` 的一次性闩锁：

| 规则 | 配置建议 | 解决的问题 |
| --- | --- | --- |
| 轨迹确认 | `LINE_CROSSING_CONFIRM_FRAMES=3` | 一次误检立刻建轨并计数 |
| 磁滞带 | `LINE_CROSSING_HYSTERESIS=0.02` | 在线附近抖动反复计数 |
| 双向分别累计 | 已有 `direction` 字段 | 进出分别统计 |
| 允许同一轨迹多次过线 | 移除 `counted` | 进-出-再进应计 2 次进 |
| 事件携带 `track_id` | 新增字段 | 为阶段二 tracklet 提供 `track_key` |

阶段二已经明确依赖这一点：`docs/cross-camera-reid.md` 把「单镜追踪器产出稳定 track_key（现有
line-crossing track_id 需评估是否够用）」列为进入条件，验收门槛 IDF1 ≥ 0.7。

---

## 7. 回滚语义 —— 待评审（易被忽略）

现状：向量队列满时，`stream_runtime.py` 回滚 `tracks` 与 `next_track_id`，删掉帧文件并退避重试。
其含义是「这次过线在内存里发生过但没落库，靠回滚让它在下一帧重新发生」。

跟踪搬到外部服务后**这个回滚失效**：服务端轨迹已经前进，侧别已经翻转，不会再有第二次过线事件，
计数会静默丢失（不是重复）。

建议改为**不依赖跟踪器回滚**：

- 进程内维护 `pending_crossings`，退避期间把已判定但未落库的过线事件暂存；
- 下一次成功提交时一并落库（有 crop 就挂上，没有就 `crop_id=None`）；
- 用 `(session, frame_index, track_id)` 做幂等键，重复投递同一帧不得重复计数。

这样计数对后压的鲁棒性与跟踪器实现无关。

---

## 8. 评测闭环 —— 骨架，待评审后再落文件

没有这个，上面所有参数都是盲调。复用 `docs/reid-walkthrough-calibration.md` 的思路，
把「ReID 走查」扩展成「计数校准」。

```text
scripts/evaluate_line_crossing.py   (骨架，尚未落地)

输入
  --video        一段真实门口视频
  --line         计数线四点（归一化）
  --truth        CSV：每行一个真实过线事件
                 timestamp_ms,direction   ("a_to_b" | "b_to_a")
  --out          结果 CSV

流程
  1. 用 VideoProcessingService.process_video_path 跑一遍，收集 CountingEvent 及其 track_id
  2. 按 |Δt| ≤ --tolerance（默认 1.5s）把预测事件与真值配对（贪心，方向必须一致）
  3. 输出指标：
       计数误差率 = (漏计 + 多计) / 真值总数        目标 < 10%
       分方向误差   a_to_b / b_to_a 分别统计
       ID 切换代理  过线事件里 distinct(track_id) 与真值人数之比  目标 < 1.15
  4. 写 --out，并打印未配对的预测/真值，便于逐条看错在哪
```

**前置条件**：需要一段带人工标注的真实门口视频。没有它，P0-2 之后的每一步都无法验收。

---

## 9. 明确不做的事

1. **不在 2 秒采样下上 ByteTrack**——装上也会退化，白改。
2. **不用 SapiensID 做逐帧外观特征**——412M 参数 / 1.5GB 常驻，逐帧不可行；那是 OSNet（2.2M）
   的位置。SapiensID 继续只做跨镜身份检索。
3. **不为「跟踪」换检测模型**——先跑 `measure_tracking_gap.py` 拿到数据。若 `raw` 本身就少，
   再考虑 `yolo11n → s`；若 `raw` 多而 `quality` 少，换模型无用。
4. **不把 `person_crop_*` 直接调松**——那会重新引入空白裁剪互相匹配的问题
   （见 `settings.py` 中 `person_crop_min_confidence` 的注释）。解耦就是为了避免这个取舍。

---

## 10. 决策点清单（评审时逐条拍板）

| # | 决策 | 备选 | 影响 |
| --- | --- | --- | --- |
| D1 | 计数线形态 | A 进程内 / **B 服务端持流** | 契约改造范围、磁盘与带宽 |
| D2 | 跟踪器 | **ByteTrack** / BoT-SORT（需外观 embedding） | 是否要引入 OSNet |
| D3 | 外观关联是否进 P0 | 是 / **否（先纯运动）** | 工期与 GPU 预算 |
| D4 | 回滚方案 | 跟踪器回滚 / **pending 事件 + 幂等键** | 后压期间计数是否可信 |
| D5 | 检测模型是否升级 | 先量再定 | 运维成本 |
| D6 | SapiensID 许可 | 保持（非商用风险）/ 换许可更宽的模型 | 是否商用，法务决策 |

---

## 附一：改动文件索引（P0-1 已落地部分）

| 文件 | 改动 |
| --- | --- |
| `app/config/settings.py` | 新增 `tracking_min_confidence` / `tracking_require_whole_body` |
| `app/services/frame_processing.py` | 新增 `is_quality_detection` / `is_trackable_detection` / `tracking_filter_detections` / `storable_crossings` |
| `app/services/stream_runtime.py` | 直播链路：跟踪用宽过滤，存储用 `storable_crossings` |
| `app/services/video_processing.py` | 上传视频链路：同上 |
| `tests/test_tracking_filter_split.py` | 10 个回归用例 |
| `scripts/measure_tracking_gap.py` | 度量工具（日志 / 帧目录 / 视频） |

---

## 附二：线上临时补丁（2026-10-16，止血用，与 P0-1 不是同一份实现）

线上 `192.168.33.140:/data/SightIndex` 停在 `fd163a3`（05-21），**落后本仓库 29 个 commit、331 个文件**，
而且上面还有 35 个从未提交的手工改动和一堆 `.bak-*` 文件。把 P0-1 直接搬过去不现实，所以先用一个
最小补丁止血，等全量升级（下称 A）落地后**这份补丁要整段丢掉**。

改动（备份 `/data/backups/sightindex-20261016/`，补丁存档 `count-widen.patch`）：

| 文件 | 改动 |
| --- | --- |
| `app/services/frame_processing.py` | 新增 `storable_detection_indices(detections)` —— 返回通过裁剪规则的**下标**，顺序与 `process_image` 返回的 crops 一致 |
| `app/services/stream_runtime.py` | 计数改吃 `raw_detections`（`tracking_detections`）；`_skip_reason` 描述计数看到的东西；`crops_by_detection_index` 改为按 `crossings[index].detection_index` 对齐 |

**为什么必须同时改三处**：线上 `process_image` 会对传进去的 detections **再过滤一次**，只返回通过的
crops。一旦计数输入变宽，旧的 `zip(crossings, crops)` 立刻错位——把唯一那张 crop 挂到另一个人的过线上。
这不是理论风险，验证脚本里可以直接复现：

```text
新代码 crops_by_detection_index = {2: 'CROP-of-det2'}
旧代码 crops_by_detection_index = {0: 'CROP-of-det2'}   <- 挂到了 det0
```

上线前验证（`/tmp/verify_patch.py`，用 `_env_file=None` + 生产同参数）：

```text
1) 计数链路看到的检出数 = 3 （补丁前 = 1）
2) quality_filter_detections 保留 = 1 ；storable_detection_indices = [2]   （入库仍严格）
3) crop 与过线对齐正确，旧写法可复现错位
```

### 上线后实测（15 分钟，两路流在跑）

```text
诊断记录 861 帧
sum(raw) = 1176   sum(quality) = 960   -> 216 个检出(18.4%)以前压根不进计数链路
raw>quality 的帧 = 208（其中 0 帧是"整帧全被丢光"）
reason 分布：no_line_crossing 469 / no_raw_detections 389 / crossing_saved 3
quality_filtered（旧的错误标签）= 0
```

注意这个 18.4% 不能直接跟前面的 8.5% 对比：这 15 分钟检出密度是 1.37 检出/帧，
24h 基线只有 0.18，场景忙得多，而且刚恢复的 33.158 也在其中。

**关于 `12:22:29` 那条过线（复核后修正）：** 该帧整帧 crop 为 0、`crop_id` 也是 NULL，我一度据此
断言"该帧所有框都被入库规则拒掉"。后来拿这张帧直接问现在的 YOLO 服务，**这个说法不成立**：帧里
有 4 个 person（conf 0.9096 / 0.8663 / 0.2178 / 0.1595），前三个都过 60×120 的尺寸门槛，只有贴
右边框那个 34×58 的行走者被拒。所以"0 crop"更可能是**该帧的过线由那个贴边行走者触发**——补丁后的
实现按 crossing 的那批框取 `storable_indices`，于是这一帧没产出 crop。至于同一帧里三个大框为什么
也没落 crop，用线上那套手工改过的构建解释不通，这恰好是 A 要用仓库干净实现替换它的理由之一。

补丁真正的证据是总量口径，不是这一条轶事。帧图仍在
`/data/SightIndex/data/frames/0b736ad1-...20261016122229036250.jpg`，可随时复核。

**精度没有被放宽破坏：** 今天逐小时过线 3 / 4 / 2 / 1 / 5，12 点这一小时（5）与该日最忙的
小时持平；过线间隔 14 秒 ~ 11 分，不是乱打；方向 a_to_b 为主。

**crop 对齐在线上确认无误：** 补丁后 7 条过线里 5 条带 crop，**5/5 都挂在与过线同一帧的图像上，
0 条挂错帧**。带 crop 率从 100% 掉到 80% 是预期内的——现在确实会有过线来自裁剪规则拒绝的框，
那 20% 正是补丁要捞回来的那部分人。

**线上没有改的东西**（A 要补的账）：

- `video_processing.py` 上传视频链路——同源问题，但线上流量是摄像头，没动它
- 跟踪器本身：贪心匹配、`counted` 一次性锁——属于 P0-2 / P0-5
- 线上 venv 是 Python 3.10.12，且**没装 ultralytics / numpy<2 / timm / transformers**（YOLO 与 ReID
  都是独立进程），所以 A 的依赖试装是最大未知数
- 线上**没有 Vue 控制台**（`frontend/` 空，跑 `app/static/*.html` 老控制台，也被手工改过）；
  A 需要把本机构建好的 `frontend/dist` 一起带过去
- 升级会让 111 个线上从未设过的配置键落到代码默认值（其中 11 个默认是打开的），需逐条评审——明细见附三

同一天在同一台机器上顺手处理的运维项：停掉 19,542 次重启的 `agx-wanke-zlm-tunnel.service`（上游
`110.40.237.78:62370` 已死），停掉每 5 分钟必失败的 `sightindex-wanke-stream-sync.timer`，
清掉 24G 的 `/data/jetlinks-edge/start_compat.log`，把上游已断的 4 路万科流从 `error` 标成 `stopped`。

---

## 附三：全量升级（A）勘察结果（2026-10-16）

目标目录 `/data/SightIndex-next`（数据盘），源码已 rsync 到位（76M，含 `.git` 和 `frontend/dist`）。
以下是开工前必须先知道的四件事。

### 1. Python 版本钉在 3.10.12，不升 3.11 / 3.12

| 证据 | 结论 |
| --- | --- |
| 线上 aarch64 的 `onnxruntime-gpu` 轮子**只有 cp310**（现有 1.20.0 来自 Ultralytics release，`/data/compile/onnxruntime_download/` 里另躺着一个 1.23.0-cp310） | 换 3.11/3.12 就丢掉 GPU 人脸路径 |
| app 里 6 处 `from datetime import UTC` 全在 `try/except ImportError` 垫片内 | app 是刻意兼容 3.10 的 |
| 215 个 `.py` 按 3.10 语法规则解析全通过；3.11 专属运行时 API 只有一处带兜底的 `Self` | 语法层面无障碍 |
| 线上用 3.10 + 生产 venv 跑新代码回归：**23 passed** | 实测通过 |
| 但 **16 个测试文件**顶层 `from datetime import UTC`（无垫片） | 全量测试套件仍需 3.11+，属测试债 |

`pyproject.toml` 声明 `requires-python = ">=3.11"`，但部署不做 `pip install -e .`，运行时不受它约束。
反过来说：**谁要动「升 Python」，就得先解决 onnxruntime-gpu 的 aarch64 轮子。**

### 2. Schema 不用手工迁移

生产库（去掉 11 张 `cleanup_*` 手工备份表）是 **11 表 / 136 列**，新代码要 **14 表 / 179 列**：

- 3 张新表 —— `crop_face_extractions`、`reid_match_feedback`、`vector_index_capacity_locks`，
  由 `create_all` 建
- 13 个既有表上的新列 —— `_ensure_compatible_schema()` 的 26 条 ALTER 全部覆盖（逐列核对过）
- 那 11 张 `cleanup_*` 备份表总共 1.5MB，不值得回收

### 3. 配置：218 个字段，生产只设了 107 个

**111 个会落代码默认值**，其中 11 个默认是「打开」的：

`APPEARANCE_TONE_ON_INGEST`、`FACE_LIBRARY_CACHE_ENABLED`、`FACE_RECOGNITION_ON_INGEST`、
`PERSON_CROP_DEDUPE_ENABLED`、`PERSON_CROP_REQUIRE_WHOLE_BODY`、`REID_ATTRIBUTE_ANALYZE_UPLOAD`、
`REID_ATTRIBUTE_ENRICH_QUERY_CROP`、`REID_ATTRIBUTE_FILTER_ENABLED`、`REID_FACE_PRIORITY_ENABLED`、
`REID_INDEX_ON_INGEST`、`STREAM_AUTOSTART_RUNNING`

其中 4 个 `REID_*` 被 `REID_ENABLED=False` 兜住。真正要在上线前拍板的只有三条：

- **`PERSON_CROP_REQUIRE_WHOLE_BODY=True`** —— 线上从来没有这条规则，升级后**入库**会开始按贴边拒绝。
  不影响计数（P0-1 已把两条路解耦），但裁剪库产出会变少。要不要显式钉成 `false` 得拍。
- `FACE_RECOGNITION_ON_INGEST=True` —— 线上本来就在做，行为不变
- `APPEARANCE_TONE_ON_INGEST=True` —— 每张 crop 多一道 cv2 色调分析；不依赖 ultralytics
  （`appearance_attributes.py` 里那次 `from ultralytics import YOLO` 是懒加载）

### 4. `requirements.agx.txt` 在这台机器上会白拉 5GB

`ultralytics>=8.3.0` 会把 `torch` 拖进来，aarch64 上解析到 **CUDA 13 的 `nvidia-*` 轮子**（实测下到
cudnn 651MB、cusparselt 221MB、nccl 216MB、cufft 214MB…）。而该文件自己的注释就写着 torch 应由
Jetson 轮子 venv 提供。这台机器的 YOLO 是 **docker 容器**（`sightindex-yolo-person`），API 进程根本
用不到 ultralytics——它只在 `PERSON_DETECTOR=yolo`（默认值）时才会被 `__init__` 立即构造。

所以 A 的装法应该是：核心 API 集 + `insightface` + **本地那个 cp310 的 onnxruntime-gpu 轮子**，
不带 `ultralytics`；线上 `.env` 已有 `PERSON_DETECTOR=yolo_service`，正是它让 API 不必装 torch。

### 5. 其它

- 线上控制台是**手写的那一套**（`app/static/` 下 7 个文件，5 月 29 日改过；`fd163a3` 里它有
  6 个文件，当前 HEAD 里是 0 个——被 `b80f6e6 Serve the SPA from FastAPI and retire the
  hand-written console` 退役了）。**那份改动没丢**：线上 `app.js` 的 md5 是 `1f847c41…`，
  与本地 stash `33831c1「本地旧改动存档 2026-10-09」`里的**逐字节相同**，`observations.html`
  在 `796d1d1` 里；要找回随时 `git show 33831c1:app/static/app.js`。
- 本机 `frontend/dist`（1.6MB）已构建好并随源码带过去了。新 SPA 有 6 个视图（Chat / Faces /
  Monitor / Observations / Reid / Search），构建产物里的 API 调用**全是相对路径**（没有写死的
  host/port），所以换端口、换域名都不用重新构建。老控制台的 URL 会落到 SPA 的 history 路由上，
  不会 404。
- `/data` 有 109G 可用
- `data/`、`logs/`、`models/`、`.env` 有意不覆盖：新实例应**共享 DATA_DIR**（`image_url` 落的是
  `/data/...` 这种路径，`main.py` 把它挂到 `settings.data_dir`），`.env` 另行准备
- 试装时用 `STREAM_AUTOSTART_RUNNING=false` 防止试跑实例把两路正在跑的流抢过去做双份采集

---

## 附四：8022 冒烟启动结果（2026-10-16）

在 `/data/SightIndex-next` 用 3.10 venv 起 uvicorn 到 `127.0.0.1:8022`，`DATABASE_URL` 指向临时库
`sightindex_trial`、`STREAM_AUTOSTART_RUNNING=false`。全程没碰生产库、没抢流。

| 检查项 | 结果 |
| --- | --- |
| 启动耗时 / 日志 | ~3.5s 就绪，日志**零告警** |
| `/health` | 200 `{"status":"ok"}` |
| `/`（SPA） | 200 `text/html`，标题 `SightIndex 视频分析台` |
| `/assets/index-*.js` | 200，30549B |
| 深链 `/monitor` | 200 html（回落 `index.html`，history 路由生效） |
| `/openapi.json` | 69 个路径 |
| `/api/streams`（空临时库） | 200 `[]` |
| `/api/nope` | 404 `{"detail":"API endpoint not found"}` |
| **`/data/frames/<线上已有帧>`** | **200，134584B —— 共享 DATA_DIR 能直接发出线上已存在的图片** |
| `init_db()` 落点 | 只在临时库建出 3 张新表 |
| 关停后生产侧 | 8021 health 正常、11 表、2 路 running、生产库 **0** 张新表 |

**接口面差异**：线上 54 条，新代码 69 条，共同 49 条 —— **没有丢任何 API**。差的 5 条是 `/`、
`/chat-ui`、`/faces`、`/observations`、`/search`，也就是老控制台那 5 个 HTML 页面路由；新 SPA 用的是
**同一批 URL**（`frontend/src/router/index.ts` 里的 6 个 section，另加 `/chat` 重定向和
`/:pathMatch(.*)*` 兜底到 `/`），所以老书签会落到对应的新视图上，不会 404。老控制台唯一用到的
attributes 接口是 `/api/attributes/person-crops/backfill`，它保留了。

新增的 20 条集中在 `reid/*`（7 条）、`attributes/jobs`、`semantic search`、`video-position`、
`media/counts`、`persons/{id}/crops`、`visit-stats`、`face/diagnostics/crops`。

**已钉进新实例 `.env`**：`PERSON_CROP_REQUIRE_WHOLE_BODY=false`（与现状一致，升级不引入这条新的
入库规则；要开启是另一个独立决定）。

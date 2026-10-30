# AGX Orin 全量升级（A）切换 runbook

> 状态：**待评审**。评审通过后按 §2 做准备、§3 执行切换。勘察依据见
> `docs/single-camera-tracking-p0.md` 的附三（依赖/schema/配置勘察）与附四（8022 冒烟结果）。

## 0. 元信息与边界

| 项 | 值 |
| --- | --- |
| 目标机 | `192.168.33.140`（AGX Orin，Ubuntu 22.04.5 LTS，aarch64） |
| 旧树（**回滚点**） | `/data/SightIndex`，git `fd163a3` + 35 个未提交改动 |
| 新树 | `/data/SightIndex-next` |
| 监听 | `0.0.0.0:8021`，**没有反向代理**，用户直连这个端口 |
| 停机 | 约 30 秒（停旧 + 起新 + 一次 `init_db`） |
| 影响 | 控制台/API 短暂不可用；两路摄像头计数中断约 30 秒 |
| 数据库 | 迁移的增量部分是 +3 表 +13 列（幂等）；**但另含 3 条幂等的数据整理语句 + 2 个唯一索引**，生产库实测影响面见 §7 → 回滚代码不需要恢复数据库 |

本次**不做**的事：

- 不动 `/data/SightIndex`（旧树原样留着当回滚点）
- 不清库、不改数据、不删表；那 11 张 `cleanup_*` 备份表（共 1.5MB）也不动
- 不改 YOLO 容器 `sightindex-yolo-person`、ReID/qwen 相关容器与进程
- 不换检测器模型、不上 ByteTrack（属 P0-2 之后）
- 不开启那 111 个会落默认值的键里的任何新规则（只钉 `PERSON_CROP_REQUIRE_WHOLE_BODY=false`）

## 1. 基线（切换前记录；验收和回滚判定都用它）

需要在机器上以 root 执行。带 Basic Auth 的接口用 `.env` 里的凭据生成一个临时 netrc，凭据值不出这台机器：

```bash
ssh root@192.168.33.140 'bash -s' <<'EOF'
cd /data/SightIndex
umask 077
U=$(grep -m1 '^APP_BASIC_AUTH_USERNAME=' .env | cut -d= -f2-)
P=$(grep -m1 '^APP_BASIC_AUTH_PASSWORD=' .env | cut -d= -f2-)
printf 'machine 127.0.0.1 login %s password %s\n' "$U" "$P" > /tmp/.rnnetrc && chmod 600 /tmp/.rnnetrc
A="--netrc-file /tmp/.rnnetrc"

echo "unit:      $(systemctl is-active sightindex.service) / $(systemctl is-enabled sightindex.service)"
echo "health:    $(curl -s http://127.0.0.1:8021/health)"
echo "API 路径数: $(curl -s $A http://127.0.0.1:8021/openapi.json | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["paths"]))')"
docker exec edge-timescaledb psql -U postgres -d sightindex -tAc \
  "select 'running 流='||count(*) from video_streams where status='running'"
docker exec edge-timescaledb psql -U postgres -d sightindex -tAc \
  "select '最近1小时过线='||count(*) from counting_events where created_at > now() - interval '1 hour'"
docker exec edge-timescaledb psql -U postgres -d sightindex -tAc \
  "select '表数='||count(*) from information_schema.tables where table_schema='public' and table_name not like 'cleanup_%'"
echo "磁盘 /data: $(df -h /data | tail -1 | awk '{print $4}') 可用"
echo "YOLO 容器:  $(docker ps --filter name=sightindex-yolo-person --format '{{.Status}}')"
rm -f /tmp/.rnnetrc
EOF
```

2026-10-16 14:2x 的实测基线（换窗口时重测一遍）：

| 项 | 值 |
| --- | --- |
| unit | `active` / `enabled`，MainPID 2954432 |
| `/health` | `{"status":"ok"}` |
| API 路径数 | 54 |
| running 流 | 2 |
| 最近 1 小时过线 | 2（当时是低峰） |
| 表数（去 `cleanup_*`） | 11 |
| 今日过线 / 今日入库 crop | 20 / 18 |
| `/data` 可用 | 107G |

## 2. 准备（零风险，可以提前全做完）

### 2.1 备份确认

`/data/backups/sightindex-20261016/` 已有：`code.tar.gz`（13M，含 `.env`，不含 `.venv/data/models/logs`）、
`sightindex.dump`（4.9M，`pg_dump -Fc`）、`sightindex.service.bak`、`count-widen.patch`。

`code.tar.gz` 里就包含那 35 个未提交改动和手改的老控制台，回滚点不需要额外备份。

### 2.2 新树清单核对

```bash
cd /data/SightIndex-next
git log --oneline -1                       # 应为本地仓库当前 HEAD
ls frontend/dist/index.html                # SPA 产物在位
.venv-310/bin/python -V                    # Python 3.10.12
.venv-310/bin/python -c "import numpy; print(numpy.__version__)"        # 1.26.4
.venv-310/bin/python -c "import onnxruntime as o; print(o.get_available_providers())"  # 含 CUDA/TensorRT
grep -E '^(PERSON_CROP_REQUIRE_WHOLE_BODY|PERSON_DETECTOR|DATA_DIR)=' .env
```

### 2.3 `.env` 的处置

新 `.env` 是生产 `.env` 的逐键拷贝（122 键），**只多一行**：

```
PERSON_CROP_REQUIRE_WHOLE_BODY=false
```

已钉好。其余两个默认打开、且行为与现状一致的键（`FACE_RECOGNITION_ON_INGEST`、
`APPEARANCE_TONE_ON_INGEST`）**不写**，走代码默认；若想更稳可以把它们也显式写死，效果等价。

`DATABASE_URL` 必须指向生产库 `sightindex`（不是试装的 `sightindex_trial`）；`STREAM_AUTOSTART_RUNNING`
保持不设 → 默认 `true` → 新实例起来会自动把两路流接回来（这正是切换要的）。

### 2.4 新 systemd unit

写成 `/etc/systemd/system/sightindex-next.service`，逐行对应旧 unit。**只写入，不 enable、不 start**
（两个 unit 同时 enabled 会让重启后抢 8021）。

```ini
[Unit]
Description=SightIndex API (Vue console + P0-1 tracking split)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/data/SightIndex-next
ExecStartPre=/usr/bin/docker start sightindex-yolo-person
ExecStart=/bin/bash -c 'set -a; source /data/SightIndex-next/.env; set +a; exec /data/SightIndex-next/.venv-310/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8021'
Restart=on-failure
RestartSec=10
TimeoutStopSec=45

[Install]
WantedBy=multi-user.target
```

两处与旧 unit 的差异，都是故意的：

- venv 目录叫 `.venv-310` 而不是 `.venv`。venv 里的 console script shebang 是绝对路径，
  `mv` 目录会弄坏它们；我们统一用 `python -m uvicorn` 调用，所以名字不同没有影响，写清楚即可。
- `ExecStart` 里的 `.env`、`WorkingDirectory` 指向新树。

### 2.5 窗口前的四项检查

1. `ss -tlnp | grep 8021` 只有旧实例一个监听者
2. `docker ps --filter name=sightindex-yolo-person` 是 Up（YOLO 是 **host 网络**的容器，19121 端口在容器内）
3. `grep '^DATABASE_URL=' /data/SightIndex-next/.env` 指向 `sightindex`
4. 挑低峰（看 §1 的"最近 1 小时过线"，个位数最好）

## 3. 切换（停机窗口）

```bash
# 1) 记基线（§1 的命令块，存下来）
# 2) 停旧
systemctl stop sightindex.service
ss -tlnp | grep 8021 || echo "8021 已释放"
# 3) 旧 unit 取消自启（回滚时要 enable 回来）
systemctl disable sightindex.service
# 4) 装载新 unit
systemctl daemon-reload
# 5) 起新
systemctl start sightindex-next.service
# 6) 等就绪（最多 30 秒）
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8021/health)" = 200 ] && { echo "就绪"; break; }
  sleep 0.5
done
# 7) 看迁移与启动日志（应只有 3 张新表 + 13 列的 ADD COLUMN，无异常）
journalctl -u sightindex-next.service --since "-3min" --no-pager | tail -30
# 8) 设为自启
systemctl enable sightindex-next.service
```

## 4. 验收（切换后 15 分钟内）

| 检查 | 期望 |
| --- | --- |
| `systemctl is-active sightindex-next` | active，且 `Restart` 计数没有增长 |
| `/health` | `{"status":"ok"}` |
| `/api/streams` | 2 路 `running`（自动接流生效） |
| SPA | `/`、`/search`、`/observations`、`/faces`、`/reid`、`/chat-ui` 六个页面都能开（浏览器会先弹 Basic Auth；命令行用 §1 那个 netrc） |
| 静态帧 | `/data/frames/<线上已有帧>` 返回 200（共享 DATA_DIR 生效） |
| **计数继续产生** | 15 分钟内有新的 `counting_events`，且带 crop 的那几条 crop 与过线**在同一帧** |
| 日志 | `journalctl -u sightindex-next --since "-15min"` 无 traceback、无重复 5xx |
| 库 | 表数 14、列数 179（对比 §1 基线的 11） |
| YOLO | 容器仍 Up，`/health` 无检测报错 |

计数的核心验收命令：

```bash
docker exec edge-timescaledb psql -U postgres -d sightindex -tAc \
 "select ce.id, ce.direction, ce.created_at, count(pc.id) as crops
  from counting_events ce left join person_crops pc on pc.image_id = ce.image_id
  where ce.created_at > now() - interval '20 minutes' group by 1,2,3 order by 3 desc limit 10"
```

## 5. 回滚

**触发条件**（任一即回滚）：

- 30 秒内起不来，或 unit 反复重启（`Restart` 计数持续增长）
- 15 分钟内**没有任何**新 `counting_events`
- 日志持续 traceback / 数据库错误 / 大量 5xx
- 控制台不可用

**步骤**：

```bash
systemctl stop sightindex-next.service
systemctl disable sightindex-next.service
systemctl enable --now sightindex.service
curl -s http://127.0.0.1:8021/health
docker exec edge-timescaledb psql -U postgres -d sightindex -tAc \
  "select count(*) from video_streams where status='running'"     # 应为 2
```

**数据库不要恢复**：新加的表和列旧代码完全不引用。但迁移里有三处**不是纯增量**（实测影响面见 §7）：
它会按 `(object_type, object_id)` 整理 `vl_embeddings`、按 `(target, object_id)` 整理
`vector_index_jobs`，并改写 `person_observation_index.captured_at`；同时建两个唯一索引
（`ux_vl_embeddings_object_type_object_id`、`ux_vector_index_jobs_target_object`），**这两个索引在
回滚后仍会留在库里**。生产库实测：去重删 0 行、`vector_index_jobs` 是空表、`captured_at` 只改写
7/5695 行，所以回滚是安全的；万一回滚后老代码在向量路径上报 `IntegrityError`（老代码没有这两条
约束时不会有的错），一行 SQL 即可退回：

```sql
DROP INDEX IF EXISTS ux_vl_embeddings_object_type_object_id;
DROP INDEX IF EXISTS ux_vector_index_jobs_target_object;
```

这也是为什么先把 `PERSON_CROP_REQUIRE_WHOLE_BODY` 钉成 `false`——回滚前后入库规则一致，行为可比。

回滚后保留 `/data/SightIndex-next` 与日志供排查，并记录回滚原因。

已知代价：切换时段两路流的计数会有一段空洞（约半分钟），这是切换的固有成本。

## 6. 收尾（观察 3~7 天后）

- 删掉试装留下的临时库：`docker exec edge-timescaledb psql -U postgres -c "DROP DATABASE sightindex_trial"`
- 清理 `/data/SightIndex-next/logs/`（试装日志）与 `constraints.txt`
- 决定旧树去留（建议先 `mv /data/SightIndex /data/backups/sightindex-rollback-<日期>`，不要直接删）
- 把仓库这 29 个提交 + 本轮改动（P0-1、测试、两份文档）提交并 push
- `systemctl reset-failed sightindex-wanke-stream-sync.service`（历史失败状态，纯观感）
- 决定 `sightindex.service` 这个 unit 名要不要回收（新实例稳定后可把 unit 改回原名）

## 7. 执行记录（2026-10-16）

### 7.1 时间线

| 时刻 | 动作 | 结果 |
| --- | --- | --- |
| 14:36:49 | 记基线 | 11 表 / 136 列、2 路 running、最近 15 分钟 0 过线（低峰） |
| 14:37:27 | `systemctl stop sightindex` | 8021 监听者 0、旧进程残留 0 |
| 14:37:27 | `systemctl disable sightindex` | 移除 multi-user.target.wants 软链 |
| 14:37:30 | `systemctl start sightindex-next` | — |
| 14:37:33 | 就绪 | **3.5 秒**；`Application startup complete`，日志零告警 |
| 14:37:34 | `systemctl enable sightindex-next` | 建软链成功 |

停机实际约 6 秒（停旧到就绪），加 `init_db` 也只有个位数秒。

### 7.2 迁移影响面（切换前在只读前提下实测）

| 语句 | 生产库实测 |
| --- | --- |
| `DELETE` 去重 `vl_embeddings` | 4855 行，按 `(object_type, object_id)` **删 0 行**；每个对象只有 1 种 embedding_model |
| `CREATE UNIQUE INDEX ux_vl_embeddings_object_type_object_id` | 原本不存在，创建成功 |
| `UPDATE` / `DELETE` `vector_index_jobs` | **空表**，0 行受影响 |
| `CREATE UNIQUE INDEX ux_vector_index_jobs_target_object` | 原本不存在，创建成功 |
| `UPDATE person_observation_index.captured_at` | 5695 行中改写 **7 行**（合成时间戳的修复） |
| 3 张新表 + 13 个新列 + 6 行容量锁 | 全部幂等创建 |

### 7.3 验收结果

| 检查 | 基线 | 切换后 | 判定 |
| --- | --- | --- | --- |
| unit | active / enabled | active / enabled，**Restart=0** | ✓ |
| `/health` | ok | ok（外网口 `192.168.33.140:8021` 也 200） | ✓ |
| API 路径数 | 54 | 69 | ✓ |
| running 流 | 2 | 2（自动接流生效） | ✓ |
| 表 / 列 | 11 / 136 | 14 / 179 | ✓ |
| 两个唯一索引 | 不存在 | 已创建 | ✓ |
| SPA 六个页面 | 无 | `/`、`/search`、`/observations`、`/faces`、`/reid`、`/chat-ui` 全 200 | ✓ |
| 共享 DATA_DIR 帧图 | — | 200，134584B | ✓ |
| 日志 | — | 20 分钟里只有 2 条 pymilvus 弃用告警 | ✓ |
| `/data` 可用 | 107G | 107G | ✓ |

**端到端的功能验收**（最硬的一条）：切换后 **14:52:47** 出现一次真实 `a_to_b` 过线
（`22cb21d2-9673-4def-873c-ae50c9a64d5a`），它与 crop `7fa799bd-0ee5-473a-928b-576ea1fae27c`
挂在**同一帧** `...20261016145246214192.jpg` 上：`crop_id` 非空、`quality_pass=true`、conf 0.888、
bbox 182×415。**宽跟踪 → 过线计数 → 严入库 crop → 两者配对**，整条链路在实盘上闭环。

### 7.4 这次踩到的坑（写给下次）

- **YOLO 服务的检测结果在 `detections` 键里，不是 `results`。** 用错键会让每次探针都"检出 0 个"，
  把正常服务误判成坏了（这次就绕了一大圈）。
- **YOLO 容器把 `/data/SightIndex` 以只读挂进去**，宿主 `/tmp` 它看不到；探针必须用
  `/data/SightIndex/...` 下的路径。容器是 host 网络，所以 `:19121` 就是它。
- **`conf=0.05` 时模型会大量误报非人目标**（实测空走廊返回 `stop sign` 0.127 / `book` 0.067）。
  API 侧 `_is_person_item()` 只收 `person`/`0` 类，所以 `raw` 不会被这些污染。
- **`images.created_at` 不是入库时间**：它比 `captured_at` 早，差值从 40 秒到 52 分不等；
  只有 `captured_at` 与帧文件名里的时间戳一致。判断"某时刻之后存了什么"要按 `image_url` 反查，
  不能拿 `created_at` 当依据。
- 线上那套手写控制台升级后不再被挂载（`app/static` 已退役），但内容逐字节等于本地 stash
  `33831c1`，随时 `git show 33831c1:app/static/app.js` 取回。

## 附：本次升级后用户可感知的变化

- **控制台换成 Vue SPA**，URL 与老控制台一致（`/`、`/search`、`/observations`、`/faces`、
  `/chat-ui`，另加 `/reid`）。你手工改过的那套老控制台**没有丢**：它逐字节等于本地 stash
  `33831c1「本地旧改动存档 2026-10-09」`，随时 `git show 33831c1:app/static/app.js` 取回。
- **新增能力**（+20 条 API）：以图找人 `reid/*`、语义检索、属性任务与补录、人物访问统计、
  视频位置回查、`media/counts`、人脸诊断 crops 等。
- **一条 API 都没丢**；5 个老 HTML 页面路由由 SPA 的同名客户端路由接住，不会 404。
- **计数行为不变**：P0-1（宽跟踪、严入库）本来就在线上的 B 补丁里生效，A 只是把它换成仓库实现。
- **入库规则不变**：`PERSON_CROP_REQUIRE_WHOLE_BODY=false`。
- 配置上有 111 个键会落到代码默认值（其中 11 个默认是打开的，4 个 `REID_*` 被 `REID_ENABLED=false`
  兜住）。除上面钉的一条外，其余默认值都与现状等价。
- 测试套件里有 16 个文件用了顶层 `datetime.UTC`，只能跑在 3.11+；AGX 的运行时是 3.10.12
  （为了 aarch64 的 `onnxruntime-gpu` 轮子），所以在机器上只能跑 3.10 兼容的那部分测试。

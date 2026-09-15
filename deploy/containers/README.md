# SightIndex 镜像部署（独立测试实例）

该入口在宿主机已有其他业务时单独运行 SightIndex，不调用 RTX 5090 systemd 安装脚本，
不升级系统 Python/Node，不复用生产数据库、Milvus 集合、摄像头或人员数据。

首次部署按本文执行（全部步骤可由 `deploy.sh` 一键完成）；部署完成后的日常
发版、运维与回退速查见 [DEPLOY.zh-CN.md](DEPLOY.zh-CN.md)。

## 镜像与资源

主路径为源码构建一个应用镜像，API 与 ReID 复用此镜像。前端在 Node 22 构建阶段编译；
Python/CUDA 来自 PyTorch 2.10.0 / CUDA 12.8 镜像。构建后保存镜像 ID、基础镜像 digest、
源码版本与 `image-requirements.txt`，后续复用该镜像，不以滚动 tag 作为发行凭证。

API 内的 YOLO 和 InsightFace 默认走 CPU；只有显式启用的 `reid` profile 使用 GPU 0。
ReID 并发和批次均为 1。容器内存限制不是显存配额，启动前须核对 GPU 空闲显存并监测加载与推理峰值。
剩余资源不足时保持 ReID 禁用，不停止其他模型来释放显存。

默认不启动 Qwen/VLM、通用视觉嵌入或摄像头。人体 ReID 与文字/通用图像检索是不同能力：
未配置视觉嵌入时不能把文字搜索结果宣称为 Qwen 向量检索结果。

## 准备

使用项目根目录作为构建上下文：

```bash
docker build -f deploy/containers/Dockerfile \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sightindex:reviewed-release .
```

基础镜像、npm/PyPI 地址可通过 `NODE_IMAGE`、`TORCH_IMAGE`、`NPM_REGISTRY` 和 `PIP_INDEX_URL`
构建参数替换为已验证可达的镜像源。不要使用其他业务容器的文件系统作为应用镜像。

若 PyTorch 大镜像的下载不可用，可先从已校验的 Debian Bookworm 基础镜像构建相同
PyTorch/CUDA 版本的 wheel 基座，再通过 `TORCH_IMAGE` 传给上述唯一应用构建入口：

```bash
docker build -f deploy/containers/Dockerfile.torch-base \
  --build-arg BASE_IMAGE=debian:bookworm-slim \
  -t sightindex-torch:2.10.0-cu128 .
docker build -f deploy/containers/Dockerfile \
  --build-arg TORCH_IMAGE=sightindex-torch:2.10.0-cu128 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sightindex:reviewed-release .
```

`BASE_IMAGE` 也可指定已缓存、已核验来源的官方 `node:22-bookworm-slim` 镜像 ID；
该回退保留 Node 工具，但删除其默认用户以供应用建立非 root 的 UID 1000。
Python 在镜像内使用独立 venv，不修改宿主机。构建会检查 CUDA 12.8 与 PyTorch 2.10.0；
不得为绕开下载问题静默降级到 CPU wheel。最终 GPU 可用性仍须在目标机器实测。
应用构建阶段安装 `python3-dev` 与编译工具，确保 wheel 基座能编译 InsightFace，
完成后移除这些构建依赖，避免回退构建再次出现 `Python.h` 缺失。

复制 `.env.example` 到宿主机私密配置，权限设为 `0600`。为四个空密码/API key 字段生成各自独立的随机值；
不要使用 SSH 密码。数据库密码应为 URL-safe 字符串。部署配置只保存在宿主机，不进入构建上下文。
已有基础设施镜像可通过四个 `*_IMAGE` 字段指定已验证的本地 image ID，避免弱网重复下载。

创建 `MEDIA_DIR`、`MODEL_DIR`、`CACHE_DIR/api` 并允许 UID/GID `1000:1000` 写入媒体和缓存。
模型只读挂载，目录布局为：

```text
models/
  yolo11n.pt
  yolov8n-pose.pt
  sapiensid_wb12m/model.yaml
  sapiensid_wb12m/model.pth
  insightface/models/buffalo_l/*.onnx
```

模型资源须预先准备并核对哈希，不能在处理请求时自动下载。参阅
`deploy/agx/reid_service/README.md` 和 `THIRD_PARTY_NOTICES.md` 中的模型与第三方使用限制。

## 统一操作入口 `manage.sh`

`deploy/containers/manage.sh` 是运维唯一入口，封装 env 文件、release 目录、
overlay compose 文件与 profile 的组合，操作人员不再手工拼
`--env-file`/`-f`/`--profile` 参数：

```bash
manage.sh [--env-file FILE] [--release DIR] COMMAND [STACK ...]
```

命令为 `up`（默认动作）、`down`（保留卷，绝不执行 `down -v`）、`restart`、
`status`（compose ps + API 健康探测）、`logs`。可组合的 stack：

| stack | 内容 | 前置条件 |
| --- | --- | --- |
| `base`（默认） | postgres etcd minio milvus api | 无 |
| `reid` | + reid 服务 | env 中 `REID_ENABLED=true`，且 GPU 容量已核对 |
| `embedding` | + Qwen3-VL 嵌入服务与 API 接线 | env 中三个 `QWEN_*` 键 |
| `semantic` | + 语义检索设置 overlay | 已完成语义索引回填 |

部署根目录默认 `/data/sightindex-bj-test`，可用 `SIGHTINDEX_ROOT` 或后续参数覆盖。
根目录下须有 `.env` 与 `releases/<版本>/deploy/containers/`（含 base compose）；
`media/`、`models/`、`cache/` 与数据卷跨版本共享。base compose 未显式指定时取
最新一个带 base compose 的 release；`compose.embedding.yaml` 与
`compose.semantic-search.yaml` overlay 跨所有 release 从新到旧解析，
因此拆分发布的版本仍然可用。release 名不得含空格。

脚本会对比 compose 项目当前运行的 base compose 与本次目标，不一致时先告警再执行
`up`/`restart` 将按新 base 重建容器。传入与割接时一致的 env 文件即可复现整套栈，
例如恢复北京全部四个 stack：

```bash
cd /data/sightindex-bj-test
bash manage.sh --env-file .env.semantic-search-v1 up base reid embedding semantic
```

## 启动与验收

以下命令也支持将 `docker compose` 替换为已有的 `docker-compose` v2：

```bash
docker compose --env-file /data/sightindex-bj-test/.env \
  -f deploy/containers/compose.yaml config --quiet
docker compose --env-file /data/sightindex-bj-test/.env \
  -f deploy/containers/compose.yaml up -d postgres etcd minio milvus api
```

确认模型资源与 GPU 容量后，将私密配置中的 `REID_ENABLED` 改为 `true`，显式启用 profile：

```bash
docker compose --env-file /data/sightindex-bj-test/.env \
  -f deploy/containers/compose.yaml --profile reid up -d
```

必须同时验证：

- 容器 healthy，数据库与 Milvus 为独立新实例，原业务容器未被重启或修改。
- API 容器探针使用 Basic Auth 查询媒体计数以验证数据库；启用 ReID 后同时检查其 readiness。
  `/health` 仅表示进程存活；这些探针不替代 CPU 模型加载和实际推理验收。
- `/health` 返回 200；匿名访问控制台、媒体与业务 API 返回 401。
- 带认证访问控制台与 `/api/media/counts` 成功；空库计数为 0。
- 启用 ReID 后 `/api/reid/status` 报告服务和索引可用，模型指纹吻合；
  通过内部 `/embed` 对合成无人员图片执行一次调用，验证有限值、4096 维和归一化。
- CPU 人脸模型和 YOLO 能加载；合成图片不产生虚构身份。真实人员准确率需另行人工标注验收。

默认仅公开宿主机 `127.0.0.1:18030`。PostgreSQL、Milvus、MinIO 和 ReID 没有宿主机端口。
远程访问使用 SSH 本地转发；公网访问需额外配置 HTTPS 网关及访问控制，不直接修改现有 FRP。
嵌入端点配置独立视觉嵌入 API key 时验证该 key；未配置时服从已启用的 Basic Auth。
不要将 SSH 口令或上游模型提供商密钥作为公开客户端凭证。

## 数据与回退

所有 Compose 资源属于 `sightindex-bj-test`，持久卷与现有业务分离。新增实例停止时执行
同一 Compose 入口的 `down`，不得使用 `down -v`、全局 prune 或删除共享目录。

升级前保存旧镜像 ID、源码发行目录、前端与私密配置，并通过 `pg_dump` 备份本实例。
回退应用只替换 `SIGHTINDEX_IMAGE` 后重建本实例 API/ReID，兼容的新增数据库列保留；
不得盲目整库恢复而覆盖后来采集的数据。GPU 不足时停止本实例 ReID 并禁用对应功能，
保留 API、数据和现有其他服务。

## 身份与计数兼容性

本次升级为 `person_crops` 增加可空的 `person_id_source` 列，并为 `images` 增加
可空的 `processed_at` 列，不删除或猜测旧数据来源。
人工标注与人工撤销记录为 `manual`，自动人脸归属记录为 `face`。自动识别不得覆盖人工决定，
也不得覆盖来源不明的已有非空身份；历史模型判断仍单独保存在识别事件中。
旧身份需要重新分配时，先人工撤销再重新标注。回退到不理解此字段的旧程序会失去这层保护。

越线 track ID 仅用于一次视频/流会话的位置跟踪，不是跨摄像头身份。轨迹默认连续失配
2 个采样帧或 6 秒未出现后失效，可用 `LINE_CROSSING_TRACK_MAX_MISSED_FRAMES` 和
`LINE_CROSSING_TRACK_IDLE_SECONDS` 调整。空帧参与失配统计，短暂单帧漏检保留轨迹。

## 本轮修复验收

- 初始化数据库后确认 `images.processed_at` 存在；顺序及并发重复处理同一图片时，
  裁剪 ID、数量和索引任务数不增加，人工标注不改变。无检测结果也应幂等。
- 长时间打开 MJPEG 预览时，媒体计数、搜索等短请求仍可取得数据库连接。
- 人工改名归属或撤销后，旧姓名不能通过历史识别事件再次搜出该裁剪；
  超过 500 条记录之前的已标注属性仍能检索。
- 合法图片可上传、预览并处理；HTML/SVG 上传被拒绝，旧 `/data/*.html`、
  `/data/*.svg` 等非媒体路径返回 404。超限和分块超限返回 413，不留下公开文件。
- `.env.example` 与 Compose 透传三个 `UPLOAD_*` 限额：图片 20 MiB、2500 万像素，
  视频 128 MiB。图片会规范化为去元数据的 RGB PNG，动图取第一帧；图片输入与输出
  都受字节限额约束。网关应同时配置合适的请求体、并发和速率限制。

以上修复需重新构建、标记镜像后才对已部署实例生效，不能只复制宿主机源码。
按“数据与回退”保存当前镜像与备份；发现媒体访问、处理或模型兼容性回归时回退应用镜像，
保留新增兼容列，不自动删除历史重复裁剪。回退到修复前版本会重新暴露这些缺陷，
此时应维持网关隔离，禁止不可信上传，避免开放访问。

## 可选 Qwen 图文向量服务

`compose.embedding.yaml` 是现有 Compose 的可选覆盖文件，显式启用 `embedding` profile。
模型服务只监听容器网络的 18032，不发布宿主机端口，不启动数据库、摄像头、人脸或 ReID。
现有 API 通过 HTTP 调用它，API 本身不分配 GPU。

本入口固定使用 Qwen 官方 `Qwen3-VL-Embedding-2B`，2048 维、BF16、SDPA、单请求推理。
图片预处理最多 786432 像素，文本上下文最多 4096 token；这些参数属于向量空间约定，
改变参数或模型时应使用新集合，而不是覆盖既有 CLIP/Qwen 向量。
PyTorch 缓存分配器上限为整卡显存的 60%，不等于所有 CUDA 分配的硬隔离配额。
容器内存上限 10 GiB；上线前必须实测其他 GPU 业务与本服务同时运行时的峰值。

在模型服务器上直接下载，不经开发电脑转发权重：

```bash
python3 deploy/containers/download_embedding_model.py \
  /data/sightindex-bj-test/models/qwen3-vl-embedding-2b-c73fa9ca
```

下载器保存官方逐文件版本及 SHA-256 清单，支持续传，逐文件校验后才改为正式文件名。
权重固定为 `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`，
推理脚本也校验哈希；运行时禁止联网下载模型。需要更换版本时先复核下载器和启动校验。

基于已经验证的 SightIndex 镜像构建增量模型镜像：

```bash
docker build --pull=false -f deploy/containers/Dockerfile.embedding \
  --build-arg SIGHTINDEX_BASE_IMAGE=sightindex:reviewed-release \
  -t sightindex-embedding:reviewed-release .
```

在私密 env 中配置 `QWEN_EMBEDDING_IMAGE`、`QWEN_EMBEDDING_MODEL_DIR`、
`QWEN_EMBEDDING_API_KEY` 和 `QWEN_VISUAL_COLLECTION_PREFIX`。
`QWEN_EMBEDDING_API_KEY` 仅供 API 调用内部模型服务，通过
`VISUAL_EMBEDDING_UPSTREAM_API_KEY` 传入客户端；它与面向调用者的
`VISUAL_EMBEDDING_SERVICE_API_KEY` 分开，避免把原有 Basic Auth 意外改为 Bearer-only。
未设置上游专用 key 时，Qwen HTTP 客户端仍兼容原有共用 key 的配置。

若只升级已部署的旧 v4，可用 `Dockerfile.embedding-client` 做最小兼容回填：
它只添加上述上游字段和两处 HTTP 调用，不覆盖旧镜像的其他文件。脚本会拒绝不匹配
的源码结构或已经支持该字段的镜像。新源码直接构建的常规 API 镜像无需此回填。

先启动模型服务，保持原 API 不变：

```bash
docker compose --env-file /data/sightindex-bj-test/.env.qwen3vl2b-v1 \
  -f deploy/containers/compose.yaml -f deploy/containers/compose.embedding.yaml \
  --profile embedding up -d --no-deps embedding
```

启动时完成模型哈希检查、GPU 加载和文本/图片预热后，`/health` 才返回 ready。
验收内部无认证请求为 401、带 key 的中文/英文及图片请求产生有限、归一化的 2048 维向量，
并确认显存余量。随后对同一 Compose 入口执行 `up -d --no-deps api`，只重建 API；
再次验证原 Basic Auth、媒体计数、ReID readiness 和实际 embedding 调用。

升级前保存原 Compose/env/镜像 ID，并执行 `pg_dump`。回退时使用原 Compose/env 和原镜像
仅重建 API，不恢复整库、不删媒体、不重建 ReID。内部模型服务可单独停止。

该切换只替换 embedding 接口，不自动启用图文向量入库，也不更改现有页面的
姓名/结构化标签搜索。若要开放 Qwen 语义搜图，应单独评审检索入口、建立新图文集合、
回填索引并做检索效果验收，不能把向量相似度直接变成人员身份。

### 可回退的 Qwen 语义候选入口

`POST /api/search/semantic/person-crops` 与原 `/api/search/person-crops` 严格标签入口分开。
请求为 `{"query":"穿红衣服的男子","top_k":20,"filters":{}}`，状态接口为
`GET /api/search/semantic/status`。开启 `SEMANTIC_SEARCH_ENABLED=true` 后，前端默认选择
“Qwen 语义候选（需核验）”；关闭开关或连接旧后端时仍使用严格标签检索。

必须已配置独立 `MILVUS_VISUAL_COLLECTION_PREFIX`、COSINE 距离、Qwen 图文 provider，
并完成当前模型/维度的历史索引回填。相机、位置、人员、拍摄时间条件先选出 SQL 对象范围，
再以 UUID 表达式限制 Milvus 的召回范围，不以全库 Top-K 后过滤代替范围内搜索。
没有拍摄时间的记录不会被当作符合时间筛选。超过 `SEMANTIC_SEARCH_MAX_SCOPE` 个已索引
对象时要求用户缩小范围；每次最多核对 400 个候选。同文件内容且相机、位置、身份标注
相同的候选合并展示，但不删原始记录、不修改人脸/人员/ReID 数据。

`SEMANTIC_SEARCH_MIN_SCORE=0.25` 是可调的候选相似度门槛，不是置信概率，也不是已完成
通用标定的准确率阈值。页面和返回体均明确说明颜色、性别及其他条件未逐项核验。
服务不可用、索引标记存在但向量未就绪时返回 503，不伪装成没有匹配；
部分索引覆盖和候选截断也在响应中说明。状态接口的 indexed 数量是 SQL 标记覆盖，
并不表示实时远端健康。属性模型未启用时，“解析最近裁剪”按钮禁用并说明原因。

北京已验证的旧 API 镜像可用 `Dockerfile.semantic-search` 做最小更新：准备专用构建目录，
仅放此 Dockerfile、`patch_semantic_settings.py`、四个检索相关 Python 文件，以及已通过
`vue-tsc`/Vite 的 `frontend-dist/`。脚本先验证旧 settings/API/vector 文件 SHA-256，
只增加三个 settings 字段，拒绝未经核对的基线；不覆盖其他本地未发布代码。
常规新版本直接使用主 Dockerfile 构建，不运行该旧基线补丁。

验收通过后，将 `compose.semantic-search.yaml` 叠加到当前已验证的 Compose/env 入口，
固定新 API 镜像摘要，并仅执行 `up -d --no-deps api`。不重建 embedding、ReID 或数据库。
`VECTOR_INDEX_ON_INGEST` 本功能不擅自开启，页面会展示增量索引未开启的状态。

上线门槛：严格标签回归不变；红衣样本作为候选可见；不存在目标的负例、相机/时间
过滤、错误态、去重和身份不变检查通过；服务健康、无新重启。颜色反例仍可能召回相似
候选，必须展示未核验提示，不能称为准确标签命中。
快速回退：设 `SEMANTIC_SEARCH_ENABLED=false` 并仅重建 API；镜像回退则恢复此前保存的
Compose/env/镜像摘要。均不恢复整库、不删除原始媒体和 ReID 索引。

# SightIndex 镜像部署（独立测试实例）

该入口在宿主机已有其他业务时单独运行 SightIndex，不调用 RTX 5090 systemd 安装脚本，
不升级系统 Python/Node，不复用生产数据库、Milvus 集合、摄像头或人员数据。

在经过审查的源码仓库根目录运行统一入口 `bash deploy.sh`，默认选择容器部署。
`bash deploy.sh --target rtx5090` 则选择 `/opt/sightindex` 的 RTX 5090 systemd 路径，
后者的宿主机依赖与权限要求见 [通用部署文档](../../docs/deployment.zh-CN.md)。
部署方式选择、同名 Compose 项目的实例限制和验收边界统一见
[一键部署与验收](../../docs/one-click-deployment.zh-CN.md)。仅修改 `--root` 不会自动改变默认
`sightindex-bj-test` 项目名。第二实例须显式设置 `--project-name`（与私密配置中的
`COMPOSE_PROJECT_NAME` 一致），使用独立端口、媒体、缓存和数据库卷；不能用两个根目录
同时管理同名项目。已经验收的根目录通过 `project-name` 绑定项目，不能重绑定为其他项目。
容器入口不要求 `sudo`，但当前账号必须能访问 Docker 及专用部署目录。
宿主机需要 Python 3.11+、Git、tar 和 Docker Compose v2；GPU stack 还需要已配置的
NVIDIA 驱动、容器运行时与 `nvidia-smi`，脚本不安装或更改这些组件。

先准备下文的私密配置与本地模型，再执行只读检查和部署：

```bash
cd /absolute/path/to/SightIndex
bash deploy.sh --check --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env --stacks "base"
bash deploy.sh --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env --stacks "base"
```

`--check` 只检查配置、已预置的模型、工具和所选 GPU stack 的设备状态；不创建目录或 env，
不构建镜像、不启动服务、不下载模型、不写数据库。它需要已有且可读的私密配置。
默认部署不下载模型、不启动摄像头、不拉取 Git 提交。需要准备权重时，显式选择
`--prepare-models` 与已审查锁文件、离线来源及条款确认；或明确允许 `--download-models`。
只核验资产可用 `bash deploy.sh --models-only ... --check`，完整示例与版本管理见
[模型部署说明](../../docs/model-deployment.zh-CN.md)。日常操作与人工回退见
[DEPLOY.zh-CN.md](DEPLOY.zh-CN.md)。

## 镜像与资源

主路径为源码构建一个应用镜像，API 与 ReID 复用此镜像。前端在 Node 22 构建阶段编译；
Python/CUDA 来自 PyTorch 2.10.0 / CUDA 12.8 镜像。构建阶段执行 ReID、检索、
观测和视频回放四个前端 Node 回归脚本，再进行类型检查与 Vite 构建。
发行目录保存 `SOURCE_REVISION`、`SNAPSHOT_MANIFEST.json` 与 `IMAGE_ID`，其中快照清单记录
源码 revision 和各快照文件的 SHA-256；镜像内保存 `image-requirements.txt`。
还应记录使用的基础镜像 digest，以便审计与复现。

API 内的 YOLO 和 InsightFace 默认走 CPU；显式启用的 `reid` 和 `embedding` profile 使用 GPU 0。
ReID 并发和批次均为 1。容器内存限制不是显存配额，启动前须核对 GPU 空闲显存并监测加载与推理峰值。
剩余资源不足时保持 ReID 禁用，不停止其他模型来释放显存。

默认不启动 Qwen/VLM、通用视觉嵌入或摄像头。人体 ReID 与文字/通用图像检索是不同能力：
未配置视觉嵌入时不能把文字搜索结果宣称为 Qwen 向量检索结果。

## 配置与模型准备

复制 `deploy/containers/.env.example` 到宿主机专用部署目录的私密配置，权限设为 `0600`。
为 `APP_BASIC_AUTH_PASSWORD`、`POSTGRES_PASSWORD`、`MINIO_ROOT_PASSWORD` 和
`REID_SERVICE_API_KEY` 生成各自独立的随机值，不复用 SSH 密码。
数据库密码仅使用字母、数字、`_`、`.`、`-`，以免破坏数据库 URL。
配置文件使用简单的 `KEY=value`，可引用带引号的字面值；部署检查拒绝重复键、
shell 替换、命令片段和保留环境变量，不会执行配置内容。

首次非 `--check` 部署在配置缺失时会从模板生成文件及上述四个随机秘密，随后仍执行
模型与配置检查；缺少模型就退出，须准备并审查配置后再部署。
所有部署配置保存在宿主机，不能放进镜像。已有基础设施镜像可通过四个 `*_IMAGE`
字段指定已验证的本地镜像身份，避免重复下载。

创建 `MEDIA_DIR`、`MODEL_DIR`、`CACHE_DIR/api`，确保 UID/GID `1000:1000` 可写媒体和缓存，
可读模型。常规启动验收包含 CPU YOLO 和人脸模型 smoke，即使基础栈未启用人脸 ingest，
也需要预置 `yolo11n.pt` 与 Buffalo-L 模型。启用 ReID 再准备 checkpoint、pose 和 DFA，
启用 embedding 再准备完整 Qwen 权重和独立模型服务镜像：

```text
models/
  yolo11n.pt
  insightface/models/buffalo_l/det_10g.onnx
  insightface/models/buffalo_l/w600k_r50.onnx
  insightface/models/buffalo_l/其他官方模型文件
  yolov8n-pose.pt
  sapiensid_wb12m/model.yaml
  sapiensid_wb12m/model.pth
  dfa_mobilenetv4_medium/mobilenetv4_Final.pth
  qwen3-vl-embedding-2b-c73fa9ca/
    config.json
    tokenizer_config.json
    preprocessor_config.json
    tokenizer.json 或 tokenizer.model
    model.safetensors
    scripts/qwen3_vl_embedding.py
```

模型只读挂载。DFA 文件绑定到上游固定路径，`create_host_path: false` 防止缺少权重时
自动创建同名目录。资源须来自已核验的模型发布并核对哈希；运行时禁用模型下载。
参阅 [ReID 模型说明](../agx/reid_service/README.md) 与
[第三方声明](../../THIRD_PARTY_NOTICES.md)。Qwen 的固定版本和单独构建流程见下文。
当前固定 embedding 镜像要求 `model.safetensors` 单文件权重和
`scripts/qwen3_vl_embedding.py`，启动时对两者校验固定 SHA-256。
若额外存在分片清单，预检另查其路径和完整性；该清单不能替代上述两个必需文件。

可选功能按已有配置显式开启：`FACE_RECOGNITION_ON_INGEST` 控制人脸 ingest；
`PERSON_CROP_VISIT_MAX_SAMPLES`、`PERSON_CROP_VISIT_SAMPLE_INTERVAL_SECONDS` 和
`PERSON_CROP_VISIT_QUALITY_IMPROVEMENT_RATIO` 控制有限的 ROI visit 采样，默认分别为
`3`、`2.0`、`0.15`。这些采样参数不改变人员身份或 ReID 准入阈值。
`REID_CROSS_CAMERA_CALIBRATION_COEF` 和 `REID_CROSS_CAMERA_CALIBRATION_INTERCEPT`
必须同时是经当前相机与图库验证的有限数值；没有标定时两项均不配置，不能填空字符串。

使用已部署的远程 VLM 时，设置 `VLM_PROVIDER=openai_compatible`，提供容器可达的
`VLM_BASE_URL`、实际 `VLM_MODEL` 与私密上游 `VLM_API_KEY`；需要后台结构化属性时显式开启
`VLM_STRUCTURED_BACKGROUND`，同步 ingest 开关是 `VLM_STRUCTURED_ON_INGEST`。
Compose 只透传这些已有设置，不新增重型 VLM 容器。端点推理、授权与标签质量须另行验证。

## 发布与成功条件

完整部署默认要求干净的 Git 工作区。审查、提交代码后从该 checkout 运行；确需使用未提交
改动时显式传 `--allow-dirty`，发行记录与镜像源码 revision 标明 dirty，不能把它当成干净提交。
`--source` 可指定另一份已审查 checkout，`--release` 指定唯一发行名。
新构建自动把本次私密配置的 `SIGHTINDEX_IMAGE` 指向新镜像，无需另传 `--set-image`；
旧的 `--set-image` 仅保留参数兼容性。

`--no-build` 复用配置中已经存在于本机的 `SIGHTINDEX_IMAGE`，并照常启动和验收。
它不把宿主机的新代码装入旧镜像，也不保证新回放或人脸覆盖契约已具备；旧镜像不满足
新验收会失败。不要用复制源码、重启容器或移动 tag 代替镜像更新。

升级先把输入私密配置、旧发行路径和已有 stack 记录保存到 `backups/<本次发行>/`，
已有 PostgreSQL 实例通过旧发行的 Compose 入口执行 `pg_dump -Fc`。
备份目录权限为 `0700`、配置快照为 `0600`；数据库备份失败或为空时，部署会在构建和
服务变更前停止。`--skip-backup` 只用于已经另行验证外部数据库备份的情况。
首次空实例没有旧库可备份；已有数据库卷却找不到可备份的旧实例时，必须先解决备份问题。

随后脚本注册带 `.deployment-pending` 的候选发行，构建或复用镜像，校验 Compose，
启动所选服务，并等待 `postgres`、`etcd`、`minio`、`milvus`、`api` 及所选
`reid`、`embedding` 全部达到 `running healthy`。默认等待上限为 600 秒，必要时可用
`SIGHTINDEX_DEPLOY_TIMEOUT_SECONDS` 设置本次上限。所有服务健康后执行
`manage.sh verify <stacks> --model-smoke`：

- 先比较 Compose 当前 API 容器的实际 image ID 与 `SIGHTINDEX_IMAGE` 的本地 inspect ID，
  防止旧容器仍响应 `/health` 却被当作新发行。
- 带 Basic Auth 查询数据库媒体计数、检查 OpenAPI 视频回放路由及人脸覆盖响应契约，
  并以只读连接检查 `images.processed_at`、`source_video_url`、`video_offset_seconds`。
- 按所选 stack 检查 ReID/Milvus readiness、ReID 模型指纹及人脸优先就绪状态，
  或 Qwen 服务的模型、维度和语义配置及覆盖计数。
- 在临时目录与独立 SQLite 中处理生成的视频，验证上传、裁剪、来源回放、Range 和媒体访问；
  不向实际 HTTP 服务上传，不写真正部署的数据库或媒体目录。
- 用生成的像素加载本地 CPU YOLO 和 InsightFace 进行 smoke；拒绝下载模型。
  该结果只验证模型能执行，不能证明真实人员检测、识别或跨摄像头准确率。

完整验收成功后才写入 `active-release`（一行绝对发行路径）与 `active-stacks`，并移除
候选的 `.deployment-pending`。失败不更新成功标记；已经启动的服务可能仍运行候选版本，
脚本不会自动恢复数据库或把容器切回旧版。须检查日志并按运维手册的显式回退流程处理。

成功只覆盖本次选择的 stack。VLM 默认关闭；启用已有远程 OpenAI 兼容端点时，验收仅核对
`VLM_PROVIDER`、`VLM_BASE_URL`、`VLM_MODEL` 等配置，不发送模型推理请求、不验证标签质量。
ReID/语义检索报告的 pending 与覆盖计数不表示历史索引已补齐，脚本也不运行历史回填。
真实检索效果、完整历史覆盖和摄像头采集必须单独验收。

## 可选：自定义镜像构建基座

默认部署已经自动构建应用镜像。只有需要审查过的替代构建基座或镜像源时，才单独构建
并将私密配置的 `SIGHTINDEX_IMAGE` 固定为该镜像，再用 `bash deploy.sh --no-build` 验收。
使用项目根目录作为手工构建上下文：

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

## 统一操作入口 `manage.sh`

`deploy/containers/manage.sh` 是运维唯一入口，封装 env 文件、release 目录、
overlay compose 文件与 profile 的组合，操作人员不再手工拼
`--env-file`/`-f`/`--profile` 参数：

```bash
manage.sh [--env-file FILE] [--release DIR] COMMAND [STACK ...]
```

必须显式给命令：`up` 应用选中的镜像与配置，必要时重建容器；`restart` 仅重启已有容器，
不应用镜像或配置变化；`down` 停止整套项目并保留卷；`status` 显示 Compose 状态及 API
存活端点；`logs api` 或 `logs --service api` 跟踪指定服务日志，服务名仅接受
`postgres/etcd/minio/milvus/api/reid/embedding`。
`config` 执行 `compose config --quiet`，不打印展开的秘密；`verify` 检查实际 API 镜像 ID
与所选能力，默认带隔离上传 smoke，显式 `--model-smoke` 再检查本地 CPU 模型。
`status` 的 `/health` 存活结果不能代替 `verify`。可组合的 stack：

| stack | 内容 | 前置条件 |
| --- | --- | --- |
| `base`（默认） | postgres etcd minio milvus api | 已预置 CPU YOLO/人脸模型与私密配置 |
| `reid` | + reid 服务 | env 中 `REID_ENABLED=true`，且 GPU 容量已核对 |
| `embedding` | + Qwen3-VL 嵌入服务与 API 接线 | env 中三个 `QWEN_*` 键 |
| `semantic` | + 语义检索设置 overlay | 同时选择 embedding、显式 `SEMANTIC_SEARCH_ENABLED=true`；历史回填另行验收 |

部署根目录默认 `/data/sightindex-bj-test`；`deploy.sh` 可用 `--root` 覆盖，
`manage.sh` 使用非默认根目录时须设置 `SIGHTINDEX_ROOT`。
根目录保存实际私密配置（默认 `.env`）与 `releases/<版本>/deploy/containers/`（含 base compose）；
`media/`、`models/`、`cache/` 与数据卷跨版本共享。`--release` 优先，未指定时先读取
`active-release`；该标记必须指向 `ROOT/releases` 内的真实目录且不是符号链接文件。
旧安装没有标记时才选最新带 base compose 且没有 `.deployment-pending` 的 release。
overlay 先查目标发行，再按新旧查其他非 pending 发行，以兼容历史拆分发布。
显式指定的 pending 发行可用于本次 `up/config/verify`，不会自动成为默认发行。
`active-stacks` 只是记录，未指定 stack 时仍默认 `base`，不会隐式启动 GPU 服务。
release 名不得含空格。

脚本对比运行项目的 base compose 与本次目标，不一致时告警。应用新配置应执行 `up`，
`restart` 只重启原有容器。必须显式传入实际私密 env 与所选 stack，例如完整栈：

```bash
cd /data/sightindex-bj-test
bash manage.sh --env-file .env.semantic-search-v1 up base reid embedding semantic
bash manage.sh --env-file .env.semantic-search-v1 verify base reid embedding semantic --model-smoke
```

## 访问与另行验收

启用可选 stack 时，私密配置的 `REID_ENABLED` 必须与是否选择 `reid` 一致，
`SEMANTIC_SEARCH_ENABLED` 必须与是否选择 `semantic` 一致；`semantic` 还要求选择
`embedding`。准备完模型与 GPU 后，统一入口可以部署完整组合：

```bash
bash deploy.sh --check --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env.semantic-search-v1 \
  --stacks "base reid embedding semantic"
bash deploy.sh --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env.semantic-search-v1 \
  --stacks "base reid embedding semantic"
```

在启动验收之外，还应核对匿名控制台、媒体与业务 API 的访问控制，带认证访问控制台，
以及已有数据的计数基线。真实人员识别、跨相机检索、历史回填覆盖、长时间摄像头录制和
其他业务共同运行时的 GPU 峰值需要专门样本及操作员验收，脚本不会自动补齐这些结果。

默认仅公开宿主机 `127.0.0.1:18030`。PostgreSQL、Milvus、MinIO 和 ReID 没有宿主机端口。
远程访问使用 SSH 本地转发；公网访问需额外配置 HTTPS 网关及访问控制，不直接修改现有 FRP。
嵌入端点配置独立视觉嵌入 API key 时验证该 key；未配置时服从已启用的 Basic Auth。
不要将 SSH 口令或上游模型提供商密钥作为公开客户端凭证。

## 数据与回退

所有 Compose 资源属于 `sightindex-bj-test`，持久卷与现有业务分离。新增实例停止时执行
同一 Compose 入口的 `down`，不得使用 `down -v`、全局 prune 或删除共享目录。

升级的自动备份在 `backups/<本次发行>/`，应连同旧镜像身份、发行目录与模型资源保留。
失败候选可能仍在运行；恢复时显式选择旧发行和已核对的备份配置，执行 `up`、`verify`，
成功后再恢复 active markers，完整命令见 [人工回退](DEPLOY.zh-CN.md#人工回退)。
新增兼容列保留，不能自动整库恢复覆盖新采集数据，也不删除媒体、历史索引或共享模型。
选择 `base` 不会自动停止已经运行的 ReID/embedding；需要释放资源时，应明确停止本实例的
对应可选服务，再禁用配置，不能靠 `up base` 假定 GPU 容器已经停止。

## 身份与计数兼容性

应用保留 `person_crops.person_id_source`、`images.processed_at` 及视频来源和时间偏移的
兼容可空列，不删除或猜测旧数据来源。旧版本可能不理解这些字段与新回放契约，
回退目标也必须核对兼容性。
人工标注与人工撤销记录为 `manual`，自动人脸归属记录为 `face`。自动识别不得覆盖人工决定，
也不得覆盖来源不明的已有非空身份；历史模型判断仍单独保存在识别事件中。
旧身份需要重新分配时，先人工撤销再重新标注。回退到不理解此字段的旧程序会失去这层保护。

越线 track ID 仅用于一次视频/流会话的位置跟踪，不是跨摄像头身份。轨迹默认连续失配
2 个采样帧或 6 秒未出现后失效，可用 `LINE_CROSSING_TRACK_MAX_MISSED_FRAMES` 和
`LINE_CROSSING_TRACK_IDLE_SECONDS` 调整。空帧参与失配统计，短暂单帧漏检保留轨迹。

## 应用功能的另行验收

以下检查用于应用功能验收，不表示部署脚本已经对真实数据执行了这些操作：

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

推荐通过[模型部署说明](../../docs/model-deployment.zh-CN.md)的已审查锁文件统一导入或
显式下载。下述独立 Qwen 下载器仍是可选的手工工具；部署入口不会自动调用它，使用前须
另审查其来源、固定版本及逐文件清单：

```bash
python3 deploy/containers/download_embedding_model.py \
  /data/sightindex-bj-test/models/qwen3-vl-embedding-2b-c73fa9ca
```

下载器保存官方逐文件版本及 SHA-256 清单，支持续传，逐文件校验后才改为正式文件名。
权重固定为 `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`，
推理脚本也校验哈希；运行时禁止联网下载模型。需要更换版本时先复核下载器和启动校验。

部署默认复用配置中的 embedding 镜像。选 `embedding` 时可显式加
`--build-model-services`，从本次应用镜像构建到 `QWEN_EMBEDDING_IMAGE`；也可先基于
已核验的 SightIndex 镜像手工构建，再通过该变量提供：

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

准备完成后，在私密配置中保持 `REID_ENABLED=false`、`SEMANTIC_SEARCH_ENABLED=false`，
可以从源码仓库根目录用统一入口部署 `base embedding`；若还使用 ReID，配置与 stack
必须一起包含它：

```bash
bash deploy.sh --check --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env.qwen3vl2b-v1 --stacks "base embedding"
bash deploy.sh --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env.qwen3vl2b-v1 --stacks "base embedding"
```

### 历史兼容补丁与分步割接（2026-09）

以下保留北京旧 v4 的兼容方案，仅适用于已经核对的同一旧镜像基线。
它不是当前新发行的默认流程，本轮未连接北京服务器执行或复测这些步骤。
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

当前源码的常规更新使用 `bash deploy.sh --stacks "base reid embedding semantic"`，
并按本节单独验收历史覆盖与检索效果。选择 semantic overlay 不会自动打开开关，
模板仍为 `SEMANTIC_SEARCH_ENABLED=false`；必须显式设为 `true` 才能通过所选能力的检查。

2026-09 北京历史方案曾对已核验的旧 API 镜像使用 `Dockerfile.semantic-search` 做最小更新；
本轮未实测该服务器，该方案也不用于当前默认部署。其专用构建目录
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

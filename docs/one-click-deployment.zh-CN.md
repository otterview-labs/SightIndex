# 一键部署与验收

新安装的统一入口是仓库根目录的 `deploy.sh`。2026-10-07 已对现有 5090 实例完成保留
`/home/SightIndex` 的独立原地发布与上线验收，见下方「5090 原地发布结果」。这次没有
执行 `/opt/sightindex` 新安装器，也没有实机演练全部容器部署组合；正式采用新安装流程
仍需审核提交并在目标环境验收。软件上线通过不等于真实人员准确率通过。

“一键”指配置和审查完成后由一个命令完成所选能力的部署与启动验收。模型可以通过
显式选项纳入准备，但不包含驱动安装、隐式下载、VLM 服务安装、历史标签/向量回填
或摄像头录像系统建设。

## 选择部署方式

| 方式 | 命令 | 约束 |
| --- | --- | --- |
| 容器，默认 | `bash deploy.sh --stacks "base reid"` | Docker Engine + Compose v2、宿主机 Python 3.11+；GPU stack 还需 NVIDIA 驱动及 Container Toolkit |
| RTX 5090 systemd | `sudo bash deploy.sh --target rtx5090` | Linux x86_64、真实 5090、可信干净源码位于 `/opt/sightindex`、Python 3.11+、Node 22.18+；保留现有账号/权限规则 |

容器方案不会调用 systemd 安装脚本，也不修改宿主机 Python/Node、驱动或其他业务容器。
默认 Compose 项目名和数据根仍为 `sightindex-bj-test` 与 `/data/sightindex-bj-test`。
仅修改 `--root` **不等于**创建独立的第二实例；不要同时用不同根目录管理同名 Compose 项目。
新实例须显式提供 `--project-name sightindex-evaluation`，并在其私密 `.env` 配置相同的
`COMPOSE_PROJECT_NAME`、独立端口、`MEDIA_DIR` 和 `CACHE_DIR`。验收后的根目录保存
`project-name` 标记；后续部署/管理拒绝改变绑定。复制到根目录的 `manage.sh` 默认管理
自身根目录，也可显式传 `--root`；源码中的脚本仍保留旧默认根目录。所有 Compose 操作及
数据库备份都显式指定项目。共享模型仅应只读挂载，不能复用业务数据库或媒体作隔离验收。
Compose 仍遵循 shell 环境插值；执行前清除与私密配置冲突的 `MEDIA_DIR`、`CACHE_DIR`、
`SIGHTINDEX_IMAGE` 等同名环境变量，不能仅凭 `.env` 推断最终挂载配置。
模型的授权、使用限制见仓库 `THIRD_PARTY_NOTICES.md`，尤其是 SapiensID 的非商业限制。

## 先准备配置和模型

容器私密配置采用 `deploy/containers/.env.example`，放在部署根 `.env`，权限 `0600`。
使用简单 `KEY=value`，不要放 `export`、shell 替换、重复键或进程保留变量。
未配置时第一次正式执行会创建私密模板及独立随机密码；缺模型会在构建和启动之前退出。
`--check` 不会生成这个文件，因此应先准备好配置再检查。

模型准备已接入同一入口。提供完整、已审查的模型锁文件后，可先独立导入或核验：

```bash
bash deploy.sh --models-only --env-file /data/sightindex-bj-test/.env \
  --stacks "base reid" --model-manifest /bundle/models.lock.json \
  --model-source /bundle/assets --acknowledge-model-terms

bash deploy.sh --models-only --env-file /data/sightindex-bj-test/.env \
  --stacks "base reid" --model-manifest /bundle/models.lock.json --check
```

正式部署加 `--prepare-models --model-manifest FILE --model-source DIR
--acknowledge-model-terms` 可合并离线准备；以 `--download-models` 代替离线来源则明确允许
下载锁文件中缺失的资产。已有文件不符不会覆盖，准备失败会阻止构建和启动。
完整角色、来源约束、授权及 RTX 路径差异见[模型部署说明](model-deployment.zh-CN.md)。

容器默认 `base` 包含 API、前端、PostgreSQL、Milvus，以及 API 内的 CPU YOLO/InsightFace。
默认不启动摄像头、GPU ReID、Qwen 或 VLM。所需模型布局：

```text
models/
  yolo11n.pt
  insightface/models/buffalo_l/det_10g.onnx
  insightface/models/buffalo_l/w600k_r50.onnx
  # 选 reid 时还需：
  sapiensid_wb12m/model.yaml
  sapiensid_wb12m/model.pth
  yolov8n-pose.pt
  dfa_mobilenetv4_medium/mobilenetv4_Final.pth
```

完整 buffalo_l 的其他 ONNX 文件也应保留。ReID 指纹是模型、配置、pose 和 DFA 组成的
完整推理管线指纹，不是任意填一个 checkpoint 哈希。最终验收会对照服务实际指纹。
RTX 方案仍使用其模板中的 `/var/lib/sightindex` 路径，以及源码 vendor 内的 DFA 权重路径。

选 `embedding` 需配置 `QWEN_EMBEDDING_IMAGE`；可提前构建，或显式加
`--build-model-services` 从本次应用镜像构建。还需准备完整本地 Qwen 模型目录和
独立 API key。当前固定 embedding 镜像需要 `model.safetensors` 单文件权重及
`scripts/qwen3_vl_embedding.py` 推理脚本，启动时校验它们的已固定哈希；只有通用分片权重
不能替代这个契约。预检也检查 config、processor、tokenizer 数据和存在的分片清单；
目录外的权重 symlink 不会通过，因为容器只挂载模型目录。`semantic` 还要求同时选 `embedding`、
显式设置 `SEMANTIC_SEARCH_ENABLED=true`；历史向量覆盖仍需单独完成并验收。

VLM 只支持配置现有 `openai_compatible` 远程服务。本入口不新建 VLM 容器、不调用付费生成
接口来测标签质量。`VLM_PROVIDER=none` 会明确显示未启用，不能把标签提取算作已验收。

## 只读预检与正式执行

从已审查的 Git checkout 执行，以下配置与路径均是示例，不包含真实凭据：

```bash
bash deploy.sh --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env --stacks "base reid" --check

bash deploy.sh --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env --stacks "base reid"

# 在 /opt/sightindex 的 RTX 主机执行：
bash deploy.sh --target rtx5090 --check
sudo bash deploy.sh --target rtx5090
```

`--check` 只读取配置、资产和工具/GPU 状态，不创建目录、不构建镜像、不启动服务、不写库。
它不是实际推理测试，也不替代 RTX 安装器严格的账号、目录所有权及可信源码检查。
可通过 `SIGHTINDEX_DEPLOY_PYTHON=/path/to/python3.11` 指定宿主机部署解释器。

容器正式部署默认拒绝未提交源码。确实需要评估工作区时可显式 `--allow-dirty`，发行记录
会带 `-dirty`，不能宣称等同于某个干净 Git 提交。`--no-build` 只复用已有本地镜像，
不会把新源码装入旧镜像；不要用它完成需要应用代码更新的发版。

正常流程：可选的显式模型准备/锁文件核验 → 只读预检 → 私密配置/数据库备份 → 最小代码快照 → 镜像构建 → 自动选择新镜像
→ Compose 配置检查 → 启动所选服务 → 全部 healthy → 应用级验收 → 标记 active release。
模型、用户媒体、数据库、私密 `.env` 与无关评估报告不打包进代码快照。

## 验收代表什么

容器验收入口也可以手动运行：

```bash
bash /data/sightindex-bj-test/manage.sh config base reid
bash /data/sightindex-bj-test/manage.sh verify base reid --model-smoke
bash /data/sightindex-bj-test/manage.sh logs api
```

验收检查所选项目内 API 及所选模型服务的实际镜像 ID，而不只看 `/health`；然后带认证核对业务 API、
OpenAPI 回放合同、数据库新增字段，检查已选择的 ReID/人脸/Milvus 和 embedding readiness。
使用临时数据库、临时目录及合成视频实际执行上传、抽帧、裁剪、回放定位和 HTTP Range 206；
不会给业务数据库加入测试视频。模型 smoke 仅用合成像素加载已有 YOLO/InsightFace 模型，
不下载权重、不读取真实人员照片，也不证明识别准确率。

以下事项不能仅凭部署成功宣称完成：

- 历史标签和向量覆盖达到 100%；pending 数量会报告，需要另行回填。
- ReID 的真实召回率、误匹配率和阈值已经适合现场；需要多人跨摄像头标注走测。
- VLM 可达、标签准确、所有模型组合显存峰值可接受；需要目标主机单独测试。
- 摄像头历史录像、NVR 回放已经接入；本轮回放只支持有原始文件与偏移的新上传视频。
- 旧视频已补齐源文件关联；缺失关联会明确返回不可回放，而非猜测位置。

## 2026-10-07 收尾修复与旧实例更新边界

本轮补充了以下保护，不能将其等同于已经完成线上切换：

- 上传视频定位只采信递增的解码媒体时钟，维护全部已解码帧的时间高水位。
  时间戳缺失、停滞或倒退时不再用平均 FPS 推测命中位置，接口明确返回定位未知；
  恢复后超过此前高水位才重新允许定位。合法 VFR 时间戳保持原样，旧媒体不自动回填。
- 容器验收核对 API 和所选 ReID、Qwen 服务的实际 image ID。Qwen 还需报告并匹配
  固定权重、推理脚本 SHA-256 和预处理参数；旧服务缺字段应明确验收失败。
- ReID 验收按配置计算并核对完整 namespace 和 SQL 索引指纹，运行时与验收复用
  无 I/O 的身份计算。既有字节公式、模型、阈值、集合名和历史标记不改变。
- 模型中断恢复采用受控操作，活跃任务不能被绕过；旧锁和损坏 partial 不静默删除。
  具体诊断、恢复确认与隔离方法见[模型部署说明](model-deployment.zh-CN.md)。
- 代码快照、Docker 构建上下文和 RTX 代码同步均排除模型恢复隔离目录；同步不删除
  目标机器原有隔离证据，也不会传输开发机的模型锁和 partial 缓存。

本次只读实机检查发现旧 5090 实例由 `/home/SightIndex` 的 systemd 服务运行，使用本地
SQLite、已加载的 SapiensID/InsightFace 和已有 Milvus 索引；没有新部署入口和回放字段。
该布局不满足新的 `/opt/sightindex` 安装器约束，不能把新版安装器直接当作旧目录更新器。

更新前应明确选择保留旧目录还是规范化迁移，保留私密配置和运行环境，先在服务器本机
取得一致的 SQLite 备份、代码及服务配置备份，再在隔离副本上验证兼容迁移、接口合同和
模型索引身份。不要把业务数据库或模型复制到开发机，不自动升级驱动/Torch、替换模型、
启用 Qwen/VLM、回填历史或修改已有 Milvus 集合。代码回退也不能自动恢复整库丢弃新数据。

兼容初始化不仅补上传视频三列，还包含既有重复标记/任务清理和观察时间修正逻辑；
应先在服务器本地的一致性副本上执行，比较行数、索引标记和受保护字段，再决定上线。
不能仅因新增字段可空，就宣称整个初始化对旧业务记录完全没有改动。

RTX 新安装的依赖清单已显式包含 `httpx`，用于必选的隔离上传回放验收，不依赖开发
extras 或偶然的间接依赖。旧实例原地更新仍应先核对现有 venv，不用为此升级整个 GPU
环境。现成的隔离上传 smoke 使用新建测试库，不能替代旧业务库副本的迁移验收。

### `/home/SightIndex` 旧实例的独立发布工具

旧目录更新不调用新的 `/opt` 安装器。`deploy/rtx5090/legacy_bundle.py` 从明确允许的
应用源码、前端生产构建和部署资料生成私密代码包，记录实际文件 SHA-256 清单和
`<commit>-dirty-<release>` 身份；本地工作区快照不能宣称是干净 Git 提交。包不包含
业务数据库、媒体、私密配置、venv、模型、ReID vendor 或用户效果评估报告。普通代码
目录 `app/models` 是 ORM 源码，必须纳入；它不是推理权重目录。

包的 `--verify` 只读取归档，不解压或上线。服务器应先用开发机记录的清单 SHA 验证
包，再解压到新建、权限 `0700` 的 staging 目录；不得直接解压进运行中的应用目录。
代码切换仅覆盖清单中的普通文件，不删除目标机器其它文件，不覆盖 systemd unit。
新静态资源先到位、`index.html` 最后切换，旧 hashed assets 保留以兼容已经打开的页面。

`deploy/rtx5090/legacy_acceptance.py --work-dir <私密验证目录>` 只接受该目录里的
`migration.sqlite`，其副本必须由服务器本地 SQLite backup API 创建。目录为当前用户
所有的 `0700`、数据库为单链接普通文件 `0600`，不接受线上数据库路径、符号链接、
正在使用的副本或已生成报告的目录。使用原 venv，从 staging 代码运行。工具在导入
应用模块前隔离配置，禁止摄像头、队列、网络、模型及 VLM，不加载生产 lifespan。

验收先拒绝重复向量/任务，再在副本运行两次兼容初始化，比较所有既有表的行数与
内容摘要；仅接受精确的观察时间修正、旧任务 lease 归一化及标准容量锁补齐。人工绑定、
人脸/人体标记、摄像头状态等均受保护。之后在同一副本上传本地生成的 5 帧视频，验证
裁剪父帧、偏移、10 个回放请求、Range 206 和负向关联检查；报告仅保存在服务器私密目录，
终端不打印业务数据。副本验收通过后，仍须在短维护窗口取得最终备份并再次核验。

切换后用原启动方式恢复 API，ReID 服务、模型和 Milvus 不重启或重建。实际 API、
OpenAPI、数据库字段、模型指纹与静态资源均应重新验收。失败时恢复旧代码和前端，
不要自动恢复整库丢弃上线后采集的数据。已启用的标签后台配置保持原状，本次操作
不额外发起历史回填，也不把“独立回填服务未运行”误认为 API 标签后台被关闭。

## 失败、备份与回退

容器配置与数据库备份放在 `backups/<release>/`，目录权限 `0700`。
升级时现有 PostgreSQL 容器存在则执行 `pg_dump -Fc`，失败会在构建/切换镜像前中止。
已有数据库卷却找不到可备份容器也会中止；`--skip-backup` 只能在已验证外部备份后使用。
媒体和 Milvus/MinIO/etcd 的一致性备份仍需运维另做，不能把 `pg_dump` 说成完整系统备份。

新 release 在验收前有 `.deployment-pending` 标记；只有成功才更新 `active-release` 和
`active-stacks`。失败 candidate 不会成为默认运维版本，但如果已启动，服务仍可能运行
candidate 镜像。脚本不会自动把数据库恢复成旧备份、丢弃新的上传/采集记录，也不承诺
全自动服务回滚。请使用[容器运维手册](../deploy/containers/DEPLOY.zh-CN.md)的显式回退流程。

RTX 方案沿用原有失败恢复：恢复旧 venv/前端后服务可能仍保持停止，需要检查部署日志和
systemd journal 后人工恢复。不要在失败状态下仅重试页面访问就宣布部署完成。

## 本轮本地验证记录（2026-10-06）

- 全量 Python 回归：1557 项通过、0 失败，8 个现有依赖或本机无 CUDA 相关警告。
- 前端 ReID、搜索、观察表、视频回放回归：64 项通过；类型检查和生产构建通过。
- 前端调用与当前 OpenAPI：47 项检查，0 项不一致。
- 模型资产和部署映射包含 221 项回归，覆盖离线导入/复用、大小/摘要、HTTP mock 续传、
  指纹、完整人脸包、权限与错误脱敏；使用合成小文件，未下载或执行真实大模型。
- 部署编排包含 55 项执行真实 Bash 的临时环境回归；Docker、Git、GPU 和 ownership
  的外部边界使用严格 mock，没有启动真实容器或更改主机权限。
- 预检、验收及相关新增测试通过 Ruff；部署 Shell 语法检查通过。

验证包含隔离 SQLite 与合成视频的真实上传、抽帧、裁剪和 Range 回放，但未使用业务人员
画面，也未在 RTX 5090 上加载实际模型。上述数字是本地软件回归结果，不是线上可用率、
人脸识别率或跨摄像头 ReID 准确率。本轮没有部署服务器或推送 GitHub。

## 原地发布前的本地验证记录（2026-10-07）

- 全量 Python 回归：1689 项通过、0 失败，8 个现有依赖或本机无 CUDA 相关警告。
- 全量之后补充的隔离目录/RTX 依赖保护，以及代码快照和不同应用镜像下的 Qwen
  构建验收，收尾定向复测 5 项通过。
- 前端 ReID、搜索、观察表、视频回放：64 项通过；类型检查及生产构建通过。
- 使用隔离配置从当前源码导出 OpenAPI，47 个前端调用检查，0 项不一致。
- 模型资产及部署映射专项：262 项通过，覆盖强制终止后的锁释放、活跃锁拒绝、
  旧锁受控迁移、恢复并发/中断、私密隔离目录和最终权重保护；均为合成小文件。
- 本轮修改的 Python 与测试文件 Ruff 检查、7 个部署 Shell 的语法检查及 diff 空白
  检查通过。未安装或宣称通过 mypy、Black 或覆盖率检查。

只读检查确认 5090 的 API、ReID/Milvus 及人脸优先状态在线，已有 14531 条人体索引、
pending 为 0；当前加载模型为 `sapiensid_wb12m`、4096 维、`squarepad-v1`。按原公式
重新计算的 namespace 与完整索引指纹均与线上状态一致。这是当次状态快照，不是
真实检索准确率或永久覆盖承诺。

截至上述本地验证阶段，还没有执行服务器备份、数据库克隆、迁移、代码切换、服务
重启或 GitHub 推送，旧服务也没有新上传视频回放接口。后续经确认采用保留旧目录的
原地更新，实际发布结果如下；不要将发布前的状态作为当前线上状态。

## 5090 初次原地发布结果（历史验收记录）

2026-10-07（Asia/Shanghai）已更新现有 5090 实例，局域网入口为
`http://<5090-LAN-IP>:18030/reid`。应用继续位于 `/home/SightIndex`，沿用
原有 systemd、venv、私密配置、模型与 Milvus。只有 API 在短维护窗口停止并重新启动，
ReID 保持原进程，人体和人脸权重没有更新。Qwen embedding 服务保持未启用，已配置的
API 标签后台保持原状；没有额外发起历史回填或索引重建。

发布身份为 `36a4adc-dirty-20261007-inplace-final`，是明确标记为 dirty 的工作区代码
快照，不是新的 Git 提交。初次运行代码上线后仅修正文档，未再次重启 API；最终
`SOURCE_MANIFEST.json` 记录 211 个发布文件的实际 SHA-256。初次发布阶段尚未推送 GitHub；
该段与下面状态表是当时的验收快照，不代表后续修复仍停留在相同版本。

### 备份与旧库验收

服务器私密备份目录采用 `/home/SightIndex/backups/update-YYYYMMDD.XXXXXX`，权限 `0700`。
其中保存原配置、服务配置、旧代码以及停写后的最终一致 SQLite 备份；业务数据库和
媒体没有下载到开发机。初次副本与停写后的最终副本都在服务器本地通过兼容迁移和
合成视频验收，生产数据库没有被测试副本或备份覆盖。

最终副本的 27 个既有表行数与原有字段内容摘要保持不变，两次初始化幂等；观察时间
修正、旧任务 lease 归一化和容量锁补齐在本次数据上均没有产生内容变更。新增四个
nullable 列：`images.processed_at`、`images.source_video_url`、
`images.video_offset_seconds`、`person_crops.person_id_source`。不能据此推断其它
旧数据库迁移也一定没有内容变更。

5 帧合成视频在隔离副本中产生 5 个裁剪，偏移为 `0.0–0.4s`。10 次回放定位检查、
裁剪父帧关联、Range `206` 和 5 项负向关联检查通过；未加载生产 lifespan、未启动
worker、未访问业务媒体或调用模型。真实线上接口合同另行验收通过。

### 上线状态与验证范围

| 检查项 | 上线后的结果 |
| --- | --- |
| API 与 ReID | active/running，异常自动重启次数均为 0 |
| 代码与页面 | 211 个文件摘要一致；局域网页面 HTTP 200，入口及 3 个引用资源与发布文件一致 |
| 数据库 | 只读 quick_check 为 ok，新增字段已存在 |
| 回放与人脸覆盖接口 | 2 个回放接口及 face_coverage 响应合同已存在 |
| 人体索引 | 当次检查 14532 条，pending 为 0，Milvus 在线 |
| 人脸优先 | InsightFace buffalo_l，CUDA，状态就绪 |
| 运行环境保护 | 私密 .env、11 个模型资产及 3 个 systemd 配置文件未变 |

上线前最终全量 Python 回归为 1797 项通过、0 失败，保留 8 个现有警告；前端 ReID、
搜索、观察表和视频回放共 64 项通过，类型检查及生产构建通过。部署后使用真实线上
OpenAPI 检查 47 个前端调用，0 项不一致。文档修正后的发布包与部署资产定向回归
131 项通过。线上验收的 6 项检查全部通过，包括数据库计数、OpenAPI、schema、
配置状态、ReID 与隔离上传回放。
VLM 检查只确认配置存在，没有验证远端生成或标签质量；没有执行 mypy、Black 或
覆盖率检查。以上是软件与部署结果，不是检索准确率或永久索引覆盖承诺。

库中仍有 2 个失败的 `person_attributes` 标签任务，此次没有自动重试或清除。人体模型
仍为 `sapiensid_wb12m`、4096 维、`squarepad-v1`，完整索引指纹和旧 Milvus 空间保留。
新版不再默认展示未现场校准的跨摄像头匹配概率，也不依据单纯人体相似度自动写入
确认身份；这些变更没有修改已有人工绑定。

新上传视频在保留源文件、取得可信解码偏移且浏览器支持编码时可定位回放。旧记录缺少
来源关联不能补猜，摄像头仍不保存连续录像、没有 NVR 适配，也不自动转码。真实跨门
召回率、误匹配率、人脸有效覆盖及标签阈值仍需多人走测。失败标签任务需单独诊断，
初次发布时，模型替换、现场校准、GitHub 发布和新安装器的完整实机演练尚未完成。

## 新上传视频的浏览器兼容准备

`VIDEO_PREPARATION_ENABLED` 默认 false，升级不会自动扫描或转码历史视频。显式开启后，
只处理新上传的本地文件：可用的 MP4 复用原件，不兼容编码在 CPU 上转成 H.264/yuv420p、
faststart MP4。原文件保留；抽帧使用实际播放文件，从同一时间轴取得偏移，不能把原 AVI
的偏移套到转码后 MP4。`VideoAsset` 保存原件、播放件、状态及声音信息，clips 接口分页
返回真实帧；未知视频返回 404，GET 不触发转码。

需要 ffmpeg、ffprobe。默认限制为 2 CPU 线程、120 秒处理时间、5 分钟输入时长、1080p
像素上限、128 MiB 输出上限，至少保留 512 MiB 空闲空间。超限或失败保留原件并返回有限
原因码，不发布半成品。跨进程锁防止并发转码挤占资源；不调用 GPU、不修改模型或索引。
`VIDEO_PREPARATION_INCLUDE_AUDIO=false` 时规范化输出不含声音；可显式设 true 转 AAC，
原件中的音频始终保留。实际播放接口的 `has_audio` 字段应作为依据。

容器镜像包含 ffmpeg，最终 runtime 验收检查工具；RTX 主机预检只在开启该能力时检查
工具是否存在，脚本不自动为旧实例升级系统软件。此能力不是摄像头连续录像/NVR 接入，
也不承诺缺失来源关联的旧抓拍可回放。

## 2026-10-07 补充诊断与隔离验收

两条历史标签失败记录均为上游连接重置，自动预算已耗尽（尝试 4 次，预算 3）。新增的
定向重试端点只为一条失败任务授予一次额外尝试，保留累计次数和原失败证据；普通采集、
补漏和队列对账不自动复活终态失败。新错误存为稳定原因码，旧任意错误正文对外脱敏。
配置显示为 worker 已配置，不把配置存在当成模型已可用。

在服务器私密目录导出 32 对人工反馈（10 正、22 负），用实际生产参数显式传给纯离线
评估器。汇总仅诊断标注时的证据快照，不访问模型、数据库或业务图片，不代表当前完整
融合排序、检索 Rank-1/mAP 或独立人员测试的准确率：

| 证据规则 | 覆盖与诊断 |
| --- | --- |
| 人体分数 ≥0.50 | 覆盖 32/32；TP=9、TN=3、FP=19、FN=1；这批候选精确率约 32.1% |
| 人脸相似度 ≥0.45、可靠性 ≥0.70 | 有效覆盖 0/32，不能评价人脸有效性，不把缺脸当负例 |
| 累计稳定标签冲突 ≥2 才排除 | 可比较 30/32；正例 10 通过、负例 20 也通过，不能单靠标签确认身份 |

这些反馈可能集中于难例，同一人员也可能重复出现，没有独立留出集，因此不能外推全量
生产准确率。训练阈值建议明确返回样本不足、threshold=null；本轮不据此自动修改生产阈值。
至少补齐有效分数的 30 正/60 负样本，并做真正按人员隔离的留出走测，再决定阈值或模型。
完整方法见[跨摄像头走测说明](reid-walkthrough-calibration.md)。

使用 staging 代码在服务器隔离 SQLite/合成媒体上完成真实 FFmpeg CPU AVI→H.264 MP4
上传：原件摘要保持不变、5 帧/5 裁剪、10 次回放关联、偏移 0.0–0.4 秒、真实 clips 分页及
Range 206 通过，没有 GPU/模型或业务媒体调用。现有本机 Chromium 对这份 2 秒、64×48
合成 MP4 验证元数据、精确 seek 0.3 秒、实际解码呈帧、播放推进及两个 Range 请求均通过。
浏览器验收只有一个原生视频用例，不是完整 Vue 页面 E2E，也不保证所有现场格式都兼容。

### 本轮旧实例更新已验收

2026-10-07 已完成第二次原地发布，代码身份为
`36a4adc-dirty-20261007-fixes-final`，219 个文件摘要一致；这是工作区发布清单身份，
不能当成某个干净 Git 提交。API 已按原 unit 恢复，ReID 原进程不变，11 个受保护模型资产
和 3 个 systemd 文件保持原状。私密配置仅新增或修改 `VIDEO_PREPARATION_ENABLED=true`，
其它秘密及模型参数保留；没有恢复整库、重建 Milvus、启用 Qwen 或连续录像。

停写后的最终 SQLite 副本通过兼容初始化及隔离合成视频验收，随后真实线上验收通过。
ReID 页面及 3 个入口资源与发布文件一致，接口 HTTP 200；当次人体索引为 14536、pending=0，
Milvus 在线，人脸 `insightface-buffalo_l` 在 `cuda:0` 就绪。两条原失败任务分别只发起一次
定向重试，均完成 VLM 属性写入，2 完成、0 失败、0 等待、0 运行；原失败证据保存在私密备份。
这些数字是当次快照，采集继续运行后计数会变化。

应用代码冻结后的全量 Python 回归为 2052 通过、1 跳过、8 个现有警告。跳过项是本机缺少
FFmpeg，服务器隔离真实转码已验证。随后增加受审查运维工具的精确发布白名单，打包/部署
资产专项 133 项通过；新版离线评估器、两份图片清理工具及走测资料也已纳入旧实例发布，
不复制任意脚本或业务 CSV。前端 64 项回归、类型检查、生产构建、47 个 API 合同检查通过；
本轮修改通过 Ruff 和部署 Shell 语法检查，未运行 mypy、Black 或覆盖率工具。

Git 发布使用 `codex/otter-merge-20261006` 功能分支，不强推、不直接改 main；运行代码由
独立 SHA 清单验收。私密配置、业务库、媒体、模型和个人效果报告不在本轮提交中。
完整 fresh RTX 安装及全部容器组合仍未实机验收：固定 unit/目录的新安装器不能无影响地
与旧实例同机演练。更换模型、Qwen 语义检索、NVR/连续录像及实地跨人员走测仍需独立选择
和验收，不能由本次旧实例代码更新自动推导为完成。

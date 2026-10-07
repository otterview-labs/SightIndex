# 模型部署、授权与版本管理

SightIndex 的代码发布、模型资产、运行环境和历史索引是四个独立的部署对象。更新代码不会自动取得模型使用权，也不应自动替换模型或重建历史向量。本指南说明模型准备和验收的边界；完整服务部署流程见 [一键部署说明](one-click-deployment.zh-CN.md)。

模型准备应在部署前由操作者明确选择能力、审查来源及条款，再执行只读检查、离线导入或显式下载。请求处理、API 启动和模型首次加载都不应成为隐式下载入口。不要为了绕过缺失模型错误而开启 `FACE_ALLOW_DOWNLOAD`。

## 模型分别负责什么

| 能力 | 资产与部署角色 | 输出及作用 | 不代表什么 |
| --- | --- | --- | --- |
| 人体检测、视频抽帧裁剪 | `yolo11n.pt`；`yolo.person` | 人体框，用于裁剪与后续处理 | 检出人体不等于识别身份，也不等于动作、场景标签已生成 |
| 人脸检测、对齐与特征 | InsightFace `buffalo_l`；`face.*` | 人脸框、关键点及 512 维身份特征；实际启用范围还取决于配置 | 人脸模型存在不等于历史媒体已经有人脸索引；也不保证模糊、侧脸等样本质量 |
| 人体 ReID | SapiensID `model.pth`、`model.yaml`；`reid.checkpoint`、`reid.config` | 4096 维人体身份特征，用于同人检索 | 不是通用语义向量；不能用“穿黑衣服”等描述替代身份质量评估 |
| ReID 预处理 | `yolov8n-pose.pt`、DFA `mobilenetv4_Final.pth`，以及代码中的 `yolo_dfa.yaml`；`reid.pose`、`reid.dfa` | 姿态与对齐预处理，是 ReID 模型身份的一部分 | 不能只更新 SapiensID 主权重而假定向量仍兼容 |
| 视觉、文本语义向量 | `Qwen/Qwen3-VL-Embedding-2B` 的本地权重、配置、分词器、处理器及固定推理脚本；`qwen.*` | 当前容器服务输出 2048 维向量，支持外观、动作、场景等语义检索 | 不是人体身份或人脸身份向量；Embedding 服务就绪不等于历史向量覆盖完整 |
| 标签、描述及结构化语义 | 独立的 VLM 上游端点与模型配置 | 按启用的任务生成标签、描述或结构化内容 | 不由 Qwen Embedding 服务代替；本模型准备入口不安装 VLM 权重、不自动启动付费上游 |

`semantic` 是搜索与索引配置，不是另一个需要下载的大模型；当前容器组合要求同时选择 `embedding`。向量搜索、标签生成、人脸和人体身份识别的开关、就绪状态及覆盖率应分别验收。`VLM_PROVIDER=none` 应报告“未启用”，不能报告“标签模型已就绪”。

## 按部署能力选择资产

| 部署目标与组合 | 必需资产 | 运行边界 |
| --- | --- | --- |
| 容器 `base` | YOLO 人体检测、完整 `buffalo_l` 包 | 当前基础 Compose 的 API 使用 CPU 检测与人脸推理；仍需准备模型，不能依赖首次请求下载 |
| 容器 `base reid` | 基础资产，加 SapiensID、YOLO pose、DFA | ReID 使用 GPU；主权重与全部预处理资产必须符合配置中的复合指纹 |
| 容器 `base embedding` | 基础资产，加固定 Qwen Embedding 包 | 使用已构建的 Embedding 服务镜像；权重和推理脚本必须符合该镜像的固定 SHA-256 |
| 容器 `base embedding semantic` | 与上一行相同 | 还需明确启用视觉索引与语义搜索；不自动回填历史媒体 |
| 容器 `base reid embedding semantic` | 上述全部资产 | 两个 GPU 模型服务可能共享同一设备；需测量联合峰值显存和延迟 |
| RTX 5090 systemd | 基础资产，加 ReID 的全部资产 | 当前 RTX 安装流程使用 YOLO、人脸和 ReID GPU 推理；该目标的模型准备入口不管理 Qwen，Embedding 应使用容器目标 |

不要为“以后可能使用”而默认下载所有权重。选择组合前确认磁盘空间、下载流量、临时文件空间、主存和显存预算。服务镜像、Python/CUDA 运行库和模型包是不同对象；只有权重包并不能启动服务。

## 锁文件是审查结果，不是可信来源的替代品

模型准备入口使用操作者提供并审查的 manifest/lockfile，按固定模型角色映射目标位置。至少应审查并记录每个文件的角色、模型 ID、发布版本、相对包内路径、精确字节数、SHA-256、发布者来源以及许可证和条款地址。下载地址应绑定已审查版本，不应把可变化的 `latest`、分支 HEAD 或远程脚本当作版本锁。

SHA-256 只能证明文件与锁文件一致。如果锁文件和文件一起来自不可信来源，校验通过并不能证明发布者真实性、授权有效或模型安全。`*.pt` / `*.pth` 加载和 Qwen 包内的 Python 推理脚本都涉及代码执行风险；必须先审查来源，在受控运行环境中加载，不能从上传请求接收模型或修改锁文件。

当前角色映射要求：

- YOLO：`yolo.person`，模型 ID 为 `yolo11n`。
- InsightFace：模型 ID 为 `buffalo_l`，完整包包含 `det_10g.onnx`、`w600k_r50.onnx`、`1k3d68.onnx`、`2d106det.onnx` 和 `genderage.onnx`。基础部署预检及人脸合成加载验收关注检测、识别两个核心模型；这不等同于全部头模型的业务质量验收。
- ReID：`reid.checkpoint`、`reid.config`、`reid.pose`、`reid.dfa`，模型族 ID 为 `sapiensid_wb12m`。`yolo_dfa.yaml` 来自可信代码发布，不由模型包替换。
- Qwen：`qwen.weights`、`qwen.config`、`qwen.tokenizer-config`、`qwen.tokenizer`、`qwen.processor`、`qwen.runtime`，模型 ID 为 `Qwen/Qwen3-VL-Embedding-2B`；所选 Qwen 文件应属于同一审查版本。当前镜像要求单文件 `model.safetensors` 和 `scripts/qwen3_vl_embedding.py`，不是任意 Hugging Face 分片布局或任意同名模型。

锁文件可以覆盖多个标准组合，但执行时只准备所选组合的角色。未知角色、越界路径、缺失必需文件、字节数或摘要不符，都应停止准备，不应静默忽略或回退到网上下载。

锁文件顶层字段为 `version` 和 `artifacts`，当前 `version` 只能为整数 `1`。每项必需字段为 `id`、`path`、`sha256`、`size_bytes`、`model`、`revision`；部署适配层还要求非空 `license` 和 HTTPS `terms_url`。离线导入可不提供 `source_url`；显式下载缺失文件时则须有已审查的 HTTPS `source_url`，不能把认证信息写入 URL。

下面仅演示字段结构，**不是可运行锁文件**：SHA、大小、版本和条款必须换成实际审查结果，且所选角色必须全部列出。`size_bytes` 的实际值须为正整数，不能保留示例字符串。不要用全零或随意生成的摘要填补未审查来源。

```json
{
  "version": 1,
  "artifacts": [
    {
      "id": "yolo.person",
      "path": "base/yolo11n.pt",
      "sha256": "<REVIEWED_SHA256_64_HEX>",
      "size_bytes": "<ACTUAL_POSITIVE_INTEGER>",
      "model": "yolo11n",
      "revision": "<REVIEWED_PUBLISHER_REVISION>",
      "source_url": "https://publisher.example.invalid/releases/REVIEWED_REVISION/yolo11n.pt",
      "terms_url": "https://publisher.example.invalid/REVIEWED_TERMS",
      "license": "<REVIEWED_LICENSE_AND_WEIGHT_TERMS>"
    }
  ]
}
```

以下是一个离线包的角色与包内路径示例。`path` 相对于 `--model-source`，不是主机目标路径；操作者可采用不同的安全相对路径，但不能通过 manifest 指定任意写入位置。

| 组合新增 | `id` | 示例 `path` | `model` |
| --- | --- | --- | --- |
| base | `yolo.person` | `base/yolo11n.pt` | `yolo11n` |
| base | `face.det_10g` | `face/det_10g.onnx` | `buffalo_l` |
| base | `face.w600k_r50` | `face/w600k_r50.onnx` | `buffalo_l` |
| base | `face.1k3d68` | `face/1k3d68.onnx` | `buffalo_l` |
| base | `face.2d106det` | `face/2d106det.onnx` | `buffalo_l` |
| base | `face.genderage` | `face/genderage.onnx` | `buffalo_l` |
| reid | `reid.checkpoint` | `reid/model.pth` | `sapiensid_wb12m` |
| reid | `reid.config` | `reid/model.yaml` | `sapiensid_wb12m` |
| reid | `reid.pose` | `reid/yolov8n-pose.pt` | `sapiensid_wb12m` |
| reid | `reid.dfa` | `reid/mobilenetv4_Final.pth` | `sapiensid_wb12m` |
| embedding | `qwen.weights` | `qwen/model.safetensors` | `Qwen/Qwen3-VL-Embedding-2B` |
| embedding | `qwen.config` | `qwen/config.json` | `Qwen/Qwen3-VL-Embedding-2B` |
| embedding | `qwen.tokenizer-config` | `qwen/tokenizer_config.json` | `Qwen/Qwen3-VL-Embedding-2B` |
| embedding | `qwen.tokenizer` | `qwen/tokenizer.json` | `Qwen/Qwen3-VL-Embedding-2B` |
| embedding | `qwen.processor` | `qwen/preprocessor_config.json` | `Qwen/Qwen3-VL-Embedding-2B` |
| embedding | `qwen.runtime` | `qwen/scripts/qwen3_vl_embedding.py` | `Qwen/Qwen3-VL-Embedding-2B` |

`reid.pose` 和 `reid.dfa` 的 `model` 是所属的 ReID 管线身份，不是该辅助模型自己的发行名称。可选 Qwen 辅助文件只允许部署适配层列明的 `qwen.extra.*` 角色，仍需逐文件审查，例如 `qwen.extra.vocab.json`、`qwen.extra.merges.txt`。不存在可用的已审查 SHA 时，入口不会猜测摘要或自动生成“可信”锁文件。

## 目标路径与权限

容器目标按私有环境文件映射到以下主机位置，再以只读挂载提供模型：

```text
MODEL_DIR/
├── yolo11n.pt
├── insightface/models/buffalo_l/
│   ├── det_10g.onnx
│   ├── w600k_r50.onnx
│   ├── 1k3d68.onnx
│   ├── 2d106det.onnx
│   └── genderage.onnx
├── sapiensid_wb12m/{model.pth,model.yaml}
├── yolov8n-pose.pt
└── dfa_mobilenetv4_medium/mobilenetv4_Final.pth

QWEN_EMBEDDING_MODEL_DIR/
├── model.safetensors
├── config.json
├── tokenizer_config.json
├── tokenizer.json
├── preprocessor_config.json
└── scripts/qwen3_vl_embedding.py
```

Qwen 包可能还有经过锁文件审查的分词器或处理器辅助文件，不能遗漏实际推理依赖。主机 `MODEL_DIR` 与 `QWEN_EMBEDDING_MODEL_DIR` 应为专用绝对路径；媒体目录、数据库、私有 `.env` 和模型目录不能混用。模型通常以只读挂载进入容器，运行缓存另放在可写 cache 目录。

RTX 目标只接受可信 `/opt/sightindex` 代码树，主要路径由私有环境文件配置：

| 角色 | 当前 RTX 主机路径 |
| --- | --- |
| 人体检测 | `YOLO_MODEL`，模板为 `/var/lib/sightindex/models/yolo11n.pt` |
| 人脸包 | `FACE_INSIGHTFACE_ROOT/models/buffalo_l/`，模板 root 为 `/var/lib/sightindex/models/insightface` |
| ReID 主模型与配置 | `REID_CHECKPOINT_DIR/{model.pth,model.yaml}`，模板目录为 `/var/lib/sightindex/models/sapiensid_wb12m` |
| ReID pose | 固定 `/var/lib/sightindex/.cache/yolov8n-pose.pt`，须与服务 HOME 解析的缓存位置一致 |
| ReID DFA | `/opt/sightindex/deploy/agx/reid_service/sapiensid/tasks/sapiensID/src/aligners/keypoint_predictor/pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth` |

不能简单把所有 RTX 文件复制到 `models/` 根目录：pose 和 DFA 有不同的运行时位置。DFA 位于可信代码树，写入须由部署侧完成；应用服务账户只应拥有读取代码与模型的权限，不应获得修改模型、推理脚本或服务代码的权限。准备后还需确认服务用户可穿越父目录并读取文件。不要用全目录递归 `chmod 777` 解决权限问题。

## 离线准备与显式下载

建议在受控环境中生成并审查锁文件，再通过离线包导入部署主机。离线包只包含所需的模型及配置文件，按锁文件中的包内路径组织；不要包含 API Key、私有环境文件、数据库、用户图片或视频。传输后重新校验每个文件，不能仅依赖打包前校验。

显式下载是独立的、有成本和网络边界的操作，应先选择目标与组合、确认条款、审查完整锁文件和下载来源，再发起。部署脚本的普通检查、构建和启动步骤不应悄悄联网获取缺失权重。已经存在但摘要不匹配的资产需要人工处理，不能用“允许重新下载”掩盖运行模型发生变化。

准备使用目标旁的 `*.partial` 临时文件保存未完成内容，续传或续拷贝后仍以最终大小和 SHA-256 验证。只有校验通过的文件才会成为最终模型；已有最终文件摘要不符时不会直接覆盖。部分完成的包不等于整个组合已准备成功，应查看每个角色的结果并重新检查。

磁盘预检按实际目标文件系统计算需求并留安全余量：离线导入按剩余缺失字节，下载按仍需传输的完整文件大小计算，涵盖服务器忽略 Range 而重传整文件的情况。RTX 的模型目录、HOME 缓存和代码树可能分别位于不同文件系统，不能只检查某一个目录。操作系统、镜像构建、已保留旧版本、源离线包和其他进程也需要空间，工具的最低余量不是完整容量规划。不要删除仍在执行的准备锁或将一个版本的 partial 文件挪作另一个版本使用。

模型准备不是服务部署：它不修改数据库、不回填索引、不启动摄像头、不安装 GPU 驱动，也不启动标签 VLM。先准备资产，再执行部署只读预检，最后进入服务部署与验收流程。

### 统一入口命令

从已审查的仓库根目录执行。下面的 `/bundle` 和私有环境文件路径均是示例；环境文件应由操作者预先审查、限制权限，且模型目录应已按实际部署主机配置。准备命令会写入模型目标目录，显式下载命令会访问锁文件中的上游，不能把它们当作只读诊断。

仅检查当前目标文件，不创建目标目录、不导入、不下载、不启动服务：

```bash
bash deploy.sh --models-only --target containers \
  --env-file /data/sightindex-bj-test/.env \
  --stacks 'base reid' \
  --model-manifest /bundle/models.lock.json \
  --check
```

离线导入所选角色，只有明确确认条款后才准备文件：

```bash
bash deploy.sh --models-only --target containers \
  --env-file /data/sightindex-bj-test/.env \
  --stacks 'base reid' \
  --model-manifest /bundle/models.lock.json \
  --model-source /bundle \
  --acknowledge-model-terms
```

明确允许从锁文件中的 HTTPS 地址下载缺失文件：

```bash
bash deploy.sh --models-only --target containers \
  --env-file /data/sightindex-bj-test/.env \
  --stacks 'base reid' \
  --model-manifest /bundle/models.lock.json \
  --download-models \
  --acknowledge-model-terms
```

`--model-source` 与 `--download-models` 互斥。只读 `--check` 不能与源目录或下载参数组合，不需要授权确认参数。切换组合时，环境文件、锁文件和目标目录也必须适配，不能只修改 `--stacks`。

### 准备中断后的诊断与受控恢复

新版准备器使用带静态版本标记的持久锁文件 `.sightindex-model-assets.lock`，配合 POSIX 非阻塞 `flock`。正常退出、强制终止或进程崩溃后，操作系统会释放内核锁；锁文件本身保留，不以“文件存在”判断正在准备，也不能手工删掉它，否则可能让两个进程锁住不同 inode 并同时写入。掉电后应先让文件系统完成恢复，再运行诊断；没有写完版本标记的文件仍按旧式空锁处理，不能自动认定无主。

此机制用于支持 `fcntl/flock`、同目录原子重命名及目录 `fsync` 的本机 POSIX 文件系统。网络盘、不同主机共享模型目录或不能保证这些语义的存储不在该互斥保证范围内，应把准备工作放在单一主机的受控本地目录。工具对不支持的操作失败退出，不通过删锁绕过。不同环境文件不得用不同锁根同时映射到同一组模型目标文件。

先通过模型 ONLY 入口诊断。命令只读取所选 manifest 和实际模型文件，不创建目录或锁，不联网、不启动服务：

```bash
bash deploy.sh --models-only --target containers \
  --env-file /data/sightindex-bj-test/.env \
  --stacks 'base reid' \
  --model-manifest /bundle/models.lock.json \
  --diagnose-models
```

诊断区分 `lock.absent`、`lock.released`、`lock.active`、`lock.legacy_empty` 和未知格式。`partial.unverified_prefix` 表示尚未完成、没有充分证据判坏的短文件；`partial.verified` 表示完整且摘要正确、可由后续准备发布的缓存。诊断 `ok` 只说明没有检测到锁阻断或完整性异常，**不表示模型已部署就绪**；缺失最终模型仍需要准备和 `--check`。

旧版 `O_EXCL` 准备器留下的空锁不能判断原进程是否还在运行，新版普通准备仍然拒绝它。必须先停止并确认所有模型准备进程已退出，尤其是旧版没有内核锁的进程，再明确选择维护恢复：

```bash
bash deploy.sh --models-only --target containers \
  --env-file /data/sightindex-bj-test/.env \
  --stacks 'base reid' \
  --model-manifest /bundle/models.lock.json \
  --recover-models --confirm-no-active-preparation
```

确认参数是操作者对“没有活跃准备任务”的明确声明，工具不能据此证明旧版进程已停止；活跃的新版内核锁仍会拒绝恢复，不能覆盖。恢复再次校验 manifest、固定角色映射及 ReID 管线指纹，不改环境文件或索引。旧空锁的精确空字节先保存在私密隔离目录，再原子替换为已持有内核锁的新版锁文件；原锁名全程存在，阻止旧版 `O_EXCL` 在切换间隙进入。未知正文、符号链接或硬链接锁不自动处理。

恢复默认只把所选角色旁“超过声明大小”或“完整但 SHA-256 不符”的 `*.partial` 原子移到同文件系统的 `.sightindex-model-assets-quarantine/<随机目录>/`，不删除，也不扫描或隔离清单外文件。隔离父目录和任务目录仅当前操作者可访问；原始文件权限、内容和 inode 保留。已有最终权重，无论是否匹配，都不被覆盖、移走或修改；不匹配会继续报告失败，需另行审查模型迁移。

短 partial 默认保留以便续跑。如果已明确确认某个角色的短缓存无法使用，例如离线导入报告前缀不一致，可以在上述恢复命令追加 `--quarantine-partial reid.checkpoint`。可重复指定多个**当前选定 manifest 的角色 ID**；不能提供任意路径，不会处理未选的 Qwen/其他模型。它只是保留并隔离缓存，不准备新文件，也不绕过最终校验。后续仍需独立运行离线导入或显式下载，再运行 `--check` 和实机模型加载验收。

恢复选项仅模型 ONLY 维护入口支持，普通部署不会自动恢复。`--diagnose-models`、`--recover-models` 和 `--check` 互斥，均不能与下载或源目录参数组合。维护结果不打印锁正文、上游 URL 或响应内容；隔离后的旧锁、缓存可供人工审查和恢复，未经确认不要删除。

普通应用部署可联动“模型准备 → 模型及环境预检 → 构建、上线、验收”。以下命令会部署服务，与上面的 model-only 边界不同：

```bash
bash deploy.sh --target containers \
  --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env \
  --stacks 'base reid' \
  --prepare-models \
  --model-manifest /bundle/models.lock.json \
  --model-source /bundle \
  --acknowledge-model-terms
```

普通部署只传 `--model-manifest`、不传 `--prepare-models` 时，只校验既有模型，不下载或导入。普通部署加 `--check` 还会检查环境、模型与部署工具；不会因准备参数而在只读检查中写入权重。需要联网准备时，明确把 `--model-source /bundle` 换为 `--download-models`，并保留条款确认。

RTX 对应命令需在可信 `/opt/sightindex` 执行，改用 `--target rtx5090`、实际 RTX 私有环境文件路径和 `--stacks 'base reid'`。RTX 的模型准备及普通安装入口都不管理 `embedding`/`semantic`，不能用这些参数暗中启动 Qwen 或标签 VLM。

容器 Embedding 的服务镜像与权重准备分开。选择包含 `embedding` 的组合，并在普通容器部署中明确加 `--build-model-services`，才会构建对应 Embedding 服务镜像。该参数不下载模型、不启动标签 VLM；未选择 `embedding` 的组合不能使用它。使用已有服务镜像时，也应保留部署的镜像身份与实际就绪校验，不要仅依赖可变 tag。

## 授权必须分别核对

本项目保留了 [第三方告知](../THIRD_PARTY_NOTICES.md) 和 [SapiensID 的 CC BY-NC 4.0 许可证](../deploy/agx/reid_service/sapiensid/LICENSE)。这些文件应随相关组件分发，部署者还需保留作者归属、上游链接、修改说明及其他要求。

- SapiensID 的 vendored 代码采用 CC BY-NC 4.0，包含非商业使用限制。其主权重由发布者另行提供，代码许可证不自动替代权重或下载平台条款。包内 OpenPose 相关预处理也保留非商业使用告知。
- InsightFace 代码采用 MIT，与官方预训练模型的许可不同。仓库当前使用的 InsightFace 0.7.3 包说明官方预训练模型限非商业研究，手动下载与自动下载都受此约束。将文件离线导入并不会改变该限制。
- Ultralytics YOLO 的代码、模型及适用的 AGPL/企业授权条件需要分别审阅；不要把“公开可下载”解释为无需履行条款。
- Qwen 权重、处理器、分词器和外部推理脚本，以及外部标签 VLM 的模型、服务和数据处理条款，均应按实际发布版本审阅。向外部 VLM 发送用户画面之前，还需确认数据出境、隐私、留存和费用边界。

`--acknowledge-model-terms` 只表示操作者已确认审查相关条款，不授予商业许可、不代表发布者同意商用，也不是法律意见。商业部署前，应取得适用的授权或换用具有合适许可的完整模型与预处理链；替换后还必须重新验证运行接口与向量兼容性。不能仅换掉主模型而沿用受限的辅助模型。

## GPU 和联合资源验收

GPU 驱动、Docker GPU runtime、镜像 CUDA/PyTorch/ONNX Runtime 版本和主机设备权限需作为同一运行链检查。`nvidia-smi` 可见、容器可枚举 GPU 或“剩余显存超过预检下限”都不能证明模型已成功加载或联合负载不会 OOM。不要在模型准备过程中自动升级驱动或替换主机 CUDA 环境。

当前容器 ReID 和 Qwen 默认都使用 GPU 0。ReID 的 batch 限制、Qwen 的队列和并发限制有助于控制负载，但不是显存隔离；Qwen 的进程显存比例设置也不能为其他服务保证剩余容量。开启两者前应记录空闲显存、加载后的常驻占用、并发抽帧/检索/文本查询的峰值占用、延迟和失败情况。RTX API 中的 YOLO 与人脸推理还会占用同一 GPU，不能只计算 ReID 的预算。

先用本地合成输入验证预置模型可加载、实际选择的推理 provider、向量维度及有限数值，再在获得授权的样本上评估质量。合成图片上没有检测到人脸或人体是正常结果；不能据此声称人脸/人体识别准确率已经通过。

## 更新模型不能混用历史向量

只有模型文件、预处理、维度、度量和逻辑索引命名空间保持一致，才可认为原向量身份未变。搬迁同一个 Milvus 的访问地址不应无意改变逻辑命名空间；换用模型也不能因维度相同而沿用旧向量。

ReID 的 checkpoint revision 是以下文件 SHA-256 的有序复合摘要：`model.pth`、`model.yaml`、可信代码中的 `yolo_dfa.yaml`、`mobilenetv4_Final.pth`、`yolov8n-pose.pt`。模型准备会与环境文件中的 `REID_CHECKPOINT_REVISION` 比较；不一致时应停止并审查迁移方案，不会自动改写该值。ReID 的索引指纹还包含模型 ID、维度、预处理版本和命名空间，集合名称包含模型身份摘要。

通用人脸和视觉向量集合不能仅靠相同维度或同名文件保证隔离。更换这些模型或其预处理时，应显式选择新集合/集合前缀，保留旧索引，受控回填，验收后再切换查询。不要把新模型生成的少量向量追加到旧集合并期待查询结果可比较。

建议更新顺序：

1. 保留旧模型锁文件、模型资产、环境配置、代码 revision、服务镜像及索引标识，明确回退方案。
2. 在新目录准备并校验新版本；审阅变更的模型、配置、脚本及许可证，确定新的完整指纹。
3. 使用独立索引集合部署并完成运行时加载、API 合同和合成上传播放验收。
4. 对明确授权的历史数据执行单独的回填操作，检查成功、失败和待处理覆盖率；普通模型部署不自动回填。
5. 在隔离评估集上比较检索质量、阈值、延迟及资源，再决定切换查询和后续旧资产清理。

切回旧模型并不自动恢复数据库或索引。代码、权重、环境配置和集合选择都要对齐；未经确认不要删除旧集合或真实媒体。

## 验收记录应该包含什么

至少记录目标、组合、代码 revision、镜像身份、审查过的锁文件、每个模型文件的摘要、实际模型 ID/维度/预处理版本、索引集合、实际推理设备/provider、加载结果及未启用能力。

文件完整性、服务就绪、覆盖率和业务质量应分开记录：

- 文件校验：所选角色齐全，大小和 SHA-256 与可信锁文件一致。
- 运行时验收：已部署服务加载预置模型，ReID 和 Embedding 实际 ready；API/数据库只读合同检查通过。
- 合成端到端验收：临时 SQLite 与临时媒体目录中的上传、抽帧、裁剪、播放映射及 HTTP Range 流程通过，不读取或写入真实媒体、数据库。
- 数据覆盖：历史 pending、indexed 和 failed 等状态明确；服务 ready 不表示覆盖率为 100%。
- 质量评估：单独使用经授权、有代表性的评估集，记录身份检索、语义检索及标签质量；外部 VLM 只检查配置时，应明确没有验证其生成质量。

任何一个层次尚未通过，都应如实报告；不能用模型文件存在或健康接口返回成功替代其他层次。

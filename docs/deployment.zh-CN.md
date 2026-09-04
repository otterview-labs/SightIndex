# 部署指南

[English](deployment.md) | [简体中文](deployment.zh-CN.md)

本文档说明本仓库中实际提供的部署资源。主要生产部署路径是从源码构建，并由 systemd
监管。PostgreSQL 以及启用时的 Milvus 使用 Docker Compose 运行。SightIndex 当前不包含
应用程序 Dockerfile，也不提供一体化 Compose 技术栈。

## 部署方案

推荐的基线方案是在一台 Linux 主机上运行 API 和控制台，将其绑定到环回地址，并置于
TLS 反向代理之后。PostgreSQL 是标准数据库。Milvus 和 GPU 模型服务均为可选组件。

SightIndex 的 API 进程负责流捕获线程和向量索引队列。必须只运行一个 Uvicorn worker。
多个 API worker 可能重复捕获同一路流，并重复处理相同的队列任务。

### 端口映射

| 组件 | 容器/进程端口 | 推荐的主机绑定 | 公网暴露方式 |
| --- | ---: | --- | --- |
| SightIndex API + 控制台 | 默认为 `8000` | `127.0.0.1:8000` | 仅通过 TLS 反向代理 |
| Vite 开发服务器 | `5173` | `127.0.0.1:5173` | 禁止公开 |
| PostgreSQL | `5432` | `127.0.0.1:5432` | 禁止公开 |
| Milvus gRPC | `19530` | `127.0.0.1:19530` | 禁止公开 |
| Milvus 健康检查/指标 | `9091` | `127.0.0.1:9091` | 禁止公开 |
| 视觉嵌入服务 | `18021` | `127.0.0.1:18021` | 禁止公开 |
| Qwen 视觉重排服务 | `18022` | `127.0.0.1:18022` | 禁止公开 |
| SapiensID ReID 服务 | `18031` | `127.0.0.1:18031` | 禁止公开 |
| 可选的外部 YOLO 服务 | `19121` | `127.0.0.1:19121` | 禁止公开 |

Milvus Compose 文件不会发布其 etcd 或 MinIO 端口。请保持这一安全边界不变。

## 前置条件

- 较新的、使用 systemd 的 Linux 发行版。
- Python 3.11 或更高版本。
- Node.js 22.18 或更高版本以及 npm。
- Git、curl 和 C/C++ 构建工具链。
- PostgreSQL 和 Milvus 所需的 Docker Engine 与 Docker Compose v2。
- 足够的存储空间，用于数据库、源媒体、生成的帧/裁剪图及备份。
- NVIDIA 推理需要与所选模型技术栈兼容的驱动和运行时组合。
- Jetson 需要与已安装 JetPack 版本兼容、支持 CUDA 的 PyTorch 构建。

以下命令使用这些路径和身份：

```text
Application:        /opt/sightindex
Deployment account: sightindex-deploy
Service account:    sightindex
Runtime data:       /var/lib/sightindex/data
```

部署账号拥有已经审核的源码、虚拟环境和前端构建产物。服务账号可以读取这些文件，但只
能写入 `/var/lib/sightindex` 下的运行时路径。这样可以防止遭入侵的应用进程替换管理员
日后使用 `sudo` 执行的脚本。`deploy/systemd/` 下的 systemd 模板使用相同的路径。

## 1. 安装源码

创建相互独立、不可登录的服务账号和部署账号。两者都使用 `sightindex` 组，但只有部署
账号拥有应用程序检出目录：

```bash
sudo groupadd --system sightindex
sudo useradd --system --gid sightindex --create-home --home-dir /var/lib/sightindex \
  --shell /usr/sbin/nologin sightindex
sudo useradd --system --gid sightindex --create-home --home-dir /var/lib/sightindex-deploy \
  --shell /usr/sbin/nologin sightindex-deploy
sudo install -d -m 0750 -o sightindex-deploy -g sightindex /opt/sightindex
sudo install -d -m 0750 -o root -g sightindex /var/lib/sightindex
sudo install -d -m 0700 -o sightindex -g sightindex /var/lib/sightindex/data
sudo install -d -m 0700 -o sightindex -g sightindex /var/lib/sightindex/.cache
```

如果这些身份已经存在，请核对它们的主目录、主组和目录所有权，而不要重新创建。

克隆经过审核的发布提交或标签。不要通过仅复制一部分变更文件来部署：

```bash
sudo -u sightindex-deploy -H sh -c 'umask 027; git clone https://github.com/otterview-labs/SightIndex.git /opt/sightindex'
sudo -u sightindex-deploy -H git -C /opt/sightindex rev-parse HEAD
```

在发布工单或部署日志中记录该提交。确认 `sightindex` 可以读取检出目录，但不能修改它：

```bash
sudo -u sightindex test -r /opt/sightindex/main.py
if sudo -u sightindex test -w /opt/sightindex; then
  echo 'unsafe: service account can modify the deployment checkout' >&2
  exit 1
fi
```

## 2. 创建 Python 环境和前端构建产物

```bash
cd /opt/sightindex
sudo -u sightindex-deploy -H python3 -m venv .venv
sudo -u sightindex-deploy -H .venv/bin/python -m pip install --upgrade pip
sudo -u sightindex-deploy -H .venv/bin/python -m pip install -r requirements.txt
sudo -u sightindex-deploy -H npm --prefix frontend ci
sudo -u sightindex-deploy -H npm --prefix frontend run build
test -f frontend/dist/index.html
```

FastAPI 直接提供 `frontend/dist/`；生产环境中没有单独的前端服务。

`requirements.txt` 安装常规 API 运行时依赖。可选的视觉和 GPU 路径另有附加依赖文件。
仅安装当前主机需要的方案：

```bash
# Local visual embedding and model tooling
sudo -u sightindex-deploy -H .venv/bin/python -m pip install -r requirements.visual.txt

# Jetson/AGX non-PyTorch dependencies
sudo -u sightindex-deploy -H .venv/bin/python -m pip install -r requirements.agx.txt
```

`requirements.agx.txt` 有意不安装 PyTorch 或 torchvision。在 Jetson 上，请提供与 JetPack
兼容的 PyTorch 环境，并通过 `REID_SERVICE_PYTHONPATH` 或
`QWEN3_VL_EMBEDDING_PYTHONPATH` 暴露该环境。若不知道具体的 JetPack 版本，就不存在
能够选择正确 Jetson wheel 的通用命令。

## 3. 管理环境文件所有权

使用仓库中的模板创建运行时文件。该文件归 root 所有，服务组只读；应用进程不能修改
部署密钥或启动参数：

```bash
cd /opt/sightindex
sudo install -o root -g sightindex -m 0640 .env.example .env
sudoedit .env
```

`.env` 由部署操作人员或密钥管理系统负责，而不是由 Git 管理。至少要检查以下内容：

```dotenv
ENVIRONMENT=production
APP_HOST=127.0.0.1
APP_PORT=8000
PUBLIC_BASE_URL=https://sightindex.example.com
DATA_DIR=/var/lib/sightindex/data

DATABASE_URL=postgresql+psycopg://sightindex:REPLACE_WITH_URL_SAFE_PASSWORD@127.0.0.1:5432/sightindex
POSTGRES_DB=sightindex
POSTGRES_USER=sightindex
POSTGRES_PASSWORD=REPLACE_WITH_A_RANDOM_SECRET

APP_BASIC_AUTH_USERNAME=operator
APP_BASIC_AUTH_PASSWORD=REPLACE_WITH_A_RANDOM_SECRET
```

使用随机生成且 URL 安全的数据库密码，或在 `DATABASE_URL` 中对密码进行百分号编码。
绝不要提交最终生成的文件。如果反向代理提供了更强的身份感知访问控制，请仍将应用端口
保持在环回地址上，并记录由哪一层负责身份验证。

示例配置将所有可选提供方保持为禁用状态。只有当对应服务、模型、维度和凭据都准备
就绪后，才能逐一启用。

systemd、Compose 和部署辅助脚本使用的环境文件必须只包含简单的 `KEY=value` 赋值。
包含空格的值应加引号。不要在其中放置 shell 命令或变量替换。
请使用十六进制或 base64url 密钥；RTX 辅助脚本会拒绝 shell 展开和命令控制字符，而不
会尝试重新解释它们。

## RTX 5090 自动化方案

`deploy/rtx5090/` 为配备 NVIDIA RTX 5090 的单台 x86_64 主机，自动执行与上述相同的
源码构建、systemd 和 Milvus 部署路径。它是硬件方案，而不是第二套应用架构。仓库中的
模板默认使用 SQLite，以适合紧凑型单机部署。对于持续并发采集、多操作人员场景，或者
站点平台已具备 PostgreSQL 备份与监控能力时，应优先使用 PostgreSQL。

该包装脚本有意不执行 `git pull`。更新时，请进入维护窗口，并在更改版本之前停止所有
可能从检出目录延迟导入或执行文件的进程。记录回填 worker 是否处于活动状态，以便明确
恢复它。确认所有写入进程停止后，对 `DATA_DIR` 和密钥管理的配置做协调一致的备份，并
让服务在检出代码和包装脚本执行数据库备份期间持续保持停止：

```bash
sudo -u sightindex-deploy -H git -C /opt/sightindex rev-parse HEAD  # record rollback revision
sudo systemctl is-active sightindex-attribute-backfill.service \
  && BACKFILL_OPTION=--start-backfill || BACKFILL_OPTION=
sudo systemctl disable --now sightindex-embedding.service 2>/dev/null || true
sudo systemctl stop sightindex-attribute-backfill.service \
  sightindex-api.service sightindex-reid.service 2>/dev/null || true
for unit in sightindex-attribute-backfill sightindex-embedding sightindex-api sightindex-reid; do
  if sudo systemctl is-active --quiet "$unit"; then
    echo "failed to stop $unit" >&2
    exit 1
  fi
done
if sudo systemctl is-enabled --quiet sightindex-embedding.service; then
  echo 'failed to disable sightindex-embedding' >&2
  exit 1
fi
# Take the DATA_DIR snapshot here; do not restart any writer afterward.
sudo -u sightindex-deploy -H git -C /opt/sightindex fetch --tags origin
sudo -u sightindex-deploy -H git -C /opt/sightindex checkout --detach <reviewed-commit-or-tag>
sudo -u sightindex-deploy -H git -C /opt/sightindex status --short
sudo -u sightindex-deploy -H git -C /opt/sightindex rev-parse HEAD
```

仅在首次安装时创建并编辑 RTX 专用环境文件：

```bash
cd /opt/sightindex
if ! sudo test -e .env; then
  sudo install -o root -g sightindex -m 0640 \
    deploy/rtx5090/sightindex.env.example .env
  sudoedit .env
fi
```

更新时绝不要覆盖线上 `.env`。对比已记录的新旧提交，检查仓库中模板的变更，然后只对
必须采用的设置执行 `sudoedit .env`。真实文件应保存在部署环境的密钥/配置备份中；不要
在差异中打印其值，也不要提交它。

至少要替换 `PUBLIC_BASE_URL` 和 `MINIO_ROOT_PASSWORD`。API、Milvus 和 ReID 监听地址
必须保持在环回地址上。可以配置应用 Basic Auth，也可以明确约定由 TLS 反向代理承担身份
验证边界。安装程序会拒绝仓库模板中的占位密钥和非 HTTPS 的公网 URL。

运行包装脚本前，请先准备好以下由操作人员管理的模型资源：

```text
/var/lib/sightindex/models/yolo11n.pt
/var/lib/sightindex/models/insightface/models/buffalo_l/det_10g.onnx
/var/lib/sightindex/models/insightface/models/buffalo_l/w600k_r50.onnx
/var/lib/sightindex/models/sapiensid_wb12m/model.pth
/var/lib/sightindex/models/sapiensid_wb12m/model.yaml
/var/lib/sightindex/.cache/yolov8n-pose.pt
```

确保 `sightindex` 可以读取每个文件。启用 SapiensID 之前请检查
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)；其上游代码和权重受非商业条款约束。

更新时，应按照上文所示，在所有写入进程停止后对 `DATA_DIR` 做文件系统或对象存储快照。
选定源码版本后，包装脚本会在更改运行时依赖前确认服务仍未运行。它会自动备份 SQLite
或由仓库管理的 PostgreSQL 容器，以无特权部署账号构建依赖，保留之前的虚拟环境和前端
构建产物，启动 Milvus，然后按依赖顺序启动 ReID 和 API。

可选的 `sightindex-embedding` 服务使用单独的依赖方案，使用此包装脚本前必须停止并禁用
它。如需该服务，请将其作为单独的、经过审核的部署进行管理。

```bash
cd /opt/sightindex
sudo bash deploy/rtx5090/install_or_update.sh ${BACKFILL_OPTION:-}
```

ReID 模型预热可能需要最多五分钟。可通过以下命令查看可用选项：

```bash
bash deploy/rtx5090/install_or_update.sh --help
```

`--skip-backup` 表示明确确认已另行完成并验证数据库备份。仅当所有路径均归
`sightindex-deploy` 所有，且 `sightindex` 对它们均无写权限时，才能使用 `--skip-deps`
和 `--skip-frontend` 复用已有构建产物；否则应重新构建。

以服务身份运行验证脚本，使 CUDA 访问、模型可读性、HOME 和 `.env` 权限与实际进程一致：

```bash
sudo -u sightindex -H bash /opt/sightindex/deploy/rtx5090/verify.sh
```

该脚本会检查 systemd 状态、真实 CUDA 张量运算、ReID 模型身份和就绪状态、由数据库
支撑的 API、关键 OpenAPI 路由、Milvus 写入/搜索/删除、合成的 InsightFace ONNX session，
以及构建后的 `/reid` 控制台。它不能证明摄像头可访问，也不能证明新帧正在持续增加；
请另外使用测试流和外部 HTTPS URL 验证这些内容。

依赖文件采用兼容的版本范围，而不是针对硬件的锁定文件。将每个主机镜像视为可重复的
生产基线之前，应记录 `pip freeze`、NVIDIA 驱动、Python、PyTorch、CUDA、ONNX Runtime
和模型版本。

## 4. 启动 PostgreSQL

根目录的 Compose 文件只运行 PostgreSQL。它从 `.env` 读取 `POSTGRES_DB`、
`POSTGRES_USER` 和 `POSTGRES_PASSWORD`，默认绑定到环回地址。

```bash
cd /opt/sightindex
sudo docker compose --env-file .env config --quiet
sudo docker compose --env-file .env up -d postgres
sudo docker compose ps
```

等待健康检查通过：

```bash
until sudo docker compose exec -T postgres \
  sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'; do sleep 2; done
```

真实部署中不要使用示例密码。

## 5. 需要向量搜索时启动 Milvus

基线 API 不强制依赖 Milvus，但使用 Milvus 的视觉和 ReID 索引路径需要它。在 `.env`
中配置高强度 MinIO 凭据，然后验证并启动技术栈：

```dotenv
MINIO_ROOT_USER=sightindex
MINIO_ROOT_PASSWORD=REPLACE_WITH_A_RANDOM_SECRET
MILVUS_ENABLED=true
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION_PREFIX=sightindex
MILVUS_NAMESPACE_ID=production
```

```bash
cd /opt/sightindex
sudo docker compose --env-file .env -f deploy/milvus/docker-compose.yml config --quiet
sudo docker compose --env-file .env -f deploy/milvus/docker-compose.yml up -d
until curl --fail --silent http://127.0.0.1:9091/healthz; do sleep 3; done
sudo -u sightindex -H sh -c 'cd /opt/sightindex && .venv/bin/python scripts/check_milvus.py'
```

`scripts/check_milvus.py` 会写入、搜索、验证并删除临时向量。仅 HTTP 健康检查端点并不能
证明配置的 collection 路径可用。

在主机之间迁移同一个逻辑数据库时，请保持 `MILVUS_NAMESPACE_ID` 不变。模型或向量维度
变化时，应使用新的视觉 collection 前缀。

## 6. 可选模型服务

在 API 之前启动可选依赖，以便单独检查它们的就绪状态。

### SapiensID ReID

Git 中不包含大型 SapiensID checkpoint。请遵循
[`deploy/agx/reid_service/README.md`](../deploy/agx/reid_service/README.md)，审阅其上游的
非商业许可证，并在启动前放置所有必需资源。

相关设置：

```dotenv
REID_ENABLED=true
REID_SERVICE_URL=http://127.0.0.1:18031
REID_SERVICE_PORT=18031
REID_SERVICE_API_KEY=REPLACE_WITH_A_RANDOM_SECRET
REID_CHECKPOINT_DIR=/var/lib/sightindex/models/sapiensid_wb12m
REID_CHECKPOINT_REVISION=sha256:REPLACE_WITH_THE_READY_RESPONSE_VALUE
```

只有在所有必需资源都存在后，才能安装并启动该 unit：

```bash
sudo install -m 644 deploy/systemd/sightindex-reid.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sightindex-reid
curl --fail http://127.0.0.1:18031/health
curl --fail http://127.0.0.1:18031/ready
```

`/health` 表示进程存活。模型加载完成且其资源身份信息可用之前，`/ready` 会返回 `503`。
API 预期的 checkpoint 版本必须与 `/ready` 的准确响应相匹配。

### 视觉嵌入服务

可选的 systemd unit 会在 `127.0.0.1:18021` 上运行本地嵌入 worker。提供方、模型、维度、
设备和 API key 均来自 `.env`：

```bash
sudo install -m 644 deploy/systemd/sightindex-embedding.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sightindex-embedding
curl --fail http://127.0.0.1:18021/health
```

### 外部 YOLO 和 Qwen 重排辅助服务

`deploy/agx/start_yolo_service.sh` 和 `start_qwen3_vl_reranker_gpu.sh` 需要操作人员提供 NVIDIA
容器镜像、工作区/运行时和模型路径。本仓库不会构建这些镜像。缺少所需参数时，脚本会
尽早失败；在 API 中启用相应 URL 前，应将它们作为单独的部署资源进行配置和测试。

Qwen 重排服务默认端口为 `18022`，与 ReID 的 `18031` 分开。

## 7. 安装并启动 API 服务

数据库和所有已启用的模型服务准备就绪后，再安装 unit：

```bash
cd /opt/sightindex
sudo install -m 644 deploy/systemd/sightindex-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sightindex-api
sudo systemctl status --no-pager sightindex-api
```

该 unit 从 `.env` 读取 `APP_HOST` 和 `APP_PORT`，将日志写入 system journal，并以
`sightindex` 用户身份运行。

如果部署中包含后台属性回填，请仅在 API 健康后安装其独立 unit：

```bash
sudo install -m 644 deploy/systemd/sightindex-attribute-backfill.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sightindex-attribute-backfill
```

## 8. 验证部署

检查服务及其近期日志：

```bash
sudo systemctl is-active sightindex-api
sudo journalctl -u sightindex-api -n 100 --no-pager
curl --fail http://127.0.0.1:8000/health
```

`/health` 只能证明 HTTP 进程存活。请调用由数据库支撑的 API 作为业务冒烟测试：

```bash
curl --fail --user operator \
  http://127.0.0.1:8000/api/media/counts
```

只提供用户名时，curl 会提示输入密码，而不会把密码写入 shell 历史或进程命令行。如果
在受信任的本地代理后禁用了应用 Basic Auth，请省略 `--user`。

对于 ReID，仅 API 状态路由返回 `200` 并不足够；还应检查 JSON 字段：

```bash
curl --fail --user operator \
  http://127.0.0.1:8000/api/reid/status
```

确认 `enabled` 和 `ready` 与预期部署状态一致，并确认积压量和索引覆盖率合理。还应从
主机直接调用 ReID 服务的 `/ready` 端点。

最终的外部检查应使用通过反向代理的 TLS URL，并使用非管理员客户端账号。

## 数据与备份清单

应一起备份所有状态所有者：

| 状态 | 默认所有者 |
| --- | --- |
| 元数据 | PostgreSQL 卷，或 SQLite 使用的本地 `sightindex.db` |
| 媒体 | `DATA_DIR`：上传内容、视频、帧、裁剪图、缩略图、诊断数据 |
| Milvus 向量 | `milvus-etcd`、`milvus-minio` 和 `milvus-data` 卷 |
| 模型资源 | 操作人员管理的模型目录和缓存 |
| 运行时配置 | 由密钥管理系统保存的 `.env` 备份，而不是 Git |

数据库是元数据的事实来源；只有在保留全部源记录和模型身份信息时，Milvus 才是可重建
的索引。

## 升级流程

本项目当前使用自动建表和增量兼容迁移，而不是 Alembic。代码回滚不一定等同于数据库
schema 回滚。更改版本前，应停止写入进程并完成备份。

1. 记录当前提交以及镜像/模型版本。
2. 停止 API 和可选 worker。
3. 备份 PostgreSQL（或 SQLite 文件）、`DATA_DIR` 以及所有不可重建的 Milvus 状态。
4. 获取并检出经过审核的发布提交。
5. 安装经过审核的依赖集，并重新构建 `frontend/dist`。
6. 验证配置并运行测试。
7. 依次启动基础依赖、模型服务和 API。
8. 执行存活检查、数据库检查、向量检查和外部冒烟测试。

数据库备份示例：

```bash
set -euo pipefail
cd /opt/sightindex
PREVIOUS_COMMIT=$(sudo -u sightindex-deploy -H git -C /opt/sightindex rev-parse HEAD)
BACKUP_DIR=/var/lib/sightindex/backups
DATABASE_BACKUP="$BACKUP_DIR/sightindex-${PREVIOUS_COMMIT}.dump"
DATA_BACKUP="$BACKUP_DIR/data-${PREVIOUS_COMMIT}.tar.gz"
DATABASE_PARTIAL="${DATABASE_BACKUP}.partial"
DATA_PARTIAL="${DATA_BACKUP}.partial"
sudo systemctl stop sightindex-api sightindex-reid sightindex-embedding \
  sightindex-attribute-backfill 2>/dev/null || true
for unit in sightindex-api sightindex-reid sightindex-embedding sightindex-attribute-backfill; do
  if sudo systemctl is-active --quiet "$unit"; then
    echo "failed to stop $unit" >&2
    exit 1
  fi
done
sudo install -d -m 700 -o root -g root "$BACKUP_DIR"
sudo install -m 600 /dev/null "$DATABASE_PARTIAL"
sudo install -m 600 /dev/null "$DATA_PARTIAL"
sudo sh -c 'cd /opt/sightindex && docker compose --env-file .env exec -T postgres \
  sh -c '\''pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc'\'' > "$1"' \
  sh "$DATABASE_PARTIAL"
sudo test -s "$DATABASE_PARTIAL"
sudo sh -c 'cd /opt/sightindex && docker compose --env-file .env exec -T postgres \
  pg_restore --list < "$1" >/dev/null' sh "$DATABASE_PARTIAL"
sudo mv "$DATABASE_PARTIAL" "$DATABASE_BACKUP"
sudo tar -C /var/lib/sightindex -czf "$DATA_PARTIAL" data
sudo tar -tzf "$DATA_PARTIAL" >/dev/null
sudo mv "$DATA_PARTIAL" "$DATA_BACKUP"
```

执行这些命令前，应确认目标文件系统有足够的可用空间，并把备份文件当作敏感数据保护。

## 回滚

1. 停止 API 和 worker。
2. 检出此前记录的提交。
3. 恢复该版本对应的 Python 依赖，并重新构建其前端。
4. 如果较新版本更改了持久化状态，恢复匹配的数据库/数据备份。
5. 恢复匹配的模型版本和向量命名空间设置。
6. 按顺序启动基础依赖、可选模型服务和 API。
7. 重复执行全部冒烟测试。

即使旧进程能够成功启动，也绝不要让旧代码连接已发生不兼容、仅可向前演进变更的数据库。

## 安全检查清单

- 除非已明确配置防火墙和身份验证边界，否则 API 必须保持在环回地址上。
- 所有非本地访问均应置于 TLS 和用户身份验证之后。
- 为 PostgreSQL、Basic Auth、MinIO 和模型服务 API 分别使用高强度且互不相同的密钥。
- 不要公开暴露 PostgreSQL、Milvus、MinIO、ReID、嵌入、重排或 YOLO 端口。
- 将 RTSP URL、人脸嵌入、人员裁剪图和模型输出视为敏感数据。
- 使用 `MEDIA_RETENTION_DAYS` 限制保留时间，并先以 dry-run 模式测试清理。
- 将 `.env`、数据库备份、媒体目录和模型缓存的访问权限限制为文档规定的服务账号、部署
  账号或 root。
- 不要记录令牌、完整 RTSP URL 或原始生物特征 payload。
- 用于商业或生物特征场景前，请审查第三方模型许可证。

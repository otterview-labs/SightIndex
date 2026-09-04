# SightIndex

[English](README.md) | [简体中文](README.zh-CN.md)

SightIndex 是一项可自行托管的视觉索引与检索服务。它接收图像、视频以及 RTSP/HTTP
视频流，提取可检索的媒体内容和人员裁剪图，并通过 FastAPI API 与 Vue 控制台提供结果。

本仓库是实验性参考实现。人脸识别和人员重识别会处理生物特征数据。请仅在具备合法依据、
适当同意、访问控制、留存期限和人工复核机制的情况下部署这些功能。

## 包含内容

- 图像、视频和实时视频流接入。
- 人员检测、裁剪图生成、缩略图和计数线。
- 人员、人脸、视觉和结构化属性搜索 API。
- 可选的 InsightFace 人脸嵌入。
- 可选的 Milvus 视觉与人员 ReID 向量索引。
- 可选的基于 SapiensID 的跨摄像头候选检索。
- 由 FastAPI 进程提供服务的 Vue 3 操作控制台。
- 常规部署使用 PostgreSQL，本地评估使用 SQLite。

SightIndex 返回排序后的证据和置信度元数据。ReID 结果只是候选项，不能作为身份认定或
摄像头之间连续轨迹的证明。

## 架构

| 组件 | 作用 | 是否必需 |
| --- | --- | --- |
| FastAPI + Vue 控制台 | API、媒体接入、搜索和操作界面 | 是 |
| PostgreSQL 或 SQLite | 元数据和规范记录 | 是 |
| 本地文件系统 | 上传的媒体、帧、裁剪图、缩略图和模型 | 是 |
| Milvus | 视觉与 ReID 搜索的向量索引 | 可选 |
| ReID 服务 | 对人员裁剪图执行 SapiensID 推理 | 可选 |
| Embedding/VLM 服务 | 视觉嵌入、重排序和结构化属性 | 可选 |

API 进程还负责视频流捕获线程和向量索引队列。除非已将这些职责迁移到独立工作进程，
否则每套部署仅运行一个 API worker。

## 快速开始

前置条件：

- Python 3.11 或更高版本。
- 锁定的前端工具链需要 Node.js 22.18 或更高版本。
- 用于构建 Python 原生依赖的 C/C++ 编译工具链。

克隆公开仓库并创建本地环境：

```bash
git clone https://github.com/otterview-labs/SightIndex.git
cd SightIndex
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.dev.txt
cp .env.example .env
```

示例配置使用 SQLite，并禁用外部模型服务。构建控制台并启动 API：

```bash
npm --prefix frontend ci
npm --prefix frontend run build
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

在另一个终端中验证进程：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/api/media/counts
```

然后打开：

- 控制台：`http://127.0.0.1:8000/`
- OpenAPI：`http://127.0.0.1:8000/docs`

进行前端开发时，运行 `npm --prefix frontend run dev`；Vite 监听 `5173` 端口，并默认将
API 请求代理到 `http://127.0.0.1:8000`。

## 部署

仓库提供的部署资源采用由 systemd 管理的源码构建方式，并通过 Docker Compose 管理
PostgreSQL 和可选的 Milvus。应用本身目前不提供 Dockerfile。

部署指南提供 [English](docs/deployment.md) 和
[简体中文](docs/deployment.zh-CN.md) 两个版本，其中包括：

- 主机准备和依赖启动顺序；
- 环境配置文件的所有权和机密信息管理；
- PostgreSQL、Milvus、API 和可选 GPU 服务；
- 端口和网络暴露规则；
- 健康检查和业务级冒烟测试；
- 升级、备份和回滚。

`deploy/rtx5090/` 下提供 RTX 5090 单机配置。其封装脚本会构建全新的 CUDA 环境、验证模型
资源、备份本地 SQLite 或仓库管理的匹配 PostgreSQL 服务、按依赖顺序启动 Milvus 和
systemd 服务，并运行业务级冒烟测试。外部数据库需要另行完成经过验证的备份。该封装脚本
不会替你更新 Git 工作区；请先选择并记录一个经过审查的提交，然后按照专门的
[RTX 5090 章节](docs/deployment.zh-CN.md#rtx-5090-自动化方案)操作。

Jetson/AGX 辅助脚本位于 `deploy/agx/`。部分可选的 YOLO 和重排序辅助程序需要运维人员
提供容器镜像与模型路径；它们是集成方案，不是仓库内置镜像。

## 配置

`.env.example` 是配置模板的权威来源。它的默认配置刻意保持精简：

- 仓库工作目录中的 SQLite 数据库；
- 本地 `data/` 存储；
- 不启用 Milvus、VLM、视觉嵌入或 ReID 服务；
- API 绑定地址由启动 Uvicorn 的命令指定。

使用 PostgreSQL 进行开发时，请设置高强度的 `POSTGRES_PASSWORD`、更新 `DATABASE_URL`，
然后运行：

```bash
docker compose config --quiet
docker compose up -d postgres
```

使用 Milvus 时：

```bash
docker compose -f deploy/milvus/docker-compose.yml config --quiet
docker compose -f deploy/milvus/docker-compose.yml up -d
curl --fail http://127.0.0.1:9091/healthz
MILVUS_ENABLED=true .venv/bin/python scripts/check_milvus.py
```

Milvus 默认仅绑定回环地址。不要将 PostgreSQL、Milvus、模型服务或原始媒体存储直接暴露给
不受信任的网络。

## 模型服务

所有大型模型文件都是运行时资源，应在服务启动前有计划地下载。不要在请求处理程序中下载
模型。

- InsightFace 模型包应放在已配置的 `FACE_INSIGHTFACE_ROOT` 目录下。
- 通过对应的 `.env` 设置选择视觉嵌入与重排序提供方。
- 本仓库不存储 SapiensID 检查点；请参阅
  [deploy/agx/reid_service/README.md](deploy/agx/reid_service/README.md)。
- 仓库内置的 SapiensID 子集中包含的轻量 DFA 对齐器资源，按其上游非商业许可证纳入
  版本管理。

当模型、维度、预处理版本或逻辑命名空间发生变化时，请使用彼此独立的向量集合。

## API 示例

上传图像：

```bash
curl --fail -X POST http://127.0.0.1:8000/api/images/upload \
  -F 'file=@/path/to/image.jpg'
```

使用占位 URL 注册视频流：

```bash
curl --fail -X POST http://127.0.0.1:8000/api/streams \
  -H 'content-type: application/json' \
  -d '{
    "name": "entrance-camera",
    "stream_url": "rtsp://camera.example.test/stream1",
    "protocol": "rtsp",
    "frame_interval_seconds": 2
  }'
```

启动已注册的视频流：

```bash
curl --fail -X POST http://127.0.0.1:8000/api/streams/{stream_id}/start
```

搜索人员裁剪图：

```bash
curl --fail -X POST http://127.0.0.1:8000/api/search/person-crops \
  -H 'content-type: application/json' \
  -d '{"query":"red jacket and backpack","top_k":20,"filters":{}}'
```

嵌入视频流 URL 的凭据会与视频流记录一同存储。请使用专用摄像头账号、限制数据库访问，
并避免在 Shell 历史记录或日志中写入真实凭据。

## 开发

后端检查：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

前端检查：

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run build
```

API 契约发生变化时，请启动后端，然后重新生成并检查前端 Schema：

```bash
npm --prefix frontend run gen:api
npm --prefix frontend run check:api
```

## 仓库结构

```text
app/                   FastAPI 路由、服务、模型、Schema 和设置
frontend/              Vue 3 控制台
tests/                 后端测试
deploy/milvus/         本地 Milvus Compose 技术栈
deploy/systemd/        Linux 服务模板
deploy/agx/            可选的 Jetson/AGX 与模型服务辅助程序
deploy/rtx5090/        RTX 5090 环境、部署与验证辅助程序
docs/                  架构、API、校准与部署说明
scripts/               维护、导入、模型与诊断工具
```

更多参考资料：

- [视觉模型 API](docs/visual-ai-apis.md)
- [跨摄像头 ReID 设计](docs/cross-camera-reid.md)
- [ReID 操作演示与校准](docs/reid-walkthrough-calibration.md)
- [多模态检索说明](docs/multimodal-retrieval/README.md)
- [实现路线图](docs/implementation-roadmap.md)

## 许可证与第三方代码

本仓库目前没有授予覆盖整个项目的开源许可证。公开可见仅允许查看，本身并不授予复用权利。

仓库内的 SapiensID 推理子集单独采用 CC BY-NC 4.0 许可证。在使用或再分发它之前，请阅读
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 以及该源码旁保留的许可证。模型权重可能受
其他附加条款约束。

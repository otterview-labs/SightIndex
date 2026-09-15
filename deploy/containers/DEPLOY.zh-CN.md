# SightIndex 容器实例运维手册

面向已按 `deploy/containers/README.md` 完成首次部署的实例,记录日常发版、
运维与回退操作。首次部署的准备、镜像构建与验收标准以 README 为准;
本手册是操作速查,不重复其内容。

首次部署的全部步骤(目录、env 生成、release 注册、镜像构建、启动、健康等待)
可由 `deploy/containers/deploy.sh` 一键执行,手工步骤与之等价:

```bash
cd <SightIndex 源码目录>        # 注意:必须是源码目录,不是已部署的根目录
bash deploy/containers/deploy.sh --root /data/sightindex-bj-test --stacks base
# 事后启用更多 stack: reid(需 REID_ENABLED=true + GPU 核对)、embedding、semantic
```

在**已部署的机器**上:`deploy.sh --source` 默认取当前目录,若在部署根目录
(`/data/sightindex-bj-test`)裸跑会被拒绝并提示改用 `manage.sh`。要重新部署
需显式给源码:

```bash
# 日常起停/状态/日志 —— 用 manage.sh,不需要 deploy.sh
bash /data/sightindex-bj-test/manage.sh status

# 用机器上已有的源码 release 重新走部署流程(该 release 无 Dockerfile 时加 --no-build)
bash /data/sightindex-bj-test/deploy.sh \
  --source /data/sightindex-bj-test/releases/20260907-b2f9f25-reviewed-v2 \
  --env-file /data/sightindex-bj-test/.env.semantic-search-v1 \
  --stacks "base reid embedding semantic"
```

## 访问与路径

| 项 | 值 |
| --- | --- |
| 部署根目录 | `/data/sightindex-bj-test` |
| 运维入口 | 根目录下 `manage.sh`(与仓库 `deploy/containers/manage.sh` 保持一致) |
| 环境配置 | 根目录 `.env`(当前生效)与 `.env.<版本>` 快照;权限 `0600` |
| 版本目录 | `releases/<日期-版本>/`,每个是完整源码树,内含 `deploy/containers/` |
| 共享数据 | `media/`、`models/`、`cache/` 与 Compose named volumes,跨版本复用 |
| API | 宿主机 `127.0.0.1:18030`;远程走 SSH 转发或既有 FRP 中继 |
| 凭据 | SSH 与 API Basic Auth 密码只保存在宿主机私密配置与运维笔记中,不写入本文件 |

北京实例(2026-09 现状):SSH 经 FRP 入口登录,公网中继由 `frpc-110` 提供;
入口与中继地址、账号均保存在实例私密运维笔记中,不写入仓库。
完整栈使用 `.env.semantic-search-v1`(含语义检索与 Qwen 嵌入配置)。

## 发新版本

```bash
# 本地:从已审查的提交打包源码
git archive --format=tar.gz -o /tmp/si-<日期-版本>.tar.gz <commit>

# 上传并解压为新 release
scp -P <端口> /tmp/si-<日期-版本>.tar.gz <用户>@<入口>:/data/sightindex-bj-test/
ssh -p <端口> <用户>@<入口>
cd /data/sightindex-bj-test
mkdir -p releases/<日期-版本> && tar -xzf si-<日期-版本>.tar.gz -C releases/<日期-版本>

# 代码有改动时构建新镜像,并在 env 文件中更新 SIGHTINDEX_IMAGE
docker build -f releases/<日期-版本>/deploy/containers/Dockerfile \
  --build-arg SOURCE_REVISION="$(git -C releases/<日期-版本> rev-parse HEAD 2>/dev/null || echo <commit>)" \
  -t sightindex:<新tag> .

# 升级:manage.sh 自动选用最新 release;检测到运行中项目来自旧 release 时
# 先告警,up/restart 将按新 base 重建容器。数据在卷与共享目录中,不受影响。
bash manage.sh --env-file .env.semantic-search-v1 up base reid embedding semantic
bash manage.sh status
```

升级前用 `pg_dump` 备份本实例;保存旧镜像 ID 与旧 release 目录以便回退。

## 日常运维

```bash
bash manage.sh status                      # 容器状态 + API 健康探测
bash manage.sh logs api                    # 跟踪指定服务日志(--tail=100)
bash manage.sh restart                     # 重启所选 stack 的服务
bash manage.sh down                        # 停全套;保留卷,绝不使用 down -v
bash manage.sh --env-file <文件> up <stacks>   # 指定 env / 组合 stack
```

stack 组合:`base`(默认,postgres etcd minio milvus api)、`reid`(需
`REID_ENABLED=true` 且 GPU 容量已核对)、`embedding`(需三个 `QWEN_*` 键)、
`semantic`(需已完成语义索引回填)。语义检索开关与验收见
`deploy/containers/README.md` 的语义章节。

## 回退

```bash
# 指回旧 release;如降级了镜像,同时把 env 中 SIGHTINDEX_IMAGE 改回旧镜像 ID
bash manage.sh --env-file .env.semantic-search-v1 \
  --release releases/<旧版本> up base reid embedding semantic
```

兼容的新增数据库列保留,不整库恢复覆盖新采集数据。GPU 不足时停用 reid
(`REID_ENABLED=false` 后按 base 起),保留 API、数据与其他服务。

## 故障处置记录

- 2026-09-14:api/embedding/reid 三个容器被定向删除(基础设施、镜像、卷、
  compose 网络完好,`down` 未发生过)。恢复命令即上文完整栈的
  `manage.sh --env-file .env.semantic-search-v1 up base reid embedding semantic`,
  恢复后 `media/counts` 与语义索引计数与删除前一致。定位要点:`docker ps -a`
  无容器残留、compose 网络仍挂着基础设施容器、`final-health.json` 提供割接时
  基线。

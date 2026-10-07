# SightIndex 容器实例运维手册

新部署和升级都从已审查源码仓库根目录运行 `bash deploy.sh`，默认选择 containers；
容器的模型准备、配置与验收范围见 [README](README.md)。
`bash deploy.sh --target rtx5090` 是另一条 systemd 安装路径，须按其专用要求执行，
不会在容器部署中自动调用。
本手册的命令是待操作员按实际实例执行的操作说明，本轮文档更新没有连接服务器、
启动容器、下载权重或推送发行。

## 路径与成功标记

| 项 | 默认值或作用 |
| --- | --- |
| 部署根目录 | `/data/sightindex-bj-test`，与源码 checkout 分离 |
| 运维入口 | `ROOT/manage.sh`，部署时复制自本次发行 |
| 私密配置 | 默认 `ROOT/.env`；也可显式指定 `.env.<版本>`，权限 `0600` |
| 发行目录 | `ROOT/releases/<发行名>/`，保存源码、Compose、`SOURCE_REVISION`、`SNAPSHOT_MANIFEST.json` 与 `IMAGE_ID`；清单记录 revision 及快照文件 SHA-256 |
| 成功标记 | `ROOT/active-release` 是已验收发行的绝对路径，`ROOT/active-stacks` 记录所选 stack |
| 失败候选 | 发行目录保留 `.deployment-pending`，不参与默认 newest/overlay 选择 |
| 升级备份 | `ROOT/backups/<本次发行>/`，目录 `0700`、私密配置 `0600` |
| 共享数据 | `media/`、`models/`、`cache/` 与本项目的 Compose named volumes，跨版本保留 |
| API | 默认宿主机 `127.0.0.1:18030`；远程用 SSH 转发，公开访问另配 HTTPS 网关 |

`manage.sh --release` 优先于成功标记；未显式指定时使用 `active-release`。
只有旧安装没有该标记时，才查找最新的非 pending base release。
overlay 先使用目标发行中的文件，再按新旧查其他非 pending 发行，以兼容历史拆分发布。
`active-stacks` 不会自动启用服务，未指定 stack 时仍是 `base`。
成功标记也不会记住另一份 env 的路径；使用自定义配置的实例，日常命令必须继续传入
实际的 `--env-file`，不能误用默认 `.env`。
所有示例使用默认根目录；如果部署使用另一处 `--root`，执行该目录的 `manage.sh` 时
还须设置 `SIGHTINDEX_ROOT` 为那一绝对路径。
更改根目录不会改变固定 Compose 项目名，不能据此并行部署第二个同名实例，详见
[一键部署与验收](../../docs/one-click-deployment.zh-CN.md)。

## 发新版本

准备好的模型目录至少含 CPU YOLO 与 Buffalo-L。所选 ReID 另需 SapiensID checkpoint/config、
pose 和 DFA aligner；所选 embedding 另需完整 Qwen 权重、tokenizer、处理器配置、
模型服务镜像及独立 key。镜像可预建，或选择 `--build-model-services` 从本次应用镜像构建。
当前固定镜像要求 `model.safetensors` 单文件和
`scripts/qwen3_vl_embedding.py`，启动校验固定哈希；额外的分片清单只做路径和完整性检查，
不能替代这两个必需文件。默认不下载资源；通过已审查锁文件显式准备、核验及管理版本的
流程见[模型部署说明](../../docs/model-deployment.zh-CN.md)。

常规路径使用干净 Git checkout，脚本不会 `git pull`。从源码根目录先检查，再部署：

```bash
cd /absolute/path/to/SightIndex
bash deploy.sh --check --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env --stacks "base"
bash deploy.sh --root /data/sightindex-bj-test \
  --env-file /data/sightindex-bj-test/.env --stacks "base"
```

容器入口不要求 `sudo`，当前账号必须具备 Docker 与部署目录访问权限。
`--check` 只读取已有私密配置、模型、工具及所选 GPU stack 的设备状态，不生成 env 或目录，
也不构建、启动、下载或写数据库。`--source /absolute/checkout` 可指定另一份审查过的源码；
默认 source 是部署脚本所在 checkout，不是已部署的根目录。

确需未提交源码时显式传 `--allow-dirty`，发行记录与镜像源码 revision 会标记 dirty。
常规新构建总是更新本次配置的 `SIGHTINDEX_IMAGE`，`--set-image` 已仅用于兼容旧调用。
`--no-build` 复用配置里已经存在的本地应用镜像，不能保证它包含宿主机上的新代码；
仍需通过同样的实际镜像 ID、契约和模型验收。

升级在修改镜像配置与服务前备份输入 env、旧发行路径、已有 stack 记录，并对已有实例
执行 `pg_dump -Fc`。对应备份文件是 `config.env`、`previous-release`、可选
`previous-stacks` 和已有库的 `postgres.dump`；备份错误记录保存在私密目录内。
数据库备份失败或为空就退出。`--skip-backup` 仅供已经验证另行备份的操作使用。

之后脚本注册 pending 发行、构建或复用镜像、静默验证 Compose、启动所选服务，
默认最多等待 600 秒，直到所有所选容器均为 `running healthy`，再执行
`manage.sh verify <stacks> --model-smoke`。它会核对 API 实际 image ID、带认证的数据库计数、
新回放与人脸覆盖契约、只读数据库字段、可选 ReID/Milvus 和 Qwen readiness，以及临时
SQLite/生成视频的上传回放 smoke 和本地 CPU 模型 smoke。
全部通过后才更新 active markers 并移除 pending 标志。

成功只说明所选 stack 通过启动验收。外部 VLM 仅检查配置，不执行付费或远程推理；
索引 pending 和覆盖计数仍需评估，历史回填、真实检索准确率与摄像头采集不在成功范围内。
选择 `semantic` 要同时选择 `embedding`，并在实际私密配置中设
`SEMANTIC_SEARCH_ENABLED=true`；模板默认仍为 `false`。
`REID_ENABLED` 同样必须与 stack 选择一致，不能靠选择 stack 猜测配置已经开启。

## 日常运维

以下例子假定使用默认 `.env` 与基础栈：

```bash
bash /data/sightindex-bj-test/manage.sh status
bash /data/sightindex-bj-test/manage.sh logs api
bash /data/sightindex-bj-test/manage.sh logs --service api
bash /data/sightindex-bj-test/manage.sh config base
bash /data/sightindex-bj-test/manage.sh verify base --model-smoke
bash /data/sightindex-bj-test/manage.sh restart base
bash /data/sightindex-bj-test/manage.sh down
```

`config` 只执行 `compose config --quiet`，不会打印展开的秘密。
`verify` 先比对运行 API image ID 与配置镜像 ID，默认带隔离上传 smoke，
`--model-smoke` 再加载预置 CPU 模型。`status` 的 `/health` 仅表示进程存活。
`logs` 默认跟随 100 行日志，服务名限于 `postgres/etcd/minio/milvus/api/reid/embedding`；
查看 embedding 时同时选择 embedding stack，以包含对应 overlay。

`restart` 仅重启已有容器，不应用新的 env、镜像或 Compose；应用变更使用 `up`。
`up` 不做数据库备份，也不会发布 active markers，新发行正常更新应走 `deploy.sh`。
`down` 停整套项目并保留卷，禁止 `down -v`、全局 prune 或删除共享目录。
未选择某个 stack 不会自动停止其既有容器，不能用 `up base` 假定 GPU 服务已停止。

完整栈需一直显式传入实际 env 与 stack：

```bash
bash /data/sightindex-bj-test/manage.sh \
  --env-file /data/sightindex-bj-test/.env.semantic-search-v1 \
  config base reid embedding semantic
bash /data/sightindex-bj-test/manage.sh \
  --env-file /data/sightindex-bj-test/.env.semantic-search-v1 \
  verify base reid embedding semantic --model-smoke
```

## 人工回退

失败不更新成功标记，也不会自动恢复数据库或容器；如果服务已经变更，API/ReID/embedding
可能还运行候选版本。先查看失败日志、候选 Compose 与 `backups/<失败发行>/`，确认旧发行、
旧镜像 ID、备份配置和原 stack；不要因为默认 marker 仍是旧发行就认为服务已恢复。
失败候选保留 pending 标志供排查，不能手工删标志来假装验收成功。

以下示例只适用于先前已验收的完整栈。把绝对路径替换为实际值；如果原来只有 `base`，
三条管理命令及最后的 stack 标记都改成 `base`。先核对备份 `config.env` 中的
`SIGHTINDEX_IMAGE` 对应旧 `IMAGE_ID`；若旧 tag 已漂移，应在私密配置快照中固定旧 image ID。
过旧镜像没有新 verifier 或契约时，不能宣称通过当前启动验收，须另行审查回退兼容性。

```bash
DEPLOY_ROOT=/data/sightindex-bj-test
OLD_RELEASE='/data/sightindex-bj-test/releases/<已验收旧发行>'
ROLLBACK_BACKUP='/data/sightindex-bj-test/backups/<失败发行>'
ROLLBACK_ENV=/data/sightindex-bj-test/.env
export SIGHTINDEX_ROOT="$DEPLOY_ROOT"

# ROLLBACK_ENV 要设为该实例实际使用的配置路径；先恢复已核对的私密快照。
install -m 0600 "$ROLLBACK_BACKUP/config.env" "$ROLLBACK_ENV" &&
bash "$DEPLOY_ROOT/manage.sh" --env-file "$ROLLBACK_ENV" --release "$OLD_RELEASE" \
  config base reid embedding semantic &&
bash "$DEPLOY_ROOT/manage.sh" --env-file "$ROLLBACK_ENV" --release "$OLD_RELEASE" \
  up base reid embedding semantic &&
bash "$DEPLOY_ROOT/manage.sh" --env-file "$ROLLBACK_ENV" --release "$OLD_RELEASE" \
  verify base reid embedding semantic --model-smoke &&
{
  umask 077
  rollback_release_marker="$(mktemp "$DEPLOY_ROOT/.active-release.rollback.XXXXXX")" &&
  rollback_stacks_marker="$(mktemp "$DEPLOY_ROOT/.active-stacks.rollback.XXXXXX")" &&
  printf '%s\n' "$OLD_RELEASE" > "$rollback_release_marker" &&
  printf '%s\n' 'base reid embedding semantic' > "$rollback_stacks_marker" &&
  mv "$rollback_release_marker" "$DEPLOY_ROOT/active-release" &&
  mv "$rollback_stacks_marker" "$DEPLOY_ROOT/active-stacks"
}
```

这是显式应用回退，不执行 `pg_restore`、不删媒体或模型、不覆盖后来采集的数据，
兼容新增列保留。如果必须恢复数据库，应另行评审备份时间、停写与新数据保留方案。
验收未通过时保持原成功标记，继续排查，不发布新的成功记录。

## 北京历史故障记录（2026-09，非本轮实测）

以下信息保留自已有运维记录，本轮未登录该实例或复测。历史记录中的 SSH 使用 FRP 入口，
公网中继为 `frpc-110`，完整栈使用 `.env.semantic-search-v1`；地址、账号和秘密仍只保存在
实例私密运维笔记中，不写入仓库。

- 2026-09-14：API、embedding、ReID 三个容器被定向删除，基础设施、镜像、卷和 Compose
  网络仍在，未执行过 `down`。当时使用
  `manage.sh --env-file .env.semantic-search-v1 up base reid embedding semantic` 恢复；
  历史记录称恢复后的 `media/counts` 与语义索引计数和删除前一致。
  当时定位依据是 `docker ps -a` 中没有这三个容器、Compose 网络仍挂基础设施容器，
  `final-health.json` 留有割接基线。该旧记录不证明当前源码、模型或新契约已经在北京验收。

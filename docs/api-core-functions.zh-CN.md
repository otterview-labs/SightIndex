# SightIndex 核心接口文档：上传、检索、人脸库

> 面向前端、业务后端和第三方系统对接。仅整理这三个功能，不包含视频流管理、计数、轨迹分析、Chat 等模块。
>
> 核对时间：**2026-09-09 09:52，北京时间**。依据北京已部署服务的 OpenAPI、运行中代码和配置；不是未来设计稿。
>
> 部署版本：`sightindex:bj-semantic-search-v1`  
> API 镜像：`sha256:40631812480c9a42a0cefdf398c47e48c32d083a4d9ac9bba13330df24ecd51b`
>
> 配套文件：同目录 `openapi-core-functions.json`，可导入支持 OpenAPI 3.1 的 Apifox、Postman 或其他接口工具。文中的 UUID、人员和文件示例均为演示数据，不是真实人员记录。

快速导航：[通用约定](#2-通用约定) · [上传](#3-上传接口) · [检索](#4-检索接口) · [人脸库](#5-人脸库接口) · [错误与重试](#6-错误处理与重试) · [页面对接顺序](#8-三个页面的最小实现建议)

## 1. 先看这张对接表

| 功能 | 页面需要做什么 | 核心接口 |
| --- | --- | --- |
| 上传 | 上传图片 | `POST /api/images/upload` |
| 上传 | 对图片执行人员检测，生成人物裁剪 | `POST /api/images/{image_id}/process` |
| 上传 | 上传视频并抽帧、检测人物 | `POST /api/videos/upload` |
| 上传 | 获取原图和人物裁剪 | `GET /api/images/{image_id}`、`GET /api/person-crops` |
| 检索 | 查看语义检索配置和索引覆盖 | `GET /api/search/semantic/status` |
| 检索 | 输入自然语言，检索历史人物图像 | **`POST /api/search/semantic/person-crops`** |
| 检索 | 查询已有结构化标签或已知人员 | `POST /api/search/person-crops` |
| 检索 | 以人物图片找相似历史人物，选接 | `POST /api/reid/search` |
| 人脸库 | 建立人员档案、查询人员 | `POST /api/persons`、`GET /api/persons` |
| 人脸库 | 为人员上传人脸照片 | `POST /api/persons/{person_id}/faces` |
| 人脸库 | 查询人员已录入的人脸 | `GET /api/persons/{person_id}/faces` |
| 人脸库 | 上传查询照片，在已登记人脸中搜索 | `POST /api/face/search` |

**当前必须知道的三个限制：**

1. **上传成功不等于可语义检索。** 北京当前 `VECTOR_INDEX_ON_INGEST=false`；需要完成图片处理并补建人物裁剪索引，才能保证新上传内容进入 Qwen 检索范围。
2. **语义候选不等于标签准确命中，更不等于身份确认。** “红衣男子”可能召回仅部分相似的图片，不能据此自动修改 `person_id`。
3. **人脸库目前不是完整 CRUD。** 已有人员创建、列表、详情、人脸添加和查询；没有人员编辑／删除或单条人脸删除接口。不要按惯例自行拼接不存在的 REST 路径。

## 2. 通用约定

### 2.1 服务地址和鉴权

| 项目 | 约定 |
| --- | --- |
| Base URL | 使用管理员提供、调用方网络实际可达的服务地址 |
| 北京内网部署地址 | `http://192.168.1.40:18030`；不代表公网可达 |
| API 路径 | 以 `/api` 开头 |
| 图片／视频资源 | 返回 `/data/...` 相对路径，与 Base URL 拼接 |
| 健康检查 | `GET /health`，不要求认证 |
| 在线说明 | `/docs`；完整规范 `/openapi.json`，当前也需要认证 |
| 鉴权 | **HTTP Basic Auth**，账号密码由管理员单独提供 |
| 成功状态码 | 本文业务接口成功通常为 **200**，创建接口也不是 201 |
| JSON 编码 | UTF-8，`Content-Type: application/json` |
| 文件上传 | `multipart/form-data`，文件字段名固定为 **`file`** |

本文不包含登录凭据。Basic Auth 不是加密，应通过可信内网或 HTTPS 网关访问；对外产品建议由业务后端代理，不要把共享服务密码写入前端源码。

示例命令在 Bash／Zsh 中执行。先填写地址，再交互输入凭据：

```bash
export SI_BASE_URL='http://192.168.1.40:18030'
printf 'SightIndex 用户名：'
read -r SI_USER
printf 'SightIndex 密码：'
read -rs SI_PASSWORD
printf '\n'
export SI_USER SI_PASSWORD

curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  "$SI_BASE_URL/api/search/semantic/status"
```

请求图片资源也需要认证。`image_url`、`crop_url` 不是公开免鉴权链接：

```bash
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  "$SI_BASE_URL/data/crops/example.jpg" \
  -o /tmp/example.jpg
```

跨域页面直接用 `<img src="...">` 不方便附加自定义认证头，宜走同源代理或认证下载后创建 Blob URL；不要把账号密码嵌入图片 URL。

### 2.2 ID、时间和空值

- `image_id`：原图或视频抽帧图片 ID。
- `crop_id`：人物裁剪 ID，一张原图可产生多个裁剪。
- `person_id`：人员档案 ID；裁剪可能没有归属，值为 `null`。
- `face_embedding_id`：一条已登记人脸特征 ID；**不是** `person_id`。
- ID 类型为 UUID 字符串，不是自增整数。
- 时间传 ISO 8601，建议使用 UTC：`2026-09-09T01:00:00Z`。带 `+08:00` 的时间放在 URL 时，`+` 应编码为 `%2B`。
- `captured_at` 表示实际拍摄时间，`created_at` 表示系统写入时间，二者不能混用。
- 原图未传 `captured_at` 时可为 `null`；继承该时间的裁剪不会通过要求具体拍摄时间的筛选。
- JSON 中不用的可选参数可以省略；查询字符串不要把 `null` 当字面字符串传入。
- 数组接口并不统一带 `items` 外壳：图片／裁剪／人员列表返回裸数组，检索接口返回对象。以每节说明为准。

### 2.3 推荐的最小调用链

```text
图片上传：
上传图片 → 得到 image_id → 处理图片 → 得到 crop_id[]
                                         ↓
                             后端补建人物裁剪索引
                                         ↓
                              Qwen 自然语言检索

视频上传：
上传视频并处理 → 得到 image_ids[]、crop_ids[]
                              ↓
                    后端补建人物裁剪索引
                              ↓
                     Qwen 自然语言检索

人脸库：
创建人员 → 得到 person_id → 给该人员添加人脸照片 → 在已登记人脸中搜索
```

入库流程与查询流程应分别实现。不要把“将图片作为查询条件”错误地做成“给某个人员录入该图片”。

---

## 3. 上传接口

### 3.1 上传图片

**`POST /api/images/upload`**

| 参数 | 位置 | 类型 | 必填 | 默认／说明 |
| --- | --- | --- | --- | --- |
| `file` | multipart 文件 | binary | 是 | 一张图片 |
| `source_type` | query | string | 否 | `upload`；普通上传沿用默认值 |
| `camera_id` | query | UUID | 否 | 来源摄像头 ID |
| `location_id` | query | UUID | 否 | 来源位置 ID |
| `captured_at` | query | datetime | 否 | 实际拍摄时间；建议传入 |

**注意：除 `file` 外，上述参数都放在 URL query，不是 multipart 文本字段。**

```bash
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  -F 'file=@/path/to/photo.jpg;type=image/jpeg' \
  "$SI_BASE_URL/api/images/upload?source_type=upload&captured_at=2026-09-09T01%3A00%3A00Z"
```

成功响应：`ImageRead`，示例：

```json
{
  "id": "22222222-2222-4222-8222-222222222222",
  "image_url": "/data/uploads/example-photo.jpg",
  "thumbnail_url": null,
  "source_type": "upload",
  "camera_id": null,
  "location_id": null,
  "location_name": null,
  "captured_at": "2026-09-09T01:00:00Z",
  "created_at": "2026-09-09T01:00:01Z"
}
```

该接口保存原图和元数据，**不会仅因上传就产生人物裁剪**。下一步调用图片处理接口。

### 3.2 处理已上传图片

**`POST /api/images/{image_id}/process`**

参数只有 path 中的 `image_id`，无需请求体。

```bash
export IMAGE_ID='22222222-2222-4222-8222-222222222222'
curl -sS --fail-with-body -X POST \
  -u "$SI_USER:$SI_PASSWORD" \
  "$SI_BASE_URL/api/images/$IMAGE_ID/process"
```

成功响应：**`PersonCropRead[]` 裸数组**。

```json
[
  {
    "id": "33333333-3333-4333-8333-333333333333",
    "image_id": "22222222-2222-4222-8222-222222222222",
    "crop_url": "/data/crops/example-crop.jpg",
    "bbox": {
      "x": 120,
      "y": 50,
      "width": 180,
      "height": 420,
      "confidence": 0.91,
      "label": "person"
    },
    "attributes": null,
    "person_id": null,
    "camera_id": null,
    "location_id": null,
    "captured_at": "2026-09-09T01:00:00Z",
    "created_at": "2026-09-09T01:00:02Z"
  }
]
```

- `bbox` 是检测框信息，`x/y/width/height` 使用原图像素坐标；实际返回可能包含原图／裁剪尺寸等额外字段。
- 无符合质量条件的人物时可能返回 `[]`；当前实现原图文件缺失时也可能返回 `[]`，因此不能把所有空数组都解释为“图中无人”。
- 图片 ID 不存在返回 404；索引队列压力可返回 503。
- 当前北京版本该接口**不是幂等处理接口**：重复执行可能生成新的重复裁剪。请求超时后应先按 `image_id` 查询已有裁剪，不要无条件重试。
- 人脸识别、属性解析、索引入库受服务配置控制，200 不表示所有可选能力都已成功。

### 3.3 上传并处理视频

**`POST /api/videos/upload`**

视频保存、抽帧与人物检测在同一次请求内执行，返回处理结果，**不是返回异步 `task_id` 的任务接口**。

| 参数 | 位置 | 类型 | 默认 | 范围／说明 |
| --- | --- | --- | --- | --- |
| `file` | multipart 文件 | binary | 无 | 必填，视频文件 |
| `frame_interval_seconds` | query | number | `1.0` | `0.1～3600`，抽样时间间隔 |
| `max_frames` | query | integer | `120` | `1～2000`，最多抽样帧数，不是总视频帧数 |
| `store_empty_frames` | query | boolean | 跟随配置 | 是否保存未检出人物的抽样帧 |
| `camera_id` | query | UUID | 无 | 来源摄像头 |
| `location_id` | query | UUID | 无 | 来源位置 |
| `captured_at` | query | datetime | 无 | 视频起始拍摄时间；未传时按服务当前时间作为基准 |

计数线参数 `line_x1/line_y1/line_x2/line_y2` 为可选 query 参数，范围 `0～1`；只做这三个功能时无需传入。

```bash
curl -sS --fail-with-body --max-time 600 \
  -u "$SI_USER:$SI_PASSWORD" \
  -F 'file=@/path/to/video.mp4;type=video/mp4' \
  "$SI_BASE_URL/api/videos/upload?frame_interval_seconds=1&max_frames=120&store_empty_frames=false&captured_at=2026-09-09T01%3A00%3A00Z"
```

成功响应：`VideoProcessResponse`。计数字段仅为结构示例，不表示完整时长已处理。

```json
{
  "video_url": "/data/videos/example-video.mp4",
  "frame_interval_seconds": 1.0,
  "frames_read": 300,
  "frames_sampled": 10,
  "images_created": 1,
  "crops_created": 1,
  "counting_events_created": 0,
  "image_ids": ["22222222-2222-4222-8222-222222222222"],
  "crop_ids": ["33333333-3333-4333-8333-333333333333"]
}
```

- 兼容格式取决于服务器解码器；对接时先验证实际使用的编码和封装，不把任意 MP4 都视为可解码。
- 达到 `max_frames` 就会结束抽样，不等于处理了整段视频。业务端应保存返回的 ID 清单。
- 长视频可能需要更长 HTTP 超时；文中 `600` 秒是客户端示例，不是服务 SLA。
- 503 队列压力响应可能有 `detail.partial=true`、`image_ids`、`crop_ids`、`images_committed` 等已提交进度。**部分帧可能已经入库，不应直接重传整个视频。**
- 当前未提供断点续传、按 `task_id` 查询进度或视频片段检索能力。

### 3.4 获取图片与人物裁剪

| 方法与路径 | 参数 | 返回 |
| --- | --- | --- |
| `GET /api/images` | query：`limit=50`，范围 `1～200`；`offset=0`，≥0；`has_crops=false` | `ImageRead[]` |
| `GET /api/images/{image_id}` | path：UUID | `ImageRead`；不存在为 404 |
| `GET /api/person-crops` | query：可选 `image_id`；`limit=50`，范围 `1～200`；`offset=0`，≥0 | `PersonCropRead[]` |
| `GET /api/person-crops/{crop_id}` | path：UUID | `PersonCropRead`；不存在为 404 |
| `GET /api/media/counts` | 无 | `{"image_with_crops_count":4,"person_crop_count":23}` |

`image_with_crops_count` 只统计有裁剪的图片，**不是所有上传图片总数**。

```bash
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  "$SI_BASE_URL/api/person-crops?image_id=$IMAGE_ID&limit=200&offset=0"
```

### 3.5 上传后的索引步骤：由业务后端／运维调用

**`POST /api/search/index/rebuild`**

| 参数 | 位置 | 默认 | 说明 |
| --- | --- | --- | --- |
| `target` | query | `person_crop` | 仅 `image` 或 `person_crop` |
| `limit` | query | `500` | `1～10000` |

```bash
curl -sS --fail-with-body --max-time 600 -X POST \
  -u "$SI_USER:$SI_PASSWORD" \
  "$SI_BASE_URL/api/search/index/rebuild?target=person_crop&limit=500"
```

```json
{
  "target": "person_crop",
  "requested": 500,
  "seen": 23,
  "indexed": 23,
  "errors": []
}
```

**必须检查返回体，而不只看 HTTP 200：**

- `errors` 应为空，`indexed` 应与本次 `seen` 相符。
- 该接口按创建时间倒序处理最近 `limit` 条记录，**不是针对刚上传某一张图片的单条索引 API**。
- 没有 `offset`、`image_id` 或 `crop_id` 参数；重复调用相同 limit 不会自动推进到下一页。
- 目前 23 个裁剪可以一次覆盖；大量历史数据不能把默认 `limit=500` 当成全量回填保证。
- 建议合并批次执行，不能让每个浏览器上传一次就并发发起一次全库重建。此接口会占用模型推理和索引资源。
- 本文“业务后端／运维”是调用职责建议，**当前共享 Basic Auth 不会在 API 层自动区分管理员和普通用户**。

---

## 4. 检索接口

### 4.1 检索状态与覆盖

**`GET /api/search/semantic/status`**

北京当前响应快照：

```json
{
  "enabled": true,
  "configured": true,
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "min_score": 0.25,
  "total_crops": 23,
  "indexed_crops": 23,
  "labeled_crops": 0,
  "attributes_enabled": false,
  "auto_index_on_ingest": false
}
```

| 字段 | 含义 |
| --- | --- |
| `enabled` | 语义入口功能开关 |
| `configured` | Qwen provider、独立视觉集合等配置是否符合要求；不是模型实时健康检查 |
| `model` | 当前视觉 embedding 模型 |
| `min_score` | 候选相似度门槛，不是置信概率 |
| `total_crops` | 数据库人物裁剪总数 |
| `indexed_crops` | 当前模型和维度对应的 **SQL 索引标记覆盖数**，不保证远端向量实时可用 |
| `labeled_crops` | 有 `attributes.source` 来源标记的裁剪数，不是所有字段完整解析的数量 |
| `attributes_enabled` | 结构化属性模型配置是否启用 |
| `auto_index_on_ingest` | 新入库内容是否配置为自动建立视觉索引 |

前端建议：`enabled && configured` 为真时默认语义模式；否则保留严格标签模式并解释原因。显示覆盖数，不要只显示“模型已部署”。

### 4.2 自然语言检索：主入口

**`POST /api/search/semantic/person-crops`**

JSON 请求：

```json
{
  "query": "穿红衣服的男子",
  "top_k": 20,
  "filters": {}
}
```

| 字段 | 类型 | 必填／默认 | 说明 |
| --- | --- | --- | --- |
| `query` | string | 必填 | 去除首尾空格后长 `1～2048`；空白查询拒绝 |
| `top_k` | integer | 默认 `20` | `1～100`，最终最多展示的去重候选数 |
| `filters` | object | 默认 `{}` | 可选的元数据筛选条件 |

`filters` 可用字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `camera_id` | UUID | 按裁剪的摄像头过滤 |
| `location_id` | UUID | 按裁剪的位置过滤 |
| `person_id` | UUID | 按数据库已有人员关联过滤；不触发新的身份识别 |
| `start_time` | datetime | 拍摄时间下界，包含边界 |
| `end_time` | datetime | 拍摄时间上界，包含边界 |
| `extra` | object | 当前语义入口不支持非空值；不要在此传帽子／颜色等标签 |

条件组合为 AND。时间未知的裁剪不通过时间条件；开始时间晚于结束时间返回 400。相机等条件在 Milvus 召回前限制对象范围，不是全库 Top-K 之后才过滤。

```bash
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  -H 'Content-Type: application/json' \
  -d '{"query":"穿红衣服的男子","top_k":20,"filters":{}}' \
  "$SI_BASE_URL/api/search/semantic/person-crops"
```

成功响应示例：

```json
{
  "mode": "semantic",
  "items": [
    {
      "crop_id": "33333333-3333-4333-8333-333333333333",
      "image_id": "22222222-2222-4222-8222-222222222222",
      "image_url": "/data/uploads/example-photo.jpg",
      "crop_url": "/data/crops/example-crop.jpg",
      "score": 0.369661,
      "captured_at": "2026-09-09T01:00:00Z",
      "person_id": null,
      "person_name": null,
      "attributes": null,
      "match_type": "semantic_candidate",
      "duplicate_crop_ids": []
    }
  ],
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "min_score": 0.25,
  "notice": "以下仅为视觉语义相似候选，颜色、性别、帽子等条件未逐项核验，相似度不是准确率，也不能确认人物身份。同内容、同相机且同身份标注的裁剪已合并。",
  "scope_crops": 23,
  "indexed_scope_crops": 23,
  "candidates_examined": 23,
  "shortlist_limited": false
}
```

示例省略了部分值为 `null` 的结果项字段，完整字段以配套 OpenAPI 为准。

| 响应字段 | 对接要求 |
| --- | --- |
| `items` | 服务器已排序的候选，保持顺序展示 |
| `score` | 视觉向量相似度，不显示为“身份准确率 36.9%” |
| `match_type` | 固定 `semantic_candidate`，据此显示“语义候选，待核验” |
| `duplicate_crop_ids` | 被合并的其他裁剪 ID；数量不含当前展示的主裁剪 |
| `notice` | 展示给用户；可能追加覆盖不足或候选截断说明 |
| `scope_crops` | 应用元数据筛选后的裁剪总数 |
| `indexed_scope_crops` | 筛选范围内有当前模型索引标记的裁剪数 |
| `candidates_examined` | 本次从向量库取得、进入后续核对的候选数量 |
| `shortlist_limited` | 是否只检查了范围内的有限相似候选 |

行为与错误：

- 候选可有 `attributes=null`；语义检索不依赖预先写好的红衣／背包标签。
- 同文件内容、同相机、同位置、同人员关联的裁剪合并，不合并不同已标注身份；原始记录不删。
- 当前最多允许检索范围中有 10000 个已索引对象，超出时返回 400 并要求缩小筛选。
- 每次最多核对 400 个候选；不足 `top_k` 不代表没有更多历史记录。
- 200 且 `items=[]`：筛选范围为空，或没有达到门槛且文件可读取的候选；不证明目标一定不存在。
- 503：功能未启用、配置不满足要求、范围内无当前模型索引、向量未就绪或模型／索引服务故障。
- **前端必须把 503 显示成错误，不得替换成“未找到目标”，也不得自动补入最近图片。**

### 4.3 严格标签／已知人员检索：兼容入口

**`POST /api/search/person-crops`**

```json
{
  "query": "红衣戴帽的人",
  "top_k": 20,
  "filters": {},
  "rerank": false
}
```

- `query`：字符串；业务端应拦截空白查询。
- `top_k`：默认 20，范围 `1～100`。
- `filters`：可使用相机、位置、人员和时间条件；不要依赖任意 `extra` 字段，当前没有通用标签表达式合同。
- `target`：兼容 schema 中存在，但该路由固定覆盖为 `person_crop`，无需传。
- `rerank`：建议保持 `false`；不是启用 Qwen 召回的开关。
- 响应为 `{"items":[...]}`，单项使用 `SearchResultItem`；无匹配返回 `{"items":[]}`。
- 该入口保留已知人员匹配、已解析结构化条件、观察表／标签匹配逻辑，**不调用语义向量兜底**。
- 解析出的多标签条件严格 AND；但这不代表自然语言的每个词都已被理解。例如当前“穿红衣服的男子”只解析出 `upper_color=red`，不能声称该入口还核验了男性。
- 北京快照中结构化属性覆盖为 `0/23`，因此服装标签查询目前为空是预期行为；需要搜索这些历史图像时使用上一节语义入口。

### 4.4 以图找历史人物：可选接入

若“检索”页面还需要上传一张人物图片找相似历史人物，可选接 **ReID**；它与“上传原图入库”和“在人脸库中找登记人员”不是同一件事。

| 方法与路径 | 参数 | 说明 |
| --- | --- | --- |
| `GET /api/reid/status` | 无 | 检查 `ready`、`indexed_crops`、`pending_crops` 和服务状态 |
| `POST /api/reid/search` | multipart：`file` 必填；query：`top_k=0`，范围 `0～200` | 上传人物图作为查询；0 表示采用服务配置默认值 |
| `POST /api/reid/crops/{crop_id}/similar` | path：UUID；query：`top_k=0`，范围 `0～200` | 用已有裁剪检索相似人物，无请求体 |

```bash
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  -F 'file=@/path/to/person-crop.jpg;type=image/jpeg' \
  "$SI_BASE_URL/api/reid/search?top_k=20"
```

返回 `ReidSearchResponse`，主要字段：

| 字段 | 说明 |
| --- | --- |
| `items` | 历史人物裁剪候选，保持服务器排序，不重新按原始 `score` 排序 |
| `items[].crop_id/crop_url/image_id/image_url` | 展示和查看原图所需字段 |
| `items[].score` | 人体特征相似度 |
| `items[].face_similarity/face_match/face_reliability` | 可选人脸辅助证据；`null` 不表示验证失败 |
| `items[].fusion_score/evidence_level/decision_reason` | 服务器的排序／证据说明 |
| `model/min_score` | 模型与该检索空间的阈值，不与 Qwen 分数直接比较 |
| `query_mode/query_frame_count` | 单帧或邻近帧查询方式 |
| `face_coverage` | 本次查询实际尝试的人脸工作覆盖，不代表全面人脸确认 |

当前上传 ReID 查询使用临时文件，不作为普通上传图片登记；部分既有裁剪查询可能补充派生缓存／属性，不能把所有查询都视为绝无副作用。返回仍是候选，不应自动入人脸库或关联人员。

当前 ReID 这两个接口没有相机／时间请求参数；不要把语义接口的 `filters` 参数照搬过来。

### 4.5 不要接入这些占位接口

以下路径虽然可能出现在完整 OpenAPI 中，但当前不提供有效业务召回，本精简规范不包含它们：

- `POST /api/search/images`
- `POST /api/search/videos`
- `POST /api/search/by-image`
- `GET /api/videos/{video_id}/clips`

“以图找历史人物”用 ReID，“查登记人员的人脸”用 `/api/face/search`，不要误用占位 `/api/search/by-image`。

---

## 5. 人脸库接口

### 5.1 创建人员档案

**`POST /api/persons`**

| JSON 字段 | 类型 | 必填 | 默认／说明 |
| --- | --- | --- | --- |
| `name` | string | 是 | 姓名；业务端应校验非空 |
| `employee_no` | string 或 null | 否 | 工号 |
| `phone` | string 或 null | 否 | 电话，按业务必要性填写 |
| `department` | string 或 null | 否 | 部门 |
| `avatar_url` | string 或 null | 否 | 头像路径；首次录入人脸时可能自动补齐 |
| `status` | string | 否 | 默认 `active`；不是已定义的启停生命周期接口 |

```bash
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  -H 'Content-Type: application/json' \
  -d '{"name":"示例人员","employee_no":"DEMO-001","department":"测试部门"}' \
  "$SI_BASE_URL/api/persons"
```

成功响应：`PersonRead`。

```json
{
  "id": "11111111-1111-4111-8111-111111111111",
  "name": "示例人员",
  "employee_no": "DEMO-001",
  "phone": null,
  "department": "测试部门",
  "avatar_url": null,
  "status": "active",
  "created_at": "2026-09-09T01:00:00Z",
  "updated_at": "2026-09-09T01:00:00Z"
}
```

创建人员不等于已录入人脸。保存响应 `id`，随后调用添加人脸接口。当前没有幂等键合同，不应把重复姓名或重复请求自动视为同一人员。

### 5.2 查询人员

| 方法与路径 | 参数 | 返回 |
| --- | --- | --- |
| `GET /api/persons` | query：可选 `query`，按姓名模糊匹配；`limit=50`，范围 `1～200` | `PersonRead[]` |
| `GET /api/persons/{person_id}` | path：UUID | `PersonRead`，不存在为 404 |

人员列表**没有 `offset`／分页游标**，`query` 也不是工号或电话号码通用搜索。超过 200 人的完整翻页浏览需要另行补接口，不应声称当前已支持。

```bash
curl -sS --fail-with-body -G \
  -u "$SI_USER:$SI_PASSWORD" \
  --data-urlencode 'query=示例' \
  --data-urlencode 'limit=50' \
  "$SI_BASE_URL/api/persons"
```

### 5.3 为人员上传人脸照片

**`POST /api/persons/{person_id}/faces`**

path：`person_id` 必填；multipart：`file` 必填，无其他请求参数。

```bash
export PERSON_ID='11111111-1111-4111-8111-111111111111'
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  -F 'file=@/path/to/face.jpg;type=image/jpeg' \
  "$SI_BASE_URL/api/persons/$PERSON_ID/faces"
```

成功响应：`FaceEmbeddingRead`。

```json
{
  "id": "44444444-4444-4444-8444-444444444444",
  "person_id": "11111111-1111-4111-8111-111111111111",
  "image_id": "22222222-2222-4222-8222-222222222222",
  "crop_id": null,
  "face_bbox": {"x": 30, "y": 20, "width": 120, "height": 140},
  "face_model": "insightface-buffalo_l",
  "quality_score": 0.92,
  "created_at": "2026-09-09T01:00:03Z"
}
```

- 会保存照片、创建图片记录和人脸特征记录；未设置人员头像时可补齐头像。
- 返回的是特征**元数据**，不含原始 embedding 数组。
- 应上传清晰、单人、面部无遮挡的照片。当前上传接口会选取质量最高的人脸，不提供请求参数指定画面中哪一个人。
- 当前北京 `face_fallback_to_full_image=false`，不把整张图片兜底当作人脸。
- 人员不存在为 404；无可读取人脸等情况为 400。
- 同一人员可以录入多张照片；重复提交可能创建多条人脸记录，不保证自动去重。
- `quality_score` 是质量信号，不是身份正确概率。

### 5.4 从已有裁剪录入人脸：需显式人工确认

**`POST /api/persons/{person_id}/faces/from-crop/{crop_id}`**

两个 path 参数均为 UUID，无请求体，返回 `FaceEmbeddingRead`。

```bash
export CROP_ID='33333333-3333-4333-8333-333333333333'
curl -sS --fail-with-body -X POST \
  -u "$SI_USER:$SI_PASSWORD" \
  "$SI_BASE_URL/api/persons/$PERSON_ID/faces/from-crop/$CROP_ID"
```

**这不是普通查询，而是身份写操作：**

- 在裁剪中严格提取可用人脸；多人、模糊或无可用人脸可能拒绝。
- 成功后新增人脸特征，并把裁剪的 `person_id` 设为指定人员、来源设为人工标注。
- 会创建或更新关联识别事件、更新观察表，可能设置头像。
- 当前该路径未像独立裁剪标注接口一样先检查其他人员绑定冲突；调用前应读取裁剪并要求明确确认，避免改写已有人员关联。
- 不能在用户点击一个“语义相似候选”时自动调用。

### 5.5 查询人员已录入的人脸

**`GET /api/persons/{person_id}/faces`**

path：`person_id`；query：`limit=50`，范围 `1～200`。返回 **`FaceEmbeddingRead[]` 裸数组**。

```bash
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  "$SI_BASE_URL/api/persons/$PERSON_ID/faces?limit=50"
```

人脸记录本身没有直接的照片 URL。展示缩略图／照片时：

1. 有 `crop_id`：调用 `GET /api/person-crops/{crop_id}`，取 `crop_url`。
2. 否则有 `image_id`：调用 `GET /api/images/{image_id}`，取 `image_url`。
3. 与 Base URL 拼接后通过认证读取。

该列表同样没有 `offset`。

### 5.6 上传照片，在登记人脸库中搜索

**`POST /api/face/search`**

| 参数 | 位置 | 默认 | 范围／说明 |
| --- | --- | --- | --- |
| `file` | multipart 文件 | 无 | 必填，查询照片 |
| `top_k` | query | `10` | `1～100` |
| `min_similarity` | query | 无 | 可选 `0～1`；不传不表示自动采用识别阈值 |

```bash
curl -sS --fail-with-body \
  -u "$SI_USER:$SI_PASSWORD" \
  -F 'file=@/path/to/query-face.jpg;type=image/jpeg' \
  "$SI_BASE_URL/api/face/search?top_k=10&min_similarity=0.45"
```

成功响应：`FaceSearchResponse`。

```json
{
  "image_id": "77777777-7777-4777-8777-777777777777",
  "face_bbox": {"x": 30, "y": 20, "width": 120, "height": 140},
  "matches": [
    {
      "person_id": "11111111-1111-4111-8111-111111111111",
      "person_name": "示例人员",
      "face_embedding_id": "44444444-4444-4444-8444-444444444444",
      "similarity": 0.71,
      "quality_score": 0.92,
      "image_id": "22222222-2222-4222-8222-222222222222",
      "crop_id": null
    }
  ]
}
```

重要区别：

- 顶层 `image_id` 是**本次查询照片**；`matches[].image_id/crop_id` 指向**库中被匹配的人脸来源**。
- 返回按人脸特征记录匹配，不保证每个人只出现一次；同一人多张脸可能产生多条候选。
- 无人脸时可返回 `face_bbox=null`、`matches=[]`；无匹配也可返回空数组。
- 此接口会保存查询照片及图片记录，但不登记为某个人的人脸，不创建该识别查询的 `RecognitionEvent`。它不是纯内存、不落盘查询。
- `min_similarity=0.45` 是示例候选门槛，不是准确率保证。

### 5.7 如需记录识别结果

**`POST /api/face/recognize`**

仅当业务确实要保存识别事件时选接；只浏览人脸候选优先用上一节 `/face/search`。

| 参数 | 位置 | 默认 | 范围／说明 |
| --- | --- | --- | --- |
| `file` | multipart 文件 | 无 | 必填 |
| `top_k` | query | `5` | `1～50` |
| `threshold` | query | 服务配置 | 可选 `0～1`；北京当前配置为 `0.45` |

返回 `FaceRecognitionResponse`：

- `result_type`：当前可能是 `known`、`unknown` 或 `no_face`。
- `person`：达到识别阈值时的人员档案，否则为 `null`。
- `similarity`、`threshold`：相似度和采用的阈值。
- `image_id`：保存的查询图。
- `event_id`：新增的识别事件。
- `face_bbox`、`matches`：检测人脸和候选清单。

`known` 是算法按阈值给出的分类，不等于人工确认。本接口会保存查询照片并创建识别事件，重试可能新增事件。

### 5.8 当前没有的管理能力

下列接口**未实现，不能调用**：

```text
PATCH /api/persons/{person_id}
PUT   /api/persons/{person_id}
DELETE /api/persons/{person_id}
DELETE /api/persons/{person_id}/faces/{face_id}
```

已有的 `DELETE /api/persons/{person_id}/crops/{crop_id}` 只解除人物裁剪的人员标注，**不是删除人员或删除人脸**，不纳入本次人脸库最小合同。

如果页面要求人员编辑、撤销同意后删除资料、误录人脸删除、批量导入或完整分页，需另行补充接口和数据删除规则。不能通过前端隐藏记录代替实际删除。

---

## 6. 错误处理与重试

| 状态／表现 | 含义和处理 |
| --- | --- |
| 200 | HTTP 调用成功；仍要检查 `items`、`errors`、`result_type`、覆盖数等业务结果 |
| 400 | 业务参数无效、无法解码／提取人脸、时间范围错误、语义范围过大等 |
| 401 | Basic Auth 缺失或错误；当前可返回纯文本 `Authentication required`，不是统一 JSON |
| 404 | 图片、裁剪或人员 ID 不存在 |
| 422 | FastAPI 参数校验失败，例如 UUID 格式错误、漏传文件、字段越界 |
| 503 | 语义模型／索引不可用、未建索引、处理队列压力等；不要当成空匹配 |
| 500／连接超时 | 异常或未完成响应；写入类接口可能已有副作用，先核对再重试 |

常见 JSON 错误：

```json
{"detail": "语义检索服务暂不可用，请稍后重试"}
```

`detail` 也可能是对象或校验错误数组，不保证总是字符串。

前端应先判断 HTTP 状态，再尝试解析 JSON；对纯文本保留可理解的错误信息。不要吞掉 401、503 后直接展示“没有找到”。

**写入类接口不要自动无限重试：**图片上传、图片处理、视频上传、人员创建、人脸录入和人脸识别都不提供通用 `Idempotency-Key` 合同。

## 7. 上传与部署边界

- 当前北京业务 API 的运行版本仍使用直接保存上传文件的实现；**没有已核实生效的统一图片／视频字节上限和像素限制合同**。
- 不要把本地其他未发布修改中的 `20 MB` 图片、`128 MB` 视频等限制写成线上保证；也不能因此认为可以无限上传。
- 建议调用方限制格式／体积并在受控网关实施配额、内容校验和访问控制；在完成相应服务端验证前，不将此测试部署作为公开匿名上传服务。
- 新增数据自动 Qwen 索引关闭、属性模型未启用是真实配置状态，不是文档步骤可以自动改变的设置。
- 人脸库照片和识别查询图会落盘，应由部署方明确留存和删除流程；当前缺少删除 API，需要在正式业务接入前评估这一缺口。
- 本文仅承诺列出的已部署接口行为，不承诺“有 OpenAPI 路径就功能完整”。

## 8. 三个页面的最小实现建议

### 上传页

1. 图片上传返回 `image_id` 后，调用一次 `/images/{image_id}/process`。
2. 显示裁剪缩略图、处理结果和索引状态；“已上传”“已检测”“已索引”分开显示。
3. 视频直接展示 `/videos/upload` 返回的帧／裁剪计数和部分失败状态。
4. 索引回填放到业务后端管理，不让前端无控制地并发全库重建。

### 检索页

1. 加载 `/search/semantic/status`，选择可用的模式。
2. 默认 Qwen 语义检索；保留严格标签模式，明确属性覆盖不足时为何为空。
3. 使用返回的排序、`notice` 和重复合并信息。
4. 可选接入 ReID 以图找人；不要把图文相似、人体相似、人脸相似混成一个分数。
5. 错误态与正常空态分开；查看图片资源时保留认证。

### 人脸库页

1. 查询／创建人员，进入人员详情。
2. 上传单人人脸照片，展示该人员的人脸列表与质量信息。
3. 如需以已有裁剪入库，增加身份确认步骤。
4. 用 `/face/search` 做“查询登记人员”；需要留识别事件时才用 `/face/recognize`。
5. 暂不展示无法执行的编辑／删除按钮，或明确标记待实现。

## 9. 配套 OpenAPI 和核对说明

`openapi-core-functions.json` 只保留本文的上传、检索、人脸库接口，以及健康检查和必要索引维护接口；排除占位检索、视频流、Chat、轨迹和计数模块。

该文件从上述部署版本的真实 OpenAPI 裁剪：

- 保留原始请求参数、默认值、校验范围、成功响应 schema 和 operationId。
- 补充中文分类／摘要和本文核实过的语义说明。
- 将实际中间件使用的 HTTP Basic 补充为安全方案；该认证在原生 OpenAPI 中没有自动完整表达。
- 通过文档注释补充已核实的错误响应；不改变线上真实 API。
- `servers` 使用可配置 Base URL 示例，导入后须替换成调用方实际可达地址并单独设置鉴权。
- 不包含密码、API key、真实人员姓名、原始媒体或 embedding 向量。

**验证范围：**本次编写文档读取线上 schema／源码／状态，没有为了写示例重复上传媒体、创建人员或录入人脸。语义检索、去重、过滤及错误恢复已有前一次北京真实浏览器验收；本文的演示 JSON 是合同示例，不是新增写接口实测记录。

文档校验结果：精简规范含 **22 条路径、24 个操作、31 个关联 schema**；相较线上源规范，请求参数和成功响应 schema 差异为 0，悬空 `$ref` 为 0；12 个 JSON 示例通过类型校验，15 个 Shell 示例通过语法检查，OpenAPI TypeScript 导入生成检查通过。未实际执行文档中的写入示例。

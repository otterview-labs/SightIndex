# 首次体验：上传、处理和查看图片

[English](first-run.md) · [简体中文](first-run.zh-CN.md) · [项目首页](../README.zh-CN.md)

本文使用默认本地配置，验证素材接入与处理，不启用外部模型服务。这条流程不用于展示语义检索效果或确认人员身份。

## 启动服务

按[首页快速开始](../README.zh-CN.md#快速开始)启动，打开 `http://127.0.0.1:8000/`。
默认 `.env.example` 使用 SQLite 与 OpenCV HOG 人员检测器，未启用 Milvus、VLM、语义检索或 ReID。

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/api/media/counts
```

统计接口返回“含裁剪图的图片数量”和“人员裁剪数量”，不是所有上传图片的数量。新环境返回零是正常情况。

## 上传、处理和查看结果

准备一张允许处理的 JPEG 或 PNG；体验人员检测时，使用能清楚看见人员的图片。仓库目前没有随附的公开样例数据集。

1. 在控制台上传图片，或使用下面的接口。
2. 保存响应中的 `id`。上传创建素材记录，人员检测需要另外执行。
3. 处理该图片，查看返回的人员裁剪列表。
4. 回到控制台或查询裁剪列表，对照原始画面。

```bash
curl --fail -X POST http://127.0.0.1:8000/api/images/upload \
  -F 'file=@/path/to/your-image.jpg'
```

将 `IMAGE_ID` 替换为上传响应返回的 UUID：

```bash
curl --fail -X POST http://127.0.0.1:8000/api/images/IMAGE_ID/process
curl --fail 'http://127.0.0.1:8000/api/person-crops?image_id=IMAGE_ID'
curl --fail http://127.0.0.1:8000/api/images/IMAGE_ID
```

路径与 ID 是占位符。若启用了 Basic Auth，请使用自己的配置账号认证接口与图片资源请求。

裁剪列表为空表示本次检测没有产生人员裁剪，不能据此判断上传是否失败。可以单独查询图片记录。HOG 是基础 CPU 检测器，不能把这条体验当成生产检测效果测试。

## 接下来启用哪种检索

| 体验目标 | 额外配置 |
| --- | --- |
| 按解析后的属性检索 | 配置 VLM，解析裁剪图；严格标签匹配不能命中缺失的属性。 |
| 用文字描述检索候选画面 | 配置 Qwen embedding、独立 Milvus 视觉集合并建立索引；确认覆盖后再启用 `SEMANTIC_SEARCH_ENABLED`。 |
| 跨摄像头人员候选检索 | 配置 ReID 服务和索引，再用有标注的数据验证。 |

语义检索配置见[容器部署说明](../deploy/containers/README.md)，检索行为见[多模态检索说明](multimodal-retrieval/README.md)。相似度用于候选排序，不能作为身份认定或跨摄像头连续轨迹的证明。

## 反馈问题

在 [Issues](https://github.com/otterview-labs/SightIndex/issues) 提供版本、相关配置项、操作步骤和状态码，说明启用了哪些模型服务。不要附带凭据或私人画面。

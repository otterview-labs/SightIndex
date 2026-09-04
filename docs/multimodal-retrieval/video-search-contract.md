# 视频检索接口契约

## 目标

将已有的图片/人员裁剪向量命中聚合为可播放的视频结果。该接口替换当前空的 `POST /api/search/videos`，但保留路径以避免调用方迁移。

## 请求

```http
POST /api/search/videos
Content-Type: application/json
```

```json
{
  "query": "穿红色外套并背黑色背包的人",
  "top_k": 20,
  "filters": {
    "camera_id": null,
    "location_id": null,
    "start_time": null,
    "end_time": null
  },
  "target": "person_crop",
  "rerank": false,
  "merge_gap_ms": 3000
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `query` | 自然语言文本；后续可增加 `query_image_id` 以支持以图搜视频 |
| `top_k` | 最终返回的视频片段数量，范围 1-100 |
| `filters` | 复用现有 `SearchFilters`，支持人员、相机、位置与时间范围过滤 |
| `target` | `image` 表示场景检索；`person_crop` 表示人员外观检索 |
| `rerank` | 复用现有精排开关 |
| `merge_gap_ms` | 同一视频中相邻命中合并的最大时间间隔 |

## 返回

```json
{
  "items": [
    {
      "video_id": "6048f293-ff19-48b5-b3ce-c5c8496fca09",
      "video_url": "/data/videos/warehouse-001.mp4",
      "thumbnail_url": "/data/frames/warehouse-001_00012000.jpg",
      "score": 0.9132,
      "start_ms": 12000,
      "end_ms": 18000,
      "matched_image_ids": ["6b81e113-f358-4381-bd4a-d2bc7295e1d7"],
      "matched_crop_ids": ["e47d2e68-a2e5-4484-93eb-4495c0ca3eb5"],
      "person_id": null,
      "person_name": null,
      "camera_id": "195369cf-c137-489c-8356-db5b362f0d22",
      "location_id": "cc0360d5-1111-441e-9425-f79f30d6093c",
      "captured_at": "2026-07-26T10:30:12+08:00"
    }
  ]
}
```

`start_ms` 和 `end_ms` 由视频片段提供；播放端用它们作为跳转和循环播放范围。`score` 是该片段中所有命中帧/裁剪的最高分，不能把不同模型的分数直接混合。

## 聚合逻辑

```text
VisualSearchService
  -> SearchResultItem (image / crop hit)
  -> Image.video_asset_id + Image.video_segment_id
  -> group by video_asset_id
  -> merge adjacent segments when gap <= merge_gap_ms
  -> sort by highest segment score
  -> VideoSearchItem
```

处理规则：

1. 只有带有 `video_asset_id` 的图片或裁剪才会进入视频结果。
2. 同一片段中，分数取最高值，预览帧取最高分帧。
3. 同一视频的相邻片段只有在时间距离不大于 `merge_gap_ms` 时合并。
4. `person_crop` 命中优先返回人员、相机和位置元数据；`image` 命中返回场景级结果。
5. 删除视频时必须删除对应的 `VideoSegment`、`Image`、`PersonCrop`、`VLEmbedding` 和 Milvus object id。

## 以图搜视频扩展

现有 `POST /api/search/by-image` 已接受已入库 `image_id`。第一阶段可新增一个可选字段 `return_mode`：

```json
{
  "image_id": "...",
  "target": "person_crop",
  "top_k": 20,
  "return_mode": "video_segments"
}
```

当 `return_mode=video_segments` 时，复用同一套聚合服务；不要新增第二个图像嵌入路径。

## 测试

`tests/test_video_search.py` 至少覆盖：

1. 同一视频相邻命中合并为一个结果。
2. 相隔较远的命中拆分为多个结果。
3. 图片/裁剪不属于视频时不进入结果。
4. 相机、位置、时间和人员过滤条件生效。
5. 最高分、预览帧、`start_ms` 和 `end_ms` 正确返回。
6. `/api/search/videos` 的文本检索和以图搜视频兼容路径均返回正确响应。

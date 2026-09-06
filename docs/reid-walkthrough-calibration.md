# 跨摄像头 ReID 走测与阈值校准

## 采集要求

1. 至少 5 人参与，每个人使用固定编号，例如 `P01` 到 `P05`。
2. 每个人依次经过两个摄像头，每个方向至少 2 次，人与人之间间隔 20 秒以上。
3. 至少包含一次正面、一次背面、一次手持物或背包变化；不要在一次走测中换整套衣服。
4. 在观察表记录每次经过产生的 crop ID、摄像头和时间。
5. 为每个正样本配 2 个相近时间的不同人负样本，避免只用外观差异特别大的简单负例。

## 标注文件

优先从观察表点「找相似」进入 ReID 页面，放大图片核对后，在每个候选上点「同一个人」或
「不是同一个人」。同一查询图和候选图再次点击会更新原记录，不会重复追加；这些反馈只进入
`reid_match_feedback` 校准表，**不会直接修改当前排序、阈值或 `person_id`**。

页面右侧「导出全部标注」会下载 `reid-feedback.csv`，格式可直接传给评估脚本。也可以在服务器上导出：

```bash
curl -fsS -o data/reports/reid-feedback.csv \
  http://127.0.0.1:18030/api/reid/feedback/export.csv
```

CSV 同时保留点击当时的人体分数、人脸相似度与可靠性、标签一致计数、融合分数和判定说明。评估脚本仍会
从当前模型和数据库重新计算核心分数；因此可以对比“当时页面证据”和“更新模型后的证据”，避免阈值或模型
变化后无法解释旧标注。

如果需要先离线组织现场编号和备注，也可以复制 `docs/reid-walkthrough-template.csv`。每行是一对抓拍：

- `query_crop_id`：查询抓拍。
- `candidate_crop_id`：候选抓拍。
- `same_person`：同一个人填 `true`，不同人填 `false`。
- `person_code`：现场编号，仅用于审计，不进入系统身份库。
- `notes`：正面、背面、遮挡、背包等情况。

至少准备 30 个正样本对和 60 个负样本对后再调整生产阈值。

## 运行评估

```bash
cd /opt/sightindex
.venv/bin/python scripts/evaluate_reid_walkthrough.py \
  data/reports/reid-feedback.csv \
  --output data/reports/reid-calibration-report.json
```

报告分别给出：

- 人体向量阈值及精确率、召回率、F1、平衡准确率。
- 具备可靠人脸的样本子集上的人脸阈值。
- 标签冲突阈值为 1、2、3 时的误接纳与误拒绝情况。

报告只提出建议，不自动修改 `.env`。阈值变更必须保留走测 CSV 和报告，并在变更后重跑页面回归。

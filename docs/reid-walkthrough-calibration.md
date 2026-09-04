# 跨摄像头 ReID 走测与阈值校准

## 采集要求

1. 至少 5 人参与，每个人使用固定编号，例如 `P01` 到 `P05`。
2. 每个人依次经过两个摄像头，每个方向至少 2 次，人与人之间间隔 20 秒以上。
3. 至少包含一次正面、一次背面、一次手持物或背包变化；不要在一次走测中换整套衣服。
4. 在观察表记录每次经过产生的 crop ID、摄像头和时间。
5. 为每个正样本配 2 个相近时间的不同人负样本，避免只用外观差异特别大的简单负例。

## 标注文件

复制 `docs/reid-walkthrough-template.csv`，每行是一对抓拍：

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
  docs/reid-walkthrough-labels.csv \
  --output data/reports/reid-calibration-report.json
```

报告分别给出：

- 人体向量阈值及精确率、召回率、F1、平衡准确率。
- 具备可靠人脸的样本子集上的人脸阈值。
- 标签冲突阈值为 1、2、3 时的误接纳与误拒绝情况。

报告只提出建议，不自动修改 `.env`。阈值变更必须保留走测 CSV 和报告，并在变更后重跑页面回归。

# Reviewed demo snapshot

These three PNG files are detector crops and an annotated derivative of the public
[NASA photograph](../README.md). The source photograph's credit and rights statement apply;
these media assets are not relicensed as Apache-2.0. No names or identity labels are assigned.

The snapshot was produced locally on 2026-11-02 using the real processing pipeline at
`0ccf7feef36f06a9764ee98070823e6ed8e84561`: OpenCV 4.11 HOG, followed by CV tone extraction
with locally available YOLO pose weights. `manifest.json` records the actual bounding boxes,
attributes, source hash, output hashes and pose-weight hash. The public deployment loads this
snapshot and searches its stored attributes; it does not run inference for visitors.

There are four people in the source and two detected boxes. The padded crops include nearby
people, so clothing labels can be ambiguous. This sample illustrates retrieval and visual
review, not detector accuracy or identity matching. The source photo is a still image, so no
camera, capture timeline, tracking or video playback is implied.

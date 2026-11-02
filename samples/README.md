# Walkthrough sample

`nasa-walk.jpg` is a 1200 × 800, metadata-free JPEG derivative of
[STS-135 crew walk to dinner](https://commons.wikimedia.org/wiki/File:STS-135_crew_walk_to_dinner.jpg).

- Photographer credit: NASA / Houston Chronicle / Smiley N. Poole (as credited on Commons).
- Photograph date: 2011-04-07. Original file revision: 2011-05-19, 12:49 UTC.
- Source file: <https://upload.wikimedia.org/wikipedia/commons/5/5e/STS-135_crew_walk_to_dinner.jpg>.
- Commons marks the photograph **public domain in the United States (PD NASA)**.
  See the source page for the rights statement. This third-party image is not relicensed as
  Apache-2.0. No affiliation or endorsement by NASA or the people pictured is implied.
- Changes: resized from 3748 × 2498; JPEG re-encoded without EXIF metadata. No people added or removed.

This is a public still photograph used to illustrate ingestion and candidate review. It is not
customer footage, a surveillance dataset, identity ground truth or an accuracy benchmark.
Do not attach the pictured people's names to detected crops or use it to evaluate identity matching.

Follow the [console tour](../docs/showcase.md) or run, against your own local instance:

```bash
.venv/bin/python scripts/demo_walkthrough.py
```

Each run uploads one new image and its actual detector crops. It does not reset existing data,
enable external services, download model weights or make paid model calls. A completed image's
processing retry should reuse its crops. Detection counts may vary.

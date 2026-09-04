# Third-party notices

This file records source that is copied into the repository rather than installed through a
package manager. Package dependencies remain subject to their own licenses.

## SapiensID

- Upstream project: [mk-minchul/sapiensid](https://github.com/mk-minchul/sapiensid)
- Upstream reference revision: `a8d8011117db23f20c7ade3bc38e8075810ac886`
- Paper: [SapiensID: Foundation for Human Recognition](https://arxiv.org/abs/2504.04708)
- Authors: Minchul Kim, Dingqiang Ye, Yiyang Su, Feng Liu, and Xiaoming Liu
- Local path: `deploy/agx/reid_service/sapiensid/`
- Upstream license: Creative Commons Attribution-NonCommercial 4.0 International
- Retained license: `deploy/agx/reid_service/sapiensid/LICENSE`

SightIndex vendors an inference-oriented subset of the upstream tree and its bundled DFA aligner
asset. Dataset preparation and validation tooling were omitted. The local integration adds an HTTP
service and includes a defensive no-detection guard in the aligner path.

The CC BY-NC 4.0 terms restrict the vendored material to non-commercial use and require
attribution. They do not automatically apply to unrelated SightIndex files, and no project-wide
license overrides them. SapiensID checkpoints are not distributed in this repository and may be
subject to additional terms from their publisher or download host.

The bundled `openpose_batched/open_pose` preprocessing code carries an additional notice from its
authors limiting that preprocessor to non-commercial use. Its source comments and attribution are
retained in place.

Before redistributing the vendored directory, a modified copy, or any model weights, review the
retained license and the current upstream terms.

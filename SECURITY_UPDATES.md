# Dependency security updates

## 2026-10-19

This update addresses five GitHub Dependabot alerts in `uv.lock` by changing three
package versions. Other locked package versions are unchanged.

| Package | Previous lock | Updated lock | GitHub advisories |
| --- | --- | --- | --- |
| sentence-transformers | 5.5.0 | 5.6.0 | [GHSA-jhr6-gm9c-rqjv](https://github.com/advisories/GHSA-jhr6-gm9c-rqjv) |
| fsspec | 2026.4.0 | 2026.6.0 | [GHSA-27vj-qcqg-25rc](https://github.com/advisories/GHSA-27vj-qcqg-25rc) |
| urllib3 | 2.7.0 | 2.8.0 | [GHSA-8988-9cw3-xx77](https://github.com/advisories/GHSA-8988-9cw3-xx77), [GHSA-vxq7-64xx-v4gw](https://github.com/advisories/GHSA-vxq7-64xx-v4gw), [GHSA-gh4c-6fx4-qh6g](https://github.com/advisories/GHSA-gh4c-6fx4-qh6g) |

`constraints.security.txt` applies the patched version floors to pip installation
profiles, including the container requirements. The uv project carries matching
constraints for transitive dependencies and a sentence-transformers requirement
in the existing major version. Constraints do not enable optional model backends.

### Existing installations

Updating the repository does not change dependencies in a running service.

- **uv:** run `uv sync --locked` with the same extras used by the installation.
- **pip:** re-run the same requirements file used to install the service, then run
  `python -m pip check`. Preserve deployment-specific CUDA/PyTorch wheel choices.
- **Containers:** rebuild the service image from the updated sources, then recreate
  the container through the deployment's normal release process. Restarting an old
  image does not apply the dependency update.

Only load model code and local model directories from trusted sources. A dependency
update does not make arbitrary custom model code safe to execute.

No production services or GPU runtimes are changed by this repository update.

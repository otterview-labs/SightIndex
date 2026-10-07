"""Use tiny temporary bundles and strict fake HTTP; never fetch actual model assets."""

from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from urllib import request

import pytest

from deploy.models import assets

DATA = b"synthetic-model-weights-for-offline-unit-tests"
URL = "https://models.example.invalid/model?token=synthetic-do-not-print"


def artifact(identifier="weights", path="pack/weights.bin", data=DATA, **kwargs):
    return assets.Artifact(
        id=identifier,
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        model="unit/model",
        revision="fixed-test-revision",
        **kwargs,
    )


def bundle(*artifacts):
    return assets.Manifest(1, tuple(artifacts or (artifact(),)))


def write_file(root, relative, data=DATA):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def write_manifest(root, manifest=None):
    path = root / "approved-manifest.json"
    manifest = manifest or bundle()
    path.write_text(
        json.dumps(
            {"version": manifest.version, "artifacts": [asdict(a) for a in manifest.artifacts]}
        )
    )
    return path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    # macOS /var and /tmp can be aliases; the helper deliberately rejects symlink parents.
    root = tmp_path.resolve()

    def no_network(*args, **kwargs):
        raise AssertionError("real network forbidden in model asset tests")

    monkeypatch.setattr(assets, "_open_http", no_network)
    source = root / "source"
    source.mkdir()
    return source, root / "models"


def offline(manifest, destination, source, **kwargs):
    return assets.prepare_assets(manifest, destination, source_dir=source, margin_bytes=0, **kwargs)


class FakeResponse:
    def __init__(self, data=DATA, status=200, headers=None, url=URL):
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(data))}
        self.body = io.BytesIO(data)
        self.url = url
        self.closed = False

    def read(self, size=-1):
        assert 0 < size <= assets.CHUNK_BYTES
        return self.body.read(size)

    def geturl(self):
        return self.url

    def close(self):
        self.closed = True


class FakeHTTP:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, req, timeout):
        assert isinstance(req, request.Request)
        assert req.full_url == URL
        assert req.get_method() == "GET"
        assert req.get_header("Accept-encoding") == "identity"
        assert 0 < timeout <= 30
        self.calls.append((req, timeout))
        assert self.responses, "unexpected extra HTTP request"
        return self.responses.pop(0)


def download(manifest, destination, http, **kwargs):
    return assets.prepare_assets(
        manifest, destination, download=True, opener=http, margin_bytes=0, **kwargs
    )


def test_manifest_local_schema_optional_metadata_and_filtered_constructor(roots):
    source, _destination = roots
    approved = artifact(
        source_url=URL, terms_url="https://models.example.invalid/terms", license="unit-license"
    )
    loaded = assets.load_manifest(write_manifest(source, bundle(approved)))
    assert loaded == bundle(approved)
    assert loaded.artifacts[0].license == "unit-license"
    assert assets.Manifest(loaded.version, (loaded.artifacts[0],)) == loaded
    assert "synthetic-do-not-print" not in repr(approved)


@pytest.mark.parametrize(
    "digest",
    [
        hashlib.sha256(DATA).hexdigest().upper(),
        hashlib.sha256(DATA).hexdigest()[:32].upper() + hashlib.sha256(DATA).hexdigest()[32:],
    ],
)
def test_sha256_is_canonicalized_for_model_identity_comparisons(roots, digest):
    source, destination = roots
    approved = replace(artifact(), sha256=digest)
    assert approved.sha256 == hashlib.sha256(DATA).hexdigest()
    loaded = assets.load_manifest(write_manifest(source, bundle(approved)))
    assert loaded.artifacts[0].sha256 == hashlib.sha256(DATA).hexdigest()
    write_file(destination, approved.path)
    assert assets.check_assets(loaded, destination).ok


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("id", "untrusted\nsecret", "manifest.invalid_id"),
        ("path", ".", "manifest.invalid_path"),
        ("path", "../weights.bin", "manifest.invalid_path"),
        ("path", "/weights.bin", "manifest.invalid_path"),
        ("path", "pack/../weights.bin", "manifest.invalid_path"),
        ("path", "pack//weights.bin", "manifest.invalid_path"),
        ("path", "pack\\weights.bin", "manifest.invalid_path"),
        ("path", "a" * 255, "manifest.invalid_path"),
        ("path", "pack/.sightindex-model-assets.lock", "manifest.invalid_path"),
        ("path", "pack/.SIGHTINDEX-MODEL-ASSETS.LOCK", "manifest.invalid_path"),
        ("path", "pack/.sightindex-model-assets-quarantine/x.bin", "manifest.invalid_path"),
        ("sha256", "not-a-review-approved-hash", "manifest.invalid_sha256"),
        ("size_bytes", 0, "manifest.invalid_size"),
        ("size_bytes", True, "manifest.invalid_size"),
        ("model", "", "manifest.invalid_metadata"),
        ("revision", "revision\nsecret", "manifest.invalid_metadata"),
        ("license", "", "manifest.invalid_license"),
        ("source_url", "http://models.example.invalid/file", "manifest.invalid_https_url"),
        ("source_url", "https://user:secret@example.invalid/file", "manifest.invalid_https_url"),
        ("source_url", "https://models.example.invalid:invalid/file", "manifest.invalid_https_url"),
        ("terms_url", "https://models.example.invalid/file#secret", "manifest.invalid_https_url"),
    ],
)
def test_manifest_rejects_unsafe_fields(field, value, code):
    with pytest.raises(assets.AssetError, match=code):
        replace(artifact(), **{field: value})


@pytest.mark.parametrize(
    ("artifacts", "code"),
    [
        ((artifact(), artifact(path="other.bin")), "manifest.duplicate_id"),
        ((artifact(), artifact("other")), "manifest.duplicate_path"),
        ((artifact(), artifact("other", "PACK/WEIGHTS.BIN")), "manifest.duplicate_path"),
        ((artifact(path="a"), artifact("other", "a/file.bin")), "manifest.path_collision"),
        ((artifact(path="a"), artifact("other", "a.partial")), "manifest.path_collision"),
        ((artifact(path="a"), artifact("other", "a.partial/x")), "manifest.path_collision"),
    ],
)
def test_manifest_rejects_duplicate_and_conflicting_layouts(artifacts, code):
    with pytest.raises(assets.AssetError, match=code):
        bundle(*artifacts)


@pytest.mark.parametrize(
    "payload",
    [
        {"version": True, "artifacts": [asdict(artifact())]},
        {"version": 2, "artifacts": [asdict(artifact())]},
        {"version": 1, "artifacts": []},
        {"version": 1, "artifacts": "invalid"},
        {"version": 1, "artifacts": [asdict(artifact())], "unknown": "secret"},
        {"version": 1, "artifacts": [{**asdict(artifact()), "unknown": "secret"}]},
        {"version": 1, "artifacts": [{"id": "weights"}]},
    ],
)
def test_load_manifest_strict_schema(roots, payload):
    source, _destination = roots
    path = source / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(assets.AssetError):
        assets.load_manifest(path)


def test_manifest_duplicate_json_keys_and_hardlinks_are_rejected(roots):
    source, _destination = roots
    path = source / "manifest.json"
    path.write_text('{"version": 1, "version": 1, "artifacts": []}')
    with pytest.raises(assets.AssetError, match="manifest.duplicate_key"):
        assets.load_manifest(path)
    write_manifest(source)
    os.link(source / "approved-manifest.json", source / "linked.json")
    with pytest.raises(assets.AssetError, match="filesystem.unsafe_file"):
        assets.load_manifest(source / "linked.json")


def test_check_missing_is_pure_read_and_has_structured_status(roots, monkeypatch):
    _source, destination = roots

    def no_mkdir(*args, **kwargs):
        raise AssertionError("check cannot create directories")

    monkeypatch.setattr(assets.os, "mkdir", no_mkdir)
    report = assets.check_assets(bundle(), destination)
    assert not report.ok
    assert report.as_dict() == {
        "mode": "check",
        "ok": False,
        "results": [
            {
                "id": "weights",
                "path": "pack/weights.bin",
                "status": "missing",
                "code": "asset.missing",
            }
        ],
    }
    assert not destination.exists()


def test_check_validates_every_file_exact_size_and_sha(roots):
    _source, destination = roots
    manifest = bundle(
        artifact("good", "good.bin"),
        artifact("size", "size.bin"),
        artifact("sha", "sha.bin"),
        artifact("absent", "absent.bin"),
    )
    write_file(destination, "good.bin")
    write_file(destination, "size.bin", DATA[:-1])
    write_file(destination, "sha.bin", b"x" * len(DATA))
    report = assets.check_assets(manifest, destination)
    assert not report.ok
    assert [result.status for result in report.results] == [
        "verified",
        "mismatch",
        "mismatch",
        "missing",
    ]
    assert [result.code for result in report.results][1:3] == [
        "asset.size_mismatch",
        "asset.sha256_mismatch",
    ]
    assert not (destination / assets.LOCK_NAME).exists()


@pytest.mark.parametrize("kind", ["symlink-file", "symlink-parent", "hardlink", "partial-hardlink"])
def test_check_and_prepare_refuse_unsafe_existing_files(roots, kind):
    source, destination = roots
    outside = write_file(source, "outside.bin")
    destination.mkdir()
    target = destination / artifact().path
    if kind == "symlink-parent":
        (destination / "pack").symlink_to(source, target_is_directory=True)
        write_file(source, "weights.bin")
    else:
        target.parent.mkdir()
        if kind == "symlink-file":
            target.symlink_to(outside)
        elif kind == "hardlink":
            os.link(outside, target)
        else:
            os.link(outside, target.with_name(target.name + ".partial"))
    for report in (
        assets.check_assets(bundle(), destination),
        offline(bundle(), destination, source),
    ):
        assert not report.ok
        assert report.results[0].status == "failed"
        assert report.results[0].code.startswith("filesystem.")
    assert outside.read_bytes() == DATA
    assert not (destination / assets.LOCK_NAME).exists()


def test_offline_prepare_then_readonly_check_and_reuse(roots, monkeypatch):
    source, destination = roots
    write_file(source, artifact().path)
    report = offline(bundle(), destination, source)
    assert report.ok, report.as_dict()
    final = destination / artifact().path
    assert final.read_bytes() == DATA
    assert final.stat().st_nlink == 1
    assert final.stat().st_mode & 0o777 == 0o644
    assert not final.with_name(final.name + ".partial").exists()
    assert (destination / assets.LOCK_NAME).read_bytes() == assets.LOCK_MARKER
    assert assets.check_assets(bundle(), destination).ok

    def no_copy(*args, **kwargs):
        raise AssertionError("matching final files must be reused")

    monkeypatch.setattr(assets, "_copy_offline", no_copy)
    reused = offline(bundle(), destination, source / "does-not-exist")
    assert reused.ok
    assert reused.results[0].status == "reused"


@pytest.mark.parametrize("existing_private_directory", [False, True])
def test_subprocess_umask077_new_directories_traversable_existing_permissions_preserved(
    roots,
    existing_private_directory,
):
    source, destination = roots
    approved = artifact(path="pack/nested/weights.bin")
    manifest = write_manifest(source, bundle(approved))
    write_file(source, approved.path)
    if existing_private_directory:
        destination.mkdir(mode=0o700)
        (destination / "pack").mkdir(mode=0o700)
    script = (
        "import os, sys; os.umask(0o077); "
        "from deploy.models.assets import main; raise SystemExit(main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--manifest",
            str(manifest),
            "--destination",
            str(destination),
            "--mode",
            "prepare",
            "--source-dir",
            str(source),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ok"]
    expected_parent_mode = 0o700 if existing_private_directory else 0o755
    assert destination.stat().st_mode & 0o777 == expected_parent_mode
    assert (destination / "pack").stat().st_mode & 0o777 == expected_parent_mode
    assert (destination / "pack/nested").stat().st_mode & 0o777 == 0o755
    assert (destination / approved.path).stat().st_mode & 0o777 == 0o644
    assert (destination / approved.path).read_bytes() == DATA


def test_mkdir_race_winner_keeps_its_existing_private_permissions(roots, monkeypatch):
    _source, destination = roots
    original_mkdir = assets.os.mkdir

    def race_winner(path, mode=0o777, *, dir_fd=None):
        original_mkdir(path, 0o700, dir_fd=dir_fd)
        raise FileExistsError("synthetic competing creator")

    monkeypatch.setattr(assets.os, "mkdir", race_winner)
    with assets._directory(destination, create=True):
        pass
    assert destination.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("existing", [DATA[:-1], b"x" * len(DATA)])
def test_wrong_existing_file_is_never_overwritten(roots, existing):
    source, destination = roots
    write_file(source, artifact().path)
    final = write_file(destination, artifact().path, existing)
    report = offline(bundle(), destination, source)
    assert not report.ok
    assert report.error == "prepare.existing_file_invalid"
    assert final.read_bytes() == existing
    assert not (destination / assets.LOCK_NAME).exists()


def test_offline_partial_resumes_only_matching_prefix(roots):
    source, destination = roots
    write_file(source, artifact().path)
    partial = write_file(destination, artifact().path + ".partial", DATA[:7])
    assert offline(bundle(), destination, source).ok
    assert not partial.exists()
    (destination / artifact().path).unlink()
    partial.write_bytes(b"wrong")
    report = offline(bundle(), destination, source)
    assert not report.ok
    assert report.results[0].code == "source.partial_prefix_mismatch"
    assert partial.read_bytes() == b"wrong"
    assert not (destination / artifact().path).exists()


@pytest.mark.parametrize("kind", ["missing", "wrong-size", "wrong-hash", "hardlink", "symlink"])
def test_offline_source_safety_and_integrity(roots, kind):
    source, destination = roots
    if kind != "missing":
        data = DATA[:-1] if kind == "wrong-size" else b"x" * len(DATA)
        input_file = write_file(source, artifact().path, data)
        if kind == "hardlink":
            os.link(input_file, source / "hardlink.bin")
        elif kind == "symlink":
            input_file.unlink()
            outside = write_file(source, "outside.bin")
            input_file.symlink_to(outside)
    report = offline(bundle(), destination, source)
    assert not report.ok
    assert not (destination / artifact().path).exists()
    assert (destination / assets.LOCK_NAME).read_bytes() == assets.LOCK_MARKER


def test_disk_gate_runs_before_creating_destination_and_counts_remaining_bytes(roots):
    source, destination = roots
    write_file(source, artifact().path)
    report = offline(bundle(), destination, source, free_bytes=lambda _path: len(DATA) - 1)
    assert not report.ok and report.error == "prepare.insufficient_disk"
    assert not destination.exists()
    write_file(destination, artifact().path + ".partial", DATA[:10])
    enough = offline(bundle(), destination, source, free_bytes=lambda _path: len(DATA) - 10)
    assert enough.ok, enough.as_dict()


def test_margin_and_aggregate_disk_requirement(roots):
    source, destination = roots
    manifest = bundle(artifact("a", "a.bin"), artifact("b", "b.bin"))
    write_file(source, "a.bin")
    write_file(source, "b.bin")
    report = assets.prepare_assets(
        manifest,
        destination,
        source_dir=source,
        margin_bytes=11,
        free_bytes=lambda _path: 2 * len(DATA) + 10,
    )
    assert not report.ok and report.error == "prepare.insufficient_disk"
    assert not destination.exists()


def test_existing_package_lock_is_preserved(roots):
    source, destination = roots
    write_file(source, artifact().path)
    lock = write_file(destination, assets.LOCK_NAME, b"another preparation owns this lock")
    report = offline(bundle(), destination, source)
    assert not report.ok and report.error == "prepare.package_locked"
    assert lock.read_bytes() == b"another preparation owns this lock"
    assert not (destination / artifact().path).parent.exists()


def test_trusted_mapping_preselected_files_still_uses_original_offline_layout(roots):
    source, destination = roots
    manifest = bundle(artifact("a", "original/a.bin"), artifact("b", "original/b.bin"))
    for approved in manifest.artifacts:
        write_file(source, approved.path)
    mapping = {
        "a": destination / "fixed-role/a.bin",
        "b": destination.parent / "external-model/fixed-role/b.bin",
    }
    report = offline(manifest, destination, source, destination_for=lambda item: mapping[item.id])
    assert report.ok, report.as_dict()
    assert all(path.read_bytes() == DATA for path in mapping.values())
    assert assets.check_assets(manifest, destination, destination_for=lambda a: mapping[a.id]).ok
    assert not (destination / "original").exists()


def test_mapper_cannot_alias_destination_or_escape_through_symlink(roots):
    source, destination = roots
    manifest = bundle(artifact("a", "a.bin"), artifact("b", "b.bin"))
    report = assets.check_assets(manifest, destination, destination_for=lambda _a: source / "same")
    assert report.error == "manifest.duplicate_destination"
    (source / "alias").symlink_to(source, target_is_directory=True)
    report = assets.check_assets(
        bundle(), destination, destination_for=lambda _a: source / "alias/weights"
    )
    assert not report.ok and report.results[0].code == "filesystem.unsafe_directory"


def test_https_fresh_download_and_valid_range_resume(roots):
    _source, destination = roots
    approved = artifact(source_url=URL)
    response = FakeResponse()
    http = FakeHTTP(response)
    report = download(bundle(approved), destination, http)
    assert report.ok, report.as_dict()
    assert http.calls[0][0].get_header("Range") is None
    assert response.closed
    (destination / approved.path).unlink()
    write_file(destination, approved.path + ".partial", DATA[:9])
    response = FakeResponse(
        DATA[9:],
        206,
        {
            "Content-Range": f"bytes 9-{len(DATA) - 1}/{len(DATA)}",
            "Content-Length": str(len(DATA) - 9),
        },
    )
    http = FakeHTTP(response)
    report = download(bundle(approved), destination, http)
    assert report.ok, report.as_dict()
    assert http.calls[0][0].get_header("Range") == "bytes=9-"
    assert (destination / approved.path).read_bytes() == DATA


def test_range_ignored_with_200_restarts_instead_of_appending(roots):
    _source, destination = roots
    write_file(destination, artifact().path + ".partial", b"incorrect-prefix")
    report = download(bundle(artifact(source_url=URL)), destination, FakeHTTP(FakeResponse()))
    assert report.ok, report.as_dict()
    assert (destination / artifact().path).read_bytes() == DATA


def test_download_disk_gate_reserves_full_size_before_range_may_restart(roots):
    _source, destination = roots
    partial = write_file(destination, artifact().path + ".partial", DATA[:10])
    http = FakeHTTP(FakeResponse())
    report = download(
        bundle(artifact(source_url=URL)), destination, http, free_bytes=lambda _path: len(DATA) - 10
    )
    assert not report.ok and report.error == "prepare.insufficient_disk"
    assert not http.calls
    assert partial.read_bytes() == DATA[:10]
    assert not (destination / assets.LOCK_NAME).exists()
    report = download(
        bundle(artifact(source_url=URL)), destination, http, free_bytes=lambda _path: len(DATA)
    )
    assert report.ok, report.as_dict()


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (FakeResponse(DATA[4:], 206, {}), "download.invalid_content_range"),
        (
            FakeResponse(DATA[4:], 206, {"Content-Range": "bytes 3-9/10"}),
            "download.invalid_content_range",
        ),
        (
            FakeResponse(DATA[4:], 206, {"Content-Range": "bytes 4-9/*"}),
            "download.invalid_content_range",
        ),
        (FakeResponse(DATA, 200, {"Content-Length": "999"}), "download.invalid_content_length"),
        (FakeResponse(DATA, 200, {"Content-Length": "-1"}), "download.invalid_content_length"),
        (FakeResponse(DATA, 200, {"Content-Encoding": "gzip"}), "download.invalid_encoding"),
        (FakeResponse(DATA, 302), "download.invalid_http_status"),
        (
            FakeResponse(DATA, url="http://models.example.invalid/insecure"),
            "manifest.invalid_https_url",
        ),
    ],
)
def test_download_rejects_unsafe_headers_before_touching_partial(roots, response, code):
    _source, destination = roots
    partial = write_file(destination, artifact().path + ".partial", DATA[:4])
    report = download(bundle(artifact(source_url=URL)), destination, FakeHTTP(response))
    assert not report.ok
    assert report.results[0].code == code
    assert partial.read_bytes() == DATA[:4]
    assert not (destination / artifact().path).exists()
    assert response.closed


def test_short_download_caches_partial_and_next_request_resumes(roots):
    _source, destination = roots
    manifest = bundle(artifact(source_url=URL))
    http = FakeHTTP(FakeResponse(DATA[:6], headers={"Content-Length": str(len(DATA))}))
    first = download(manifest, destination, http)
    assert not first.ok and first.results[0].code == "download.incomplete"
    assert (destination / (artifact().path + ".partial")).read_bytes() == DATA[:6]
    http = FakeHTTP(
        FakeResponse(
            DATA[6:],
            206,
            {
                "Content-Range": f"bytes 6-{len(DATA) - 1}/{len(DATA)}",
                "Content-Length": str(len(DATA) - 6),
            },
        )
    )
    assert download(manifest, destination, http).ok
    assert http.calls[0][0].get_header("Range") == "bytes=6-"


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (DATA + b"unexpected", "download.response_too_large"),
        (b"x" * len(DATA), "asset.sha256_mismatch"),
    ],
)
def test_wrong_download_never_promotes(roots, data, code):
    _source, destination = roots
    report = download(
        bundle(artifact(source_url=URL)), destination, FakeHTTP(FakeResponse(data, headers={}))
    )
    assert not report.ok and report.results[0].code == code
    assert not (destination / artifact().path).exists()


def test_full_verified_partial_promotes_without_network(roots):
    _source, destination = roots
    write_file(destination, artifact().path + ".partial")
    assert download(bundle(artifact()), destination, FakeHTTP()).ok


def test_download_deadline_is_bounded_and_never_promotes(roots, monkeypatch):
    _source, destination = roots
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(assets.time, "monotonic", lambda: next(ticks))
    http = FakeHTTP(FakeResponse())
    report = download(bundle(artifact(source_url=URL)), destination, http, max_download_seconds=1.0)
    assert not report.ok and report.results[0].code == "download.deadline_exceeded"
    assert http.calls[0][1] == 1.0
    assert not (destination / artifact().path).exists()


def test_download_without_approved_source_url_never_opens_http_or_creates_directory(roots):
    _source, destination = roots
    http = FakeHTTP()
    report = download(bundle(), destination, http)
    assert not report.ok and report.error == "download.source_url_missing"
    assert not http.calls and not destination.exists()


def test_network_exception_and_cli_errors_never_echo_url_body_or_token(roots, capsys):
    source, destination = roots

    def failure(_req, _timeout):
        raise TimeoutError(URL + " response-body-secret")

    report = download(bundle(artifact(source_url=URL)), destination, failure)
    assert not report.ok and report.results[0].code == "download.failed"
    serialized = json.dumps(report.as_dict())
    assert URL not in serialized and "response-body-secret" not in serialized
    assert assets.main(["--not-a-real-option", URL]) == 1
    assert "synthetic-do-not-print" not in capsys.readouterr().out
    manifest = write_manifest(source)
    assert (
        assets.main(
            [
                "--manifest",
                str(manifest),
                "--destination",
                str(destination),
                "--mode",
                "check",
                "--download",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "check.source_flags_forbidden"


def test_https_redirect_handler_refuses_downgrade_and_credentials():
    handler = assets._HTTPSRedirect()
    original = request.Request(URL)
    for url in ("http://models.example.invalid/file", "https://user:secret@example.invalid/file"):
        with pytest.raises(assets.AssetError, match="download.unsafe_redirect"):
            handler.redirect_request(original, io.BytesIO(), 302, "redirect", {}, url)
    target = handler.redirect_request(
        original, io.BytesIO(), 302, "redirect", {}, "https://cdn.example.invalid/file"
    )
    assert target is not None and target.full_url == "https://cdn.example.invalid/file"


def test_atomic_publication_preserves_destination_created_by_race(roots, monkeypatch):
    source, destination = roots
    write_file(source, artifact().path)
    original_link = assets.os.link

    def raced_link(src, dst, **kwargs):
        descriptor = os.open(
            dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=kwargs["dst_dir_fd"]
        )
        with os.fdopen(descriptor, "wb") as raced:
            raced.write(b"other-operation")
        return original_link(src, dst, **kwargs)

    monkeypatch.setattr(assets.os, "link", raced_link)
    report = offline(bundle(), destination, source)
    assert not report.ok and report.results[0].code == "prepare.destination_appeared"
    assert (destination / artifact().path).read_bytes() == b"other-operation"
    assert (destination / (artifact().path + ".partial")).read_bytes() == DATA


def test_partial_swapped_at_publication_cannot_leave_unsafe_destination(roots, monkeypatch):
    source, destination = roots
    write_file(source, artifact().path)
    outside = write_file(source, "outside.bin", b"must-stay-unchanged")
    original_link = assets.os.link

    def raced_link(src, dst, **kwargs):
        os.unlink(src, dir_fd=kwargs["src_dir_fd"])
        os.symlink(str(outside), src, dir_fd=kwargs["src_dir_fd"])
        return original_link(src, dst, **kwargs)

    monkeypatch.setattr(assets.os, "link", raced_link)
    report = offline(bundle(), destination, source)
    assert not report.ok and report.results[0].code == "filesystem.file_changed"
    assert not (destination / artifact().path).is_symlink()
    assert not (destination / artifact().path).exists()
    assert outside.read_bytes() == b"must-stay-unchanged"


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"source_dir": Path("unused"), "download": True},
        {"download": True, "timeout_seconds": float("inf")},
        {"download": True, "max_download_seconds": 0},
        {"download": True, "margin_bytes": -1},
    ],
)
def test_invalid_prepare_options_are_structured_and_side_effect_free(roots, options):
    _source, destination = roots
    report = assets.prepare_assets(bundle(), destination, **options)
    assert not report.ok and report.error
    assert not destination.exists()


def test_cli_check_prepare_and_help_do_not_consult_application_settings(roots, capsys):
    source, destination = roots
    manifest = write_manifest(source)
    write_file(source, artifact().path)
    arguments = ["--manifest", str(manifest), "--destination", str(destination)]
    assert assets.main([*arguments, "--mode", "check"]) == 1
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "missing"
    assert not destination.exists()
    assert assets.main([*arguments, "--mode", "prepare", "--source-dir", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "prepared"
    assert assets.main([*arguments, "--mode", "check"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    with pytest.raises(SystemExit) as exit_info:
        assets.main(["--help"])
    assert exit_info.value.code == 0
    assert "--source-dir" in capsys.readouterr().out


def test_empty_legacy_lock_still_blocks_normal_prepare_without_guessing_stale(roots):
    source, destination = roots
    write_file(source, artifact().path)
    lock = write_file(destination, assets.LOCK_NAME, b"")
    report = offline(bundle(), destination, source)
    assert not report.ok and report.error == "prepare.package_locked"
    assert lock.read_bytes() == b""
    assert assets.diagnose_assets(bundle(), destination).lock == "lock.legacy_empty"


def test_kernel_lock_is_persistent_reusable_and_never_overridden_while_active(roots):
    source, destination = roots
    write_file(source, artifact().path)
    with assets._package_lock(destination):
        before = (destination / assets.LOCK_NAME).stat()
        assert assets.diagnose_assets(bundle(), destination).lock == "lock.active"
        prepare = offline(bundle(), destination, source)
        recover = assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
        assert not prepare.ok and prepare.error == "prepare.package_locked"
        assert not recover.ok and recover.error == "prepare.package_locked"
        assert (destination / assets.LOCK_NAME).stat().st_ino == before.st_ino
        assert not (destination / assets.QUARANTINE_NAME).exists()
    assert assets.diagnose_assets(bundle(), destination).lock == "lock.released"
    assert offline(bundle(), destination, source).ok
    assert (destination / assets.LOCK_NAME).stat().st_ino == before.st_ino


def test_sigkill_releases_kernel_lock_without_manual_recovery(roots):
    source, destination = roots
    write_file(source, artifact().path)
    # Only this synthetic child is killed; no service or real model process is used.
    script = (
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "from deploy.models.assets import _package_lock\n"
        "with _package_lock(Path(sys.argv[1])):\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(destination)],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == -signal.SIGKILL
    assert (destination / assets.LOCK_NAME).read_bytes() == assets.LOCK_MARKER
    assert assets.diagnose_assets(bundle(), destination).lock == "lock.released"
    assert offline(bundle(), destination, source).ok


@pytest.mark.parametrize("existing", [False, True])
def test_diagnosis_is_readonly_and_does_not_create_locks_or_directories(
    roots, monkeypatch, existing
):
    _source, destination = roots
    if existing:
        write_file(destination, artifact().path + ".partial", DATA[:4])
        write_file(destination, assets.LOCK_NAME, b"")
    before = {
        str(path): path.read_bytes() if path.is_file() else None
        for path in destination.parent.rglob("*")
    }

    def forbidden(*args, **kwargs):
        pytest.fail("diagnosis must not write or access the network")

    monkeypatch.setattr(assets.os, "mkdir", forbidden)
    monkeypatch.setattr(assets.os, "write", forbidden)
    monkeypatch.setattr(assets, "_open_http", forbidden)
    report = assets.diagnose_assets(bundle(), destination)
    assert report.mode == "diagnose"
    assert report.lock == ("lock.legacy_empty" if existing else "lock.absent")
    after = {
        str(path): path.read_bytes() if path.is_file() else None
        for path in destination.parent.rglob("*")
    }
    assert after == before


def test_recovery_requires_explicit_confirmation_and_known_manifest_roles_before_writes(roots):
    _source, destination = roots
    report = assets.recover_assets(bundle(), destination)
    assert not report.ok and report.error == "recovery.confirm_no_active_preparation_required"
    report = assets.recover_assets(
        bundle(),
        destination,
        confirm_no_active_preparation=True,
        quarantine_partial_ids=frozenset({"unknown"}),
    )
    assert not report.ok and report.error == "recovery.unknown_partial_role"
    assert not destination.exists()


def test_explicit_empty_legacy_recovery_preserves_bytes_and_never_has_unlocked_name_gap(
    roots,
    monkeypatch,
):
    source, destination = roots
    write_file(source, artifact().path)
    write_file(destination, assets.LOCK_NAME, b"")
    replace = assets.os.replace

    def guarded_replace(src, dst, **kwargs):
        assert dst == assets.LOCK_NAME
        with pytest.raises(FileExistsError):
            os.open(
                dst,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=kwargs["dst_dir_fd"],
            )
        return replace(src, dst, **kwargs)

    monkeypatch.setattr(assets.os, "replace", guarded_replace)
    report = assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
    assert report.ok and report.lock == "lock.legacy_quarantined"
    archived = tuple((destination / assets.QUARANTINE_NAME).glob("*/legacy.lock"))
    assert len(archived) == 1 and archived[0].read_bytes() == b""
    assert archived[0].stat().st_mode & 0o777 == 0o600
    assert archived[0].parent.stat().st_mode & 0o777 == 0o700
    assert (destination / assets.LOCK_NAME).read_bytes() == assets.LOCK_MARKER
    assert offline(bundle(), destination, source).ok


def test_legacy_migration_crash_before_replacement_is_recoverable(roots, monkeypatch):
    _source, destination = roots
    lock = write_file(destination, assets.LOCK_NAME, b"")
    replace = assets.os.replace

    def interrupted(*args, **kwargs):
        raise OSError("synthetic stop with secret " + URL)

    monkeypatch.setattr(assets.os, "replace", interrupted)
    first = assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
    assert not first.ok and first.error == "filesystem.recovery_failed"
    assert lock.read_bytes() == b"" and lock.stat().st_nlink == 1
    assert URL not in json.dumps(first.as_dict())
    monkeypatch.setattr(assets.os, "replace", replace)
    assert assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True).ok


def test_second_recovery_cannot_split_or_override_migrated_active_lock(roots, monkeypatch):
    _source, destination = roots
    write_file(destination, assets.LOCK_NAME, b"")
    inspect = assets._partial_state
    nested = []

    def check_inside_lock(*args, **kwargs):
        monkeypatch.setattr(assets, "_partial_state", inspect)
        nested.append(
            assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
        )
        return inspect(*args, **kwargs)

    monkeypatch.setattr(assets, "_partial_state", check_inside_lock)
    assert assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True).ok
    assert len(nested) == 1
    assert not nested[0].ok and nested[0].error == "prepare.package_locked"


@pytest.mark.parametrize("data", [DATA + b"too-large", b"x" * len(DATA)])
def test_recovery_isolates_only_proven_bad_selected_partial_then_prepare_succeeds(roots, data):
    source, destination = roots
    write_file(source, artifact().path)
    partial = write_file(destination, artifact().path + ".partial", data)
    outside = write_file(destination, "unlisted.bin.partial", b"not in manifest")
    before = partial.stat()
    assert not assets.diagnose_assets(bundle(), destination).ok
    report = assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
    assert report.ok and report.results[0].status == "quarantined"
    assert not partial.exists()
    isolated = tuple((partial.parent / assets.QUARANTINE_NAME).glob("*/*"))
    assert len(isolated) == 1 and isolated[0].read_bytes() == data
    assert isolated[0].stat().st_ino == before.st_ino
    assert isolated[0].stat().st_dev == before.st_dev
    assert outside.read_bytes() == b"not in manifest"
    assert offline(bundle(), destination, source).ok


@pytest.mark.parametrize("data", [DATA[:4], DATA])
def test_recovery_keeps_good_or_unverifiable_partial_unless_specific_role_requested(roots, data):
    _source, destination = roots
    partial = write_file(destination, artifact().path + ".partial", data)
    report = assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
    assert report.ok and report.results[0].status == "preserved"
    assert partial.read_bytes() == data
    assert not (partial.parent / assets.QUARANTINE_NAME).exists()
    report = assets.recover_assets(
        bundle(),
        destination,
        confirm_no_active_preparation=True,
        quarantine_partial_ids=frozenset({artifact().id}),
    )
    assert report.ok and report.results[0].status == "quarantined"
    assert not partial.exists()


@pytest.mark.parametrize("final_data", [DATA, b"x" * len(DATA)])
def test_recovery_never_changes_final_weight_even_if_partial_reset_requested(roots, final_data):
    _source, destination = roots
    final = write_file(destination, artifact().path, final_data)
    partial = write_file(destination, artifact().path + ".partial", b"bad")
    before = (final.read_bytes(), final.stat().st_ino, final.stat().st_mode)
    report = assets.recover_assets(
        bundle(),
        destination,
        confirm_no_active_preparation=True,
        quarantine_partial_ids=frozenset({artifact().id}),
    )
    assert report.ok is (final_data == DATA)
    assert (final.read_bytes(), final.stat().st_ino, final.stat().st_mode) == before
    assert partial.read_bytes() == b"bad"


@pytest.mark.parametrize("kind", ["unknown", "symlink", "hardlink"])
def test_recovery_refuses_unknown_or_unsafe_lock(roots, kind):
    source, destination = roots
    destination.mkdir()
    lock = destination / assets.LOCK_NAME
    outside = write_file(source, "outside", b"not a model lock " + URL.encode())
    if kind == "unknown":
        lock.write_bytes(outside.read_bytes())
    elif kind == "symlink":
        lock.symlink_to(outside)
    else:
        os.link(outside, lock)
    report = assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
    assert not report.ok
    assert URL not in json.dumps(report.as_dict())
    assert outside.read_bytes() == b"not a model lock " + URL.encode()
    assert not (destination / assets.QUARANTINE_NAME).exists()


def test_recovery_skips_missing_parent_but_still_inspects_later_mapped_artifact(roots):
    _source, destination = roots
    first, second = artifact("first", "first.bin"), artifact("second", "second.bin")
    mapping = {
        "first": destination / "missing/first.bin",
        "second": destination / "existing/second.bin",
    }
    partial = write_file(destination, "existing/second.bin.partial", b"x" * len(DATA))
    report = assets.recover_assets(
        bundle(first, second),
        destination,
        confirm_no_active_preparation=True,
        destination_for=lambda selected: mapping[selected.id],
    )
    assert report.ok and [result.status for result in report.results] == ["missing", "quarantined"]
    assert not (destination / "missing").exists()
    assert not partial.exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_diagnostic_and_recovery_reject_unsafe_partial_even_with_verified_final(roots, kind):
    source, destination = roots
    final = write_file(destination, artifact().path)
    outside = write_file(source, "outside", b"preserve this external file")
    partial = final.with_name(final.name + ".partial")
    if kind == "symlink":
        partial.symlink_to(outside)
    else:
        os.link(outside, partial)
    diagnostic = assets.diagnose_assets(bundle(), destination)
    recovery = assets.recover_assets(
        bundle(),
        destination,
        confirm_no_active_preparation=True,
        quarantine_partial_ids=frozenset({artifact().id}),
    )
    assert not diagnostic.ok and not recovery.ok
    assert diagnostic.results[0].code == "filesystem.unsafe_file"
    assert outside.read_bytes() == b"preserve this external file"
    assert final.read_bytes() == DATA


def test_recovery_refuses_public_or_symlink_quarantine_without_moving_partial(roots):
    source, destination = roots
    partial = write_file(destination, artifact().path + ".partial", b"x" * len(DATA))
    quarantine = partial.parent / assets.QUARANTINE_NAME
    quarantine.mkdir(mode=0o755)
    report = assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
    assert not report.ok and report.error == "recovery.quarantine_not_private"
    assert partial.read_bytes() == b"x" * len(DATA)
    quarantine.rmdir()
    quarantine.symlink_to(source, target_is_directory=True)
    report = assets.recover_assets(bundle(), destination, confirm_no_active_preparation=True)
    assert not report.ok
    assert partial.read_bytes() == b"x" * len(DATA)
    assert not tuple(source.iterdir())

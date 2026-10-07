"""Synthetic-only tests for the offline, non-inference human verdict evaluator."""

import ast
import csv
import hashlib
import json
import os
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from scripts import evaluate_reid_walkthrough as evaluator

SCRIPT = Path(evaluator.__file__).resolve()
REPO = SCRIPT.parent.parent
FIELDS = [
    "query_crop_id",
    "candidate_crop_id",
    "same_person",
    "source",
    "body_score",
    "face_similarity",
    "face_reliability",
    "face_match",
    "attribute_agreement",
    "attribute_comparable_count",
    "attribute_match_count",
    "attribute_conflict_count",
    "fusion_score",
    "evidence_level",
    "decision_reason",
    "created_at",
    "updated_at",
]


def verdict(index=0, **values):
    """Use deterministic synthetic IDs; never load business feedback or images."""
    return {
        **dict.fromkeys(FIELDS, ""),
        "query_crop_id": str(uuid.UUID(int=2 * index + 1)),
        "candidate_crop_id": str(uuid.UUID(int=2 * index + 2)),
        "same_person": "true",
        "source": "search",
        **values,
    }


def feedback_file(tmp_path, rows, fields=FIELDS):
    path = tmp_path / "synthetic-verdicts.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def report_for(tmp_path, rows, **parameters):
    return evaluator.evaluate(evaluator.read_feedback(feedback_file(tmp_path, rows)), **parameters)


@pytest.mark.parametrize("value", ["true", "1", "yes", "y", " TRUE ", "Yes"])
def test_true_verdict_aliases_are_explicit(tmp_path, value):
    data = evaluator.read_feedback(feedback_file(tmp_path, [verdict(same_person=value)]))
    assert data.pairs[0].same_person is True


@pytest.mark.parametrize("value", ["false", "0", "no", "n", " FALSE ", "No"])
def test_false_verdict_aliases_are_explicit(tmp_path, value):
    data = evaluator.read_feedback(feedback_file(tmp_path, [verdict(same_person=value)]))
    assert data.pairs[0].same_person is False


@pytest.mark.parametrize("value", ["", " ", "unknown", "null", "2", "same", "different", "nan"])
def test_uncertain_verdicts_never_become_negative(tmp_path, value):
    with pytest.raises(evaluator.EvaluationError, match="^invalid_same_person$"):
        evaluator.read_feedback(feedback_file(tmp_path, [verdict(same_person=value)]))


@pytest.mark.parametrize("source", ["automatic", "body", "reid", "person_id", "pseudo_label"])
def test_explicit_auto_label_sources_are_rejected(tmp_path, source):
    with pytest.raises(evaluator.EvaluationError, match="^non_manual_label_source$"):
        evaluator.read_feedback(feedback_file(tmp_path, [verdict(source=source)]))


@pytest.mark.parametrize("source", ["", "search", "camera_link", "manual", "walkthrough"])
def test_manual_source_csv_compatibility(tmp_path, source):
    data = evaluator.read_feedback(feedback_file(tmp_path, [verdict(source=source)]))
    assert len(data.pairs) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("body_score", "nan"),
        ("body_score", "inf"),
        ("body_score", "-inf"),
        ("body_score", "1.01"),
        ("body_score", "not-a-score"),
        ("face_similarity", "nan"),
        ("face_similarity", "-1.01"),
        ("face_reliability", "1.01"),
        ("face_reliability", "-0.01"),
        ("attribute_agreement", "NaN"),
        ("attribute_agreement", "1.01"),
        ("fusion_score", "Infinity"),
        ("fusion_score", "2.1"),
        ("face_match", "unknown"),
        ("face_query_identity_verified", "unknown"),
        ("face_candidate_identity_verified", "2"),
        ("attribute_conflict_count", "-1"),
        ("attribute_conflict_count", "1.5"),
        ("attribute_comparable_count", "21"),
        ("attribute_match_count", "100"),
    ],
)
def test_evidence_is_bounded_finite_or_explicitly_missing(tmp_path, field, value):
    fields = FIELDS if field in FIELDS else [*FIELDS, field]
    with pytest.raises(evaluator.EvaluationError, match=f"^invalid_{field}$"):
        evaluator.read_feedback(feedback_file(tmp_path, [verdict(**{field: value})], fields=fields))


def test_conflicts_cannot_exceed_comparable_labels(tmp_path):
    path = feedback_file(
        tmp_path, [verdict(attribute_comparable_count="1", attribute_conflict_count="2")]
    )
    with pytest.raises(evaluator.EvaluationError, match="^invalid_attribute_counts$"):
        evaluator.read_feedback(path)


def test_no_scores_and_zero_comparable_tags_abstain(tmp_path):
    report = report_for(
        tmp_path, [verdict(attribute_comparable_count="0", attribute_conflict_count="0")]
    )
    fixed = report["fixed_threshold_evaluation"]
    assert fixed["positive_pairs"] == 1
    for modality in ("body", "face", "attributes"):
        assert fixed[modality]["coverage"] == 0
        assert fixed[modality]["unscored_pairs"] == 1
        metrics = fixed[modality]["metrics"]
        assert metrics["tp"] == metrics["tn"] == metrics["fp"] == metrics["fn"] == 0
        assert metrics["precision"] is metrics["recall"] is metrics["balanced_accuracy"] is None


def test_body_fixed_threshold_confusion_matrix(tmp_path):
    report = report_for(
        tmp_path,
        [
            verdict(0, body_score="0.9"),
            verdict(1, same_person="false", body_score="0.8"),
            verdict(2, body_score="0.2"),
            verdict(3, same_person="false", body_score="0.1"),
        ],
    )
    body = report["fixed_threshold_evaluation"]["body"]
    assert body["scored_pairs"] == 4
    assert body["coverage"] == 1
    assert body["metrics"] == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.5,
        "f1": 0.5,
        "balanced_accuracy": 0.5,
    }


def test_partial_face_coverage_and_low_reliability_do_not_become_negatives(tmp_path):
    report = report_for(
        tmp_path,
        [
            verdict(0),
            verdict(
                1,
                same_person="false",
                body_score=".8",
                face_similarity=".9",
                face_reliability=".69",
            ),
            verdict(2, body_score=".7", face_similarity=".8"),
            verdict(
                3, same_person="false", body_score=".2", face_similarity=".3", face_reliability=".9"
            ),
            verdict(4, face_similarity=".45", face_reliability=".7"),
        ],
    )
    fixed = report["fixed_threshold_evaluation"]
    assert fixed["body"]["coverage"] == 0.6
    face = fixed["face"]
    assert face["coverage"] == 0.4
    assert face["missing_score"] == face["missing_reliability"] == face["low_reliability"] == 1
    assert face["unscored_pairs"] == 3
    assert face["metrics"]["tp"] == face["metrics"]["tn"] == 1
    assert face["metrics"]["fn"] == face["metrics"]["fp"] == 0


def test_tags_are_count_diagnostics_not_live_fusion(tmp_path):
    report = report_for(
        tmp_path,
        [
            verdict(0, attribute_comparable_count="0", attribute_conflict_count="0"),
            verdict(
                1, same_person="false", attribute_comparable_count="4", attribute_conflict_count="2"
            ),
            verdict(2, attribute_comparable_count="3", attribute_conflict_count="1"),
            verdict(3, same_person="false"),
        ],
    )
    attributes = report["fixed_threshold_evaluation"]["attributes"]
    assert attributes["diagnostic_only_not_runtime_fusion"] is True
    assert attributes["coverage"] == 0.5
    assert attributes["metrics"]["tp"] == attributes["metrics"]["tn"] == 1
    assert attributes["unscored_pairs"] == 2


def test_exact_duplicates_are_deduplicated_and_dataset_has_content_identity(tmp_path):
    row = verdict(body_score=".6")
    path = feedback_file(tmp_path, [row, row])
    data = evaluator.read_feedback(path)
    assert len(data.pairs) == 1
    assert data.deduplicated_rows == 1
    assert data.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_duplicate_score_disagreement_is_rejected(tmp_path):
    path = feedback_file(tmp_path, [verdict(body_score=".6"), verdict(body_score=".7")])
    with pytest.raises(evaluator.EvaluationError, match="^conflicting_duplicate_pair$"):
        evaluator.read_feedback(path)


@pytest.mark.parametrize("reverse", [True, False])
def test_pair_truth_disagreement_is_rejected_in_either_direction(tmp_path, reverse):
    row = verdict()
    conflicting = {**row, "same_person": "false"}
    if reverse:
        conflicting.update(
            query_crop_id=row["candidate_crop_id"], candidate_crop_id=row["query_crop_id"]
        )
    with pytest.raises(evaluator.EvaluationError, match="^contradictory_pair_truth$"):
        evaluator.read_feedback(feedback_file(tmp_path, [row, conflicting]))


@pytest.mark.parametrize("bad_id", ["", "not-an-id", "123"])
def test_invalid_ids_fail_without_reprinting_values(tmp_path, bad_id):
    with pytest.raises(evaluator.EvaluationError, match="^invalid_crop_id$"):
        evaluator.read_feedback(feedback_file(tmp_path, [verdict(candidate_crop_id=bad_id)]))


def test_self_pair_is_not_an_easy_positive(tmp_path):
    row = verdict()
    row["candidate_crop_id"] = row["query_crop_id"]
    with pytest.raises(evaluator.EvaluationError, match="^self_pair$"):
        evaluator.read_feedback(feedback_file(tmp_path, [row]))


@pytest.mark.parametrize(
    ("contents", "code"),
    [
        ("", "missing_csv_header"),
        ("same_person\ntrue\n", "missing_required_columns"),
        ("query_crop_id,candidate_crop_id,same_person,same_person\n", "duplicate_csv_header"),
        ("query_crop_id,candidate_crop_id,same_person\n", "no_labelled_pairs"),
        ('query_crop_id,candidate_crop_id,same_person\n"unclosed', "invalid_csv"),
        ("query_crop_id,candidate_crop_id,same_person\none,two\n", "malformed_csv_row"),
        ("query_crop_id,candidate_crop_id,same_person\none,two,true,extra\n", "malformed_csv_row"),
    ],
)
def test_malformed_csv_fails_closed(tmp_path, contents, code):
    path = tmp_path / "invalid.csv"
    path.write_text(contents)
    with pytest.raises(evaluator.EvaluationError, match=f"^{code}$"):
        evaluator.read_feedback(path)


def test_utf8_bom_is_supported_and_invalid_encoding_rejected(tmp_path):
    path = feedback_file(tmp_path, [verdict()])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert len(evaluator.read_feedback(path).pairs) == 1
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(evaluator.EvaluationError, match="^invalid_csv_encoding$"):
        evaluator.read_feedback(path)


def test_bounded_file_and_pair_count(tmp_path, monkeypatch):
    path = feedback_file(tmp_path, [verdict(), verdict(1)])
    monkeypatch.setattr(evaluator, "MAX_PAIRS", 1)
    with pytest.raises(evaluator.EvaluationError, match="^too_many_pairs$"):
        evaluator.read_feedback(path)
    monkeypatch.setattr(evaluator, "MAX_CSV_BYTES", 10)
    with pytest.raises(evaluator.EvaluationError, match="^input_too_large$"):
        evaluator.read_feedback(path)


def test_input_symlink_rejected(tmp_path):
    path = feedback_file(tmp_path, [verdict()])
    link = tmp_path / "linked.csv"
    link.symlink_to(path)
    with pytest.raises(evaluator.EvaluationError, match="^input_link$"):
        evaluator.read_feedback(link)


def test_fifo_rejected_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "pipe.csv"
    os.mkfifo(path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["code"] == "input_not_regular"


@pytest.mark.parametrize("parameter", ["body_threshold", "face_threshold", "face_min_reliability"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_thresholds_are_rejected(tmp_path, parameter, value):
    with pytest.raises(evaluator.EvaluationError, match="^invalid_threshold_parameter$"):
        report_for(tmp_path, [verdict()], **{parameter: value})


@pytest.mark.parametrize("value", [0, 21, True, 1.5])
def test_invalid_attribute_threshold_rejected(tmp_path, value):
    with pytest.raises(evaluator.EvaluationError, match="^invalid_attribute_threshold$"):
        report_for(tmp_path, [verdict()], attribute_hard_conflicts=value)


def test_actual_cli_parameters_are_reported_not_loaded_from_production(tmp_path):
    report = report_for(
        tmp_path,
        [verdict()],
        body_threshold=0.61,
        face_threshold=0.72,
        face_min_reliability=0.81,
        attribute_hard_conflicts=3,
    )
    assert report["parameters_loaded_from_production"] is False
    assert report["parameters"] == {
        "body_threshold": 0.61,
        "face_threshold": 0.72,
        "face_min_reliability": 0.81,
        "attribute_hard_conflicts": 3,
    }
    assert report["split"] == "no_independent_holdout"
    assert report["identity_disjoint_test_proven"] is False
    assert any("Rank-1/mAP" in item for item in report["limitations"])


def test_small_realistic_counts_do_not_calibrate(tmp_path):
    rows = [
        verdict(i, same_person="true" if i < 10 else "false", body_score=".9" if i < 10 else ".1")
        for i in range(32)
    ]
    report = report_for(tmp_path, rows)
    suggested = report["training_threshold_suggestions"]["body"]
    assert suggested["positive_pairs"] == 10
    assert suggested["negative_pairs"] == 22
    assert suggested["status"] == "insufficient_scored_samples"
    assert suggested["threshold"] is None
    assert suggested["training_only"] is True
    assert report["fixed_threshold_evaluation"]["body"]["metrics"]["balanced_accuracy"] == 1


def test_sufficient_scored_pairs_only_produce_training_suggestions(tmp_path):
    rows = [
        verdict(
            i,
            same_person="true" if i < 30 else "false",
            body_score=".50005" if i < 30 else ".50004",
        )
        for i in range(90)
    ]
    suggestion = report_for(tmp_path, rows)["training_threshold_suggestions"]["body"]
    assert suggestion["status"] == "training_suggestion_not_production_calibration"
    assert suggestion["training_only"] is True
    assert 0.50004 < suggestion["threshold"] <= 0.50005
    assert suggestion["in_sample_metrics"]["balanced_accuracy"] == 1


def test_threshold_optimizer_matches_brute_force_with_negative_and_boundary_scores():
    scored = [(True, 1.0)] * 30 + [(False, -0.1)] * 20 + [(False, 0.0)] * 20 + [(False, 0.4)] * 20
    suggestion = evaluator._recommend_threshold(scored)
    threshold = suggestion["threshold"]
    expected = evaluator._metrics(
        [label for label, _ in scored], [score >= threshold for _, score in scored]
    )
    assert suggestion["in_sample_metrics"] == expected
    assert expected["balanced_accuracy"] == 1


def grouped_feedback(tmp_path, rows):
    return evaluator.read_feedback(
        feedback_file(tmp_path, rows, fields=[*FIELDS, "walk_group"]), "walk_group"
    )


def test_explicit_crop_disjoint_holdout_is_not_claimed_identity_disjoint(tmp_path):
    data = grouped_feedback(
        tmp_path,
        [
            verdict(0, body_score=".9", walk_group="private-train-name"),
            verdict(1, same_person="false", body_score=".8", walk_group="private-holdout-name"),
        ],
    )
    report = evaluator.evaluate(
        data, group_column="walk_group", holdout_groups=["private-holdout-name"]
    )
    assert report["split"] == "operator_group_crop_disjoint_holdout"
    assert report["training_fixed_threshold_evaluation"]["labelled_pairs"] == 1
    assert report["holdout_fixed_threshold_evaluation"]["body"]["metrics"]["fp"] == 1
    assert report["training_threshold_suggestions"]["body"]["negative_pairs"] == 0
    assert report["identity_disjoint_test_proven"] is False
    assert "private-" not in json.dumps(report)


def test_holdout_never_contributes_to_training_threshold_sufficiency(tmp_path):
    rows = [
        verdict(
            i,
            same_person="true" if i < 30 else "false",
            body_score=".9" if i < 30 else ".1",
            walk_group="holdout" if i in {29, 89} else "train",
        )
        for i in range(90)
    ]
    report = evaluator.evaluate(
        grouped_feedback(tmp_path, rows), group_column="walk_group", holdout_groups=["holdout"]
    )
    suggestion = report["training_threshold_suggestions"]["body"]
    assert suggestion["positive_pairs"] == 29
    assert suggestion["negative_pairs"] == 59
    assert suggestion["threshold"] is None


@pytest.mark.parametrize(
    ("group_column", "groups", "code"),
    [
        ("walk_group", [], "group_split_incomplete"),
        (None, ["test"], "group_split_incomplete"),
        ("walk_group", ["unknown"], "unknown_holdout_group"),
        ("walk_group", ["train", "test"], "empty_train_or_holdout"),
    ],
)
def test_invalid_split_requests_rejected(tmp_path, group_column, groups, code):
    data = grouped_feedback(
        tmp_path, [verdict(0, walk_group="train"), verdict(1, walk_group="test")]
    )
    with pytest.raises(evaluator.EvaluationError, match=f"^{code}$"):
        evaluator.evaluate(data, group_column=group_column, holdout_groups=groups)


def test_group_schema_must_be_explicit_and_complete(tmp_path):
    with pytest.raises(evaluator.EvaluationError, match="^missing_group_column$"):
        evaluator.read_feedback(feedback_file(tmp_path, [verdict()]), "walk_group")
    with pytest.raises(evaluator.EvaluationError, match="^missing_group_value$"):
        grouped_feedback(tmp_path, [verdict(walk_group="")])


def test_shared_crop_on_either_side_rejects_holdout(tmp_path):
    train = verdict(0, walk_group="train")
    test = verdict(1, walk_group="test", candidate_crop_id=train["query_crop_id"])
    with pytest.raises(evaluator.EvaluationError, match="^crop_leakage_between_splits$"):
        evaluator.evaluate(
            grouped_feedback(tmp_path, [train, test]),
            group_column="walk_group",
            holdout_groups=["test"],
        )


def test_report_exclusive_creation_and_private_permissions(tmp_path):
    target = tmp_path / "report.json"
    evaluator.write_report(target, {"safe": True})
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        evaluator.write_report(target, {"overwrite": True})
    assert target.read_bytes() == original


def test_report_rejects_dangling_symlink_and_parent_symlink(tmp_path):
    dangling = tmp_path / "report.json"
    dangling.symlink_to(tmp_path / "uncreated.json")
    with pytest.raises(FileExistsError):
        evaluator.write_report(dangling, {})
    assert not (tmp_path / "uncreated.json").exists()
    parent = tmp_path / "real-parent"
    parent.mkdir()
    linked = tmp_path / "linked-parent"
    linked.symlink_to(parent, target_is_directory=True)
    with pytest.raises(evaluator.EvaluationError, match="^output_parent_link$"):
        evaluator.write_report(linked / "report.json", {})
    assert not (parent / "report.json").exists()


def test_report_rejects_parent_escape_and_nonfinite_json(tmp_path):
    with pytest.raises(evaluator.EvaluationError, match="^unsafe_output_path$"):
        evaluator.write_report(tmp_path / ".." / "report.json", {})
    target = tmp_path / "report.json"
    with pytest.raises(ValueError):
        evaluator.write_report(target, {"invalid": float("nan")})
    assert not target.exists()


def test_cli_aggregate_only_and_input_unchanged(tmp_path):
    row = verdict(body_score=".7", decision_reason="PRIVATE_FREE_TEXT")
    source = feedback_file(tmp_path, [row])
    original = source.read_bytes()
    output = tmp_path / "private-result.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            str(source),
            "--output",
            str(output),
            "--body-threshold",
            ".67",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    aggregate = json.loads(result.stdout)
    assert aggregate["parameters"]["body_threshold"] == 0.67
    assert aggregate["parameters_loaded_from_production"] is False
    assert source.read_bytes() == original
    for value in (
        row["query_crop_id"],
        row["candidate_crop_id"],
        "PRIVATE_FREE_TEXT",
        str(source),
        str(output),
    ):
        assert value not in result.stdout + result.stderr + output.read_text()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text())["parameters_loaded_from_production"] is False
    repeated = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(source), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert repeated.returncode == 1
    assert json.loads(repeated.stdout) == {"ok": False, "code": "input_or_output_unavailable"}
    assert json.loads(output.read_text())["parameters"]["body_threshold"] == 0.67


def test_cli_safe_error_does_not_leak_bad_row_or_create_output(tmp_path):
    source = feedback_file(tmp_path, [verdict(same_person="PRIVATE_BAD_VERDICT")])
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(source), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"ok": False, "code": "invalid_same_person"}
    assert "PRIVATE_BAD_VERDICT" not in result.stdout + result.stderr
    assert not output.exists()


def test_cli_has_no_application_import_config_db_model_or_network_access(tmp_path):
    source = feedback_file(tmp_path, [verdict(body_score=".6")])
    # These are synthetic sentinels, not production files or credentials.
    for filename in (".env", "production.db", "model.pth"):
        (tmp_path / filename).write_text("SYNTHETIC_UNTOUCHED")
    guard = """
import builtins, os, pathlib, runpy, sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'app', 'sqlalchemy', 'torch', 'numpy', 'cv2', 'requests', 'httpx'}:
        raise RuntimeError('forbidden application dependency')
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
def audit(event, args):
    if event.startswith('socket.') or event.startswith('subprocess.'):
        raise RuntimeError('forbidden external activity')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        name = pathlib.Path(os.fsdecode(args[0])).name
        if name in {'.env', 'production.db', 'model.pth'}:
            raise RuntimeError('forbidden production-like access')
sys.addaudithook(audit)
sys.argv = [sys.argv[1], sys.argv[2], '--output', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", guard, str(SCRIPT), str(source), str(tmp_path / "safe.json")],
        cwd=tmp_path,
        env={**os.environ, "DATABASE_URL": "sqlite:///production.db", "ENV_FILE": ".env"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    for filename in (".env", "production.db", "model.pth"):
        assert (tmp_path / filename).read_text() == "SYNTHETIC_UNTOUCHED"


def test_module_imports_only_standard_library():
    tree = ast.parse(SCRIPT.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
    assert imports <= sys.stdlib_module_names


def test_gitignore_protects_private_outputs_without_hiding_source_or_synthetic_template():
    ignored = [
        "data/reports/private-evaluation.json",
        "reid-calibration-report.json",
        "reid-feedback.csv",
        "docs/overall-effect-assessment-20260908.md",
        "docs/overall-effect-assessment-20260908.docx",
    ]
    public = [
        "scripts/evaluate_reid_walkthrough.py",
        "tests/test_reid_walkthrough_evaluation.py",
        "docs/reid-walkthrough-template.csv",
    ]
    for path in ignored:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", path],
            cwd=REPO,
            capture_output=True,
            timeout=5,
        )
        assert result.returncode == 0, path
    for path in public:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", path],
            cwd=REPO,
            capture_output=True,
            timeout=5,
        )
        assert result.returncode == 1, path

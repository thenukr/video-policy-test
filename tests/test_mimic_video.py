from pathlib import Path

import pytest

from video_policy_test.backends.mimic_video import MODELS, MimicVideoPolicy, get_model


def test_registry_covers_released_libero_suites() -> None:
    assert {model.suite for model in MODELS.values()} == {
        "libero_goal",
        "libero_object",
        "libero_spatial",
    }
    assert len(MODELS) == 10


def test_object_full_paths_match_release() -> None:
    model = get_model("object_full")
    assert model.action_checkpoint.endswith("iter_000050274.pt")
    assert model.video_checkpoint.endswith("iter_000008260_fused.pt")
    assert model.statistics == "dataset_statistics/libero_object_full.json"


def test_unknown_model_is_actionable() -> None:
    with pytest.raises(ValueError, match="Unknown MimicVideo model"):
        get_model("nope")


def test_policy_validates_sampling_step_before_import(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 35"):
        MimicVideoPolicy(
            upstream_root=tmp_path,
            checkpoint_root=tmp_path,
            model_name="object_full",
            stop_after_step=36,
            artifact_dir=tmp_path,
        )


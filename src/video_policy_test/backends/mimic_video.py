from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class MimicVideoModel:
    name: str
    suite: str
    experiment_name: str
    action_checkpoint: str
    video_checkpoint: str
    statistics: str
    image_horizon: int = 5
    lowdim_horizon: int = 1


def _model(name: str, suite: str, video_iter: str, action_iter: str) -> MimicVideoModel:
    suite_short = suite.removeprefix("libero_")
    video_stem = (
        f"v2w_libero_{suite_short}_agentview_lora_rank256_"
        f"lr1.778e-04_bsz32_iter_{video_iter}_fused"
    )
    experiment = (
        f"w2a_libero_{name}_{video_stem}_"
        "lr1.000e-04_layer20_bsz128"
    )
    return MimicVideoModel(
        name=name,
        suite=suite,
        experiment_name=experiment,
        action_checkpoint=f"action_decoder/{experiment}_iter_{action_iter}.pt",
        video_checkpoint=f"video_backbone/{video_stem}.pt",
        statistics=f"dataset_statistics/libero_{name}.json",
    )


# Names and iterations are pinned to the checkpoints released by mimic-video.
MODELS: dict[str, MimicVideoModel] = {
    model.name: model
    for model in (
        _model("goal_half", "libero_goal", "000007020", "000050022"),
        _model("goal_tenth", "libero_goal", "000007020", "000040014"),
        _model("goal_one", "libero_goal", "000007020", "000019998"),
        _model("object_full", "libero_object", "000008260", "000050274"),
        _model("object_half", "libero_object", "000008260", "000030090"),
        _model("object_tenth", "libero_object", "000008260", "000039984"),
        _model("object_one", "libero_object", "000008260", "000029997"),
        _model("spatial_full", "libero_spatial", "000007540", "000051212"),
        _model("spatial_tenth", "libero_spatial", "000007540", "000030012"),
        _model("spatial_one", "libero_spatial", "000007540", "000019998"),
    )
}


def get_model(name: str) -> MimicVideoModel:
    try:
        return MODELS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown MimicVideo model {name!r}. Choose one of: {choices}") from exc


def _load_upstream_module(upstream_root: Path) -> ModuleType:
    upstream_root = upstream_root.resolve()
    model_root = upstream_root / "model"
    libero_root = upstream_root / "eval" / "libero" / "LIBERO"
    run_path = upstream_root / "eval" / "libero" / "run.py"
    for required in (model_root, libero_root, run_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing MimicVideo component: {required}. Run scripts/setup_lambda.sh first."
            )

    for path in (str(model_root), str(libero_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location("_mimic_video_libero", run_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MimicVideo evaluator from {run_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MimicVideoPolicy:
    """Adapter from MimicVideo's VAMInference to the common policy interface."""

    def __init__(
        self,
        *,
        upstream_root: Path,
        checkpoint_root: Path,
        model_name: str,
        stop_after_step: int = 0,
        execute_actions: int = 5,
        artifact_dir: Path,
    ) -> None:
        if not 0 <= stop_after_step <= 35:
            raise ValueError("stop_after_step must be between 0 and 35")
        if execute_actions < 1:
            raise ValueError("execute_actions must be positive")

        config = get_model(model_name)
        paths = {
            "video model": checkpoint_root / config.video_checkpoint,
            "action model": checkpoint_root / config.action_checkpoint,
            "statistics": checkpoint_root / config.statistics,
        }
        missing = [f"{label}: {path}" for label, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing MimicVideo checkpoints:\n"
                + "\n".join(missing)
                + "\nRun scripts/setup_lambda.sh --models "
                + model_name
            )

        upstream = _load_upstream_module(upstream_root)
        self._policy: Any = upstream.VAMInference(
            config.experiment_name,
            str(paths["video model"]),
            str(paths["action model"]),
            paths["statistics"],
            config.image_horizon,
            config.lowdim_horizon,
            stop_after_step,
            execute_actions,
            artifact_dir,
        )

    def reset(self, task_description: str) -> None:
        self._policy.reset(task_description)

    def act(self, image: Any, task_description: str, observation: dict[str, Any]) -> Any:
        return self._policy.step(image, task_description, observation)


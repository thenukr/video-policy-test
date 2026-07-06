from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .backends.mimic_video import MODELS, MimicVideoPolicy, get_model


def _default_cache() -> Path:
    return Path(os.environ.get("VIDEO_POLICY_CACHE", ".cache"))


def _parse_tasks(value: str) -> list[int]:
    try:
        tasks = sorted(set(int(part) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tasks must be comma-separated integers") from exc
    if not tasks:
        raise argparse.ArgumentTypeError("at least one task is required")
    return tasks


def _rollout(args: argparse.Namespace) -> None:
    from .libero_runner import run_libero

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache = args.cache_dir.resolve()
    model = get_model(args.model)
    suite = args.suite or model.suite
    if suite != model.suite:
        raise SystemExit(f"Model {model.name} is trained for {model.suite}, not {suite}")
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{args.model}"
    output = args.output.resolve() / suite / run_name
    policy = MimicVideoPolicy(
        upstream_root=cache / "mimic-video",
        checkpoint_root=cache / "checkpoints",
        model_name=args.model,
        stop_after_step=args.stop_after_step,
        execute_actions=args.execute_actions,
        artifact_dir=output,
        record_predictions=args.record_predictions,
    )
    run_libero(
        policy=policy,
        suite_name=suite,
        task_ids=args.tasks,
        episodes=args.episodes,
        output_dir=output,
        max_steps=args.max_steps,
        settle_steps=args.settle_steps,
        seed=args.seed,
    )
    print(f"Rollout complete. View with: video-policy view {output}")


def _doctor(args: argparse.Namespace) -> None:
    cache = args.cache_dir.resolve()
    checks = {
        "MimicVideo source": cache / "mimic-video" / "model" / "pyproject.toml",
        "MimicVideo Python": cache / "mimic-video" / "model" / ".venv" / "bin" / "python",
        "Checkpoint root": cache / "checkpoints",
    }
    failed = False
    for label, path in checks.items():
        ok = path.exists()
        failed |= not ok
        print(f"{'OK' if ok else 'MISSING':7} {label}: {path}")
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"OK      GPU: {output}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        failed = True
        print("MISSING NVIDIA GPU/driver")
    if failed:
        raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-policy")
    parser.add_argument("--cache-dir", type=Path, default=_default_cache())
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="list available policy checkpoints")
    models.set_defaults(func=lambda _: print("\n".join(
        f"{name:15} {config.suite}" for name, config in sorted(MODELS.items())
    )))

    doctor = subparsers.add_parser("doctor", help="check the GPU and local installation")
    doctor.set_defaults(func=_doctor)

    rollout = subparsers.add_parser("rollout", help="run and record LIBERO episodes")
    rollout.add_argument("--model", choices=sorted(MODELS), default="object_full")
    rollout.add_argument("--suite", choices=["libero_goal", "libero_object", "libero_spatial"])
    rollout.add_argument("--tasks", type=_parse_tasks, default=[0], help="comma-separated task IDs")
    rollout.add_argument("--episodes", type=int, default=1)
    rollout.add_argument("--max-steps", type=int)
    rollout.add_argument("--settle-steps", type=int, default=10)
    rollout.add_argument("--stop-after-step", type=int, default=0)
    rollout.add_argument("--execute-actions", type=int, default=5)
    rollout.add_argument(
        "--record-predictions",
        action="store_true",
        help="fully denoise and record predicted-vs-real action chunks",
    )
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--output", type=Path, default=Path("results"))
    rollout.set_defaults(func=_rollout)

    view = subparsers.add_parser("view", help="serve recorded MP4s in a browser gallery")
    view.add_argument("results_dir", type=Path, nargs="?", default=Path("results"))
    view.add_argument("--host", default="0.0.0.0")
    view.add_argument("--port", type=int, default=8000)
    view.set_defaults(
        func=lambda args: __import__(
            "video_policy_test.viewer", fromlist=["serve_gallery"]
        ).serve_gallery(args.results_dir, args.host, args.port)
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)

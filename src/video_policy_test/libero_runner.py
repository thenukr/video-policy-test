from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .policy import RobotPolicy

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
}
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


@dataclass(frozen=True)
class EpisodeResult:
    suite: str
    task_id: int
    episode: int
    task_description: str
    success: bool
    steps: int
    elapsed_seconds: float
    video: str
    comparison_video: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _agentview(observation: dict[str, Any]) -> np.ndarray:
    image = observation["agentview_image"][::-1, ::-1]
    if image.shape != (480, 640, 3):
        raise ValueError(f"Expected a 480x640 RGB agent view, got {image.shape}")
    return image


def _video_name(task_id: int, episode: int, success: bool) -> str:
    status = "success" if success else "failure"
    return f"task-{task_id:02d}_episode-{episode:03d}_{status}.mp4"


def _write_video(frames: Iterable[np.ndarray], path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=20) as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_comparison(
    chunks: list[tuple[np.ndarray, list[np.ndarray]]],
    path: Path,
) -> None:
    """Write temporally aligned predicted-vs-executed action chunks."""
    import cv2
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=20) as writer:
        for chunk_index, (prediction, real_frames) in enumerate(chunks):
            if len(prediction) == 0 or not real_frames:
                continue
            for real_index, real_frame in enumerate(real_frames):
                # LIBERO executes at 20 Hz; MimicVideo predicts at 10 Hz.
                prediction_index = min(real_index // 2, len(prediction) - 1)
                predicted_frame = prediction[prediction_index]
                canvas = np.concatenate((predicted_frame, real_frame), axis=1)
                header = np.full((32, canvas.shape[1], 3), 18, dtype=np.uint8)
                cv2.putText(
                    header,
                    f"FULLY DENOISED PREDICTION    chunk {chunk_index + 1}",
                    (16, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    header,
                    "REAL EXECUTION",
                    (canvas.shape[1] // 2 + 16, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )
                writer.append_data(np.concatenate((header, canvas), axis=0))


def run_libero(
    *,
    policy: RobotPolicy,
    suite_name: str,
    task_ids: list[int],
    episodes: int,
    output_dir: Path,
    max_steps: int | None = None,
    settle_steps: int = 10,
    seed: int = 0,
) -> list[EpisodeResult]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if suite_name not in SUITE_MAX_STEPS:
        raise ValueError(f"Unsupported suite {suite_name!r}; choose from {sorted(SUITE_MAX_STEPS)}")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()[suite_name]()
    invalid = [task_id for task_id in task_ids if not 0 <= task_id < suite.n_tasks]
    if invalid:
        raise ValueError(f"Invalid task IDs {invalid}; {suite_name} has tasks 0..{suite.n_tasks - 1}")

    results: list[EpisodeResult] = []
    result_path = output_dir / "episodes.jsonl"
    horizon = max_steps or SUITE_MAX_STEPS[suite_name]

    for task_id in task_ids:
        task = suite.get_task(task_id)
        description = task.language.replace("black bowl", "bowl")
        bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=480,
            camera_widths=640,
        )
        env.seed(seed)
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) == 0:
            env.close()
            raise RuntimeError(f"Task {task_id} has no initial states")

        try:
            for episode in range(episodes):
                started = time.monotonic()
                env.reset()
                observation = env.set_init_state(initial_states[episode % len(initial_states)])
                policy.reset(description)
                frames: list[np.ndarray] = []
                comparison_chunks: list[tuple[np.ndarray, list[np.ndarray]]] = []
                current_prediction: np.ndarray | None = None
                current_real_frames: list[np.ndarray] = []
                success = False
                steps_taken = 0

                for _ in range(settle_steps):
                    observation, _, done, _ = env.step(DUMMY_ACTION)
                    if done:
                        success = True
                        break

                if not success:
                    for step in range(horizon):
                        image = _agentview(observation)
                        frames.append(image)
                        action = policy.act(image, description, observation)
                        pop_prediction = getattr(policy, "pop_prediction", None)
                        prediction = pop_prediction() if pop_prediction is not None else None
                        if prediction is not None:
                            if current_prediction is not None:
                                comparison_chunks.append((current_prediction, current_real_frames))
                            current_prediction = prediction
                            current_real_frames = []
                        if current_prediction is not None:
                            current_real_frames.append(image)
                        observation, _, done, _ = env.step(action.tolist())
                        steps_taken = step + 1
                        if done:
                            success = True
                            break

                video_path = output_dir / _video_name(task_id, episode, success)
                _write_video(frames, video_path)
                if current_prediction is not None:
                    comparison_chunks.append((current_prediction, current_real_frames))
                comparison_path = (
                    video_path.with_name(video_path.stem + "_comparison.mp4")
                    if comparison_chunks
                    else None
                )
                if comparison_path is not None:
                    _write_comparison(comparison_chunks, comparison_path)
                result = EpisodeResult(
                    suite=suite_name,
                    task_id=task_id,
                    episode=episode,
                    task_description=description,
                    success=success,
                    steps=steps_taken,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    video=video_path.name,
                    comparison_video=comparison_path.name if comparison_path else None,
                )
                results.append(result)
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(result)) + "\n")
                print(
                    f"[{suite_name} task={task_id} episode={episode}] "
                    f"{'SUCCESS' if success else 'failure'} in {steps_taken} steps -> {video_path}"
                )
        finally:
            env.close()

    successes = sum(result.success for result in results)
    summary = {
        "suite": suite_name,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return results

# video-policy-test

A small, modular harness for evaluating video-generation policies as robot
policies. The first backend is
[MimicVideo](https://github.com/mimic-video/mimic-video), evaluated on LIBERO.

The benchmark loop depends only on the `RobotPolicy` interface in
`src/video_policy_test/policy.py`. Model-specific loading and inference live in
`src/video_policy_test/backends/`, so another video model can be added without
forking the LIBERO runner.

## A100 quickstart

From a fresh Lambda SSH session:

```bash
git clone https://github.com/thenukr/video-policy-test.git
cd video-policy-test
```

Install the system packages required for headless MuJoCo rendering. The script
selects the NVIDIA EGL library matching the installed driver:

```bash
./scripts/setup_system.sh
```

Install the pinned MimicVideo source, its CUDA 12.6 environment, LIBERO, and one
released checkpoint:

```bash
./scripts/setup_lambda.sh --models object_full
```

The setup downloads heavyweight, reproducible assets to `.cache/`, which is
excluded from Git, and links them into the path expected by MimicVideo. To put
them on a persistent Lambda filesystem instead:

```bash
export VIDEO_POLICY_CACHE=/path/on/persistent/storage/video-policy-cache
./scripts/setup_lambda.sh --models object_full goal_half spatial_full
```

If Hugging Face requests authentication, run `hf auth login` first. MimicVideo
guardrails are disabled for the verified LIBERO task descriptions, matching the
upstream evaluator, so the gated guardrail weights are not required.

Use the Python environment created by MimicVideo:

```bash
MV_PY=.cache/mimic-video/model/.venv/bin
$MV_PY/video-policy doctor
$MV_PY/video-policy models
```

## Roll out LIBERO

One episode on task 0 of LIBERO-Object:

```bash
.cache/mimic-video/model/.venv/bin/video-policy rollout \
  --model object_full \
  --tasks 0 \
  --episodes 1
```

Run several task IDs:

```bash
.cache/mimic-video/model/.venv/bin/video-policy rollout \
  --model spatial_full \
  --tasks 0,1,2 \
  --episodes 5
```

Each run gets its own timestamped directory under `results/`. It contains MP4s,
an append-only `episodes.jsonl`, and `summary.json`. `--stop-after-step` controls
how many of MimicVideo's 35 video denoising steps run (default `0`, matching its
fast one-forward-pass setting); `--execute-actions` controls action chunking
(default `5`).

To fully denoise the predicted future and record it beside the real execution
for every action chunk:

```bash
.cache/mimic-video/model/.venv/bin/video-policy rollout \
  --model object_full --tasks 0 --episodes 10 --record-predictions
```

Comparison videos are labeled `_comparison.mp4`. They align MimicVideo's 10 Hz
prediction with LIBERO's 20 Hz execution and show the portion corresponding to
each five-action chunk.

MimicVideo released checkpoints for `libero_goal`, `libero_object`, and
`libero_spatial`. Model and suite must match. List exact choices with
`video-policy models`.

## Watch rollouts remotely

```bash
.cache/mimic-video/model/.venv/bin/video-policy view results --port 8000
```

Open port 8000 through Lambda's firewall or forward it over SSH:

```bash
ssh -L 8000:localhost:8000 ubuntu@YOUR_INSTANCE
```

Then visit `http://localhost:8000`. The MP4 files can also be opened directly
from a remote editor or notebook.

## Reference run

The repository includes the completed LIBERO-Object task 0 run under
`results/libero_object/20260706T013340Z_object_full/`. It contains ten raw
execution videos, ten predicted-vs-executed comparison videos, per-episode
metrics, and a 70% success-rate summary.

## Notes

- The integration is pinned to MimicVideo commit
  `e3355dbc93132b576c02f920a59b4fc18a4f5906`.
- Setup applies a small A100-only dependency patch because recent `uv` versions
  otherwise resolve an upstream CUDA 12.9 build dependency while locking the
  CUDA 12.6 extra. The same patch fixes an upstream glob that would download
  every LIBERO video backbone instead of only the selected suite.
- Setup pins MuJoCo 2.3.7 because LIBERO's robosuite 1.4 integration uses its
  API, while robosuite's open-ended dependency now resolves to incompatible
  MuJoCo 3.x releases.
- A single A100 runs episodes serially to avoid model duplication and GPU OOM.
- LIBERO uses offscreen rendering; no desktop session is needed.
- Model checkpoints remain excluded from Git and are recreated by the setup
  script. The small reference run above is committed; new results remain
  ignored unless deliberately added.

Run the lightweight unit tests with:

```bash
uv run --extra dev pytest
```

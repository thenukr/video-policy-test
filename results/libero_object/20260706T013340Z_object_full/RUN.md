# LIBERO-Object task 0 reference run

- Policy: MimicVideo `object_full`
- Task: “pick up the alphabet soup and place it in the basket”
- Episodes: 10
- Successes: 7
- Success rate: 70%
- Action chunk: 5 simulator steps
- Video prediction: 35 denoising steps, recorded alongside execution

Files ending in `_comparison.mp4` show the fully denoised prediction on the
left and real execution on the right. `episodes.jsonl` contains exact outcomes,
step counts, elapsed times, and filenames.

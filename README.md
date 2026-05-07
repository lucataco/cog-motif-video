# cog-motif-video

[Cog](https://cog.run) wrapper for [Motif-Technologies/Motif-Video-2B](https://huggingface.co/Motif-Technologies/Motif-Video-2B) — a 2B-parameter text-to-video diffusion transformer with a T5Gemma2 text encoder and Wan2.1 VAE.

Deployed at: **https://replicate.com/lucataco/motif-video**

## Highlights

- Text-to-video at native 720p (1280×736) up to 121 frames, or fast 480p (832×480) for quick iteration.
- ~12.8 GB VRAM resident in BF16 — fits comfortably on L40S 48 GB.
- Apache 2.0 weights.
- Flow-matching DiT with `AdaptiveProjectedGuidance`.
- Optional `torch.compile` for ~15% speedup on repeat predictions.

## Run on Replicate

Python:
```python
import replicate

output = replicate.run(
    "lucataco/motif-video:<version>",
    input={
        "prompt": "A vibrant blue jay perches on a slender branch in a sunlit forest",
        "width": 832,
        "height": 480,
        "num_frames": 49,
        "num_inference_steps": 30,
        "guidance_scale": 8.0,
        "frame_rate": 24,
        "seed": 0,
    },
)
print(output)  # mp4 URL
```

cURL:
```bash
curl -s -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "<version>",
    "input": {
      "prompt": "A vibrant blue jay perches on a slender branch in a sunlit forest",
      "num_frames": 49,
      "num_inference_steps": 30
    }
  }'
```

## Run locally with Cog

```bash
git clone https://github.com/lucataco/cog-motif-video
cd cog-motif-video

# Quick 480p / 49-frame test (~30s on L40S after warm setup)
cog predict -i prompt="A vibrant blue jay perches on a slender branch"

# Native 720p / 5s clip
cog predict \
  -i prompt="A cat walking through tall grass at golden hour" \
  -i width=1280 -i height=736 \
  -i num_frames=121 -i num_inference_steps=50
```

Weights are auto-downloaded from a Replicate-hosted tarball (~17 GB BF16) on first run via `pget`, and cached in `./checkpoints/` for subsequent runs.

## Inputs

| Input | Type | Default | Notes |
| --- | --- | --- | --- |
| `prompt` | str | sample prompt | Text describing the scene. |
| `negative_prompt` | str | `""` | Empty falls back to the model's built-in trained negative prompt. |
| `width` | int | 832 | 832 fast 480p, 1280 for native 720p. |
| `height` | int | 480 | 480 fast 480p, 736 for native 720p. |
| `num_frames` | int | 49 | 49 ≈ 2s test, 121 ≈ 5s full clip @ 24 fps. |
| `num_inference_steps` | int | 30 | 30 is a good balance, 50 for max quality. |
| `guidance_scale` | float | 8.0 | APG guidance scale. |
| `frame_rate` | int | 24 | Output mp4 fps. |
| `enable_compile` | bool | false | `torch.compile` the transformer (~15% faster, ~90s warmup). |
| `seed` | int | 0 | 0 randomizes. |

## Implementation notes

Two non-obvious things this wrapper gets right and worth re-using:

1. **`flow_shift=15.0` on the FlowMatch scheduler.** The shipped `model_index.json` uses `FlowMatchEulerDiscreteScheduler` whose active field is `shift`. The model card recommends `flow_shift=15.0` but that field is only read by `DPMSolverMultistepScheduler`. Without the override the sigma schedule defaults to `shift=1.0` and the model produces correct subjects on top of distorted/noisy backgrounds. We rebuild the scheduler in `setup()` with `shift=15.0`.
2. **Pipeline kwargs match the upstream PR signature, not the README.** The model card documents `frame_rate=` and `use_linear_quadratic_schedule=False` as `pipe(...)` kwargs but neither exists in `MotifVideoPipeline.__call__` (the diffusers integration is the unmerged PR `waitingcheung/diffusers@feat/motif-video`). `frame_rate` is passed to `export_to_video(fps=...)` instead.

Both pitfalls are documented in the [Cog ship-open-weight-model skill](https://github.com/lucataco) used to build this repo.

## Hardware

Tested on:
- L40S 48 GB (sandbox-l40s) — recommended
- A100 80 GB

Resident VRAM at setup: ~12.8 GB BF16.

## License

This wrapper code is released under the Apache 2.0 license — see `LICENSE`.

The underlying Motif-Video-2B weights are also Apache 2.0 — see [the model card](https://huggingface.co/Motif-Technologies/Motif-Video-2B) for the canonical license text and citation requirements.

from __future__ import annotations
import os
import subprocess
import time
from typing import Any

import torch
# torch 2.8 wheel cudnn 9 vs container libcudnn 9 mismatch crashes SDPA at runtime
torch.backends.cudnn.enabled = False

from cog import BasePredictor, Input, Path

MODEL_ID = "Motif-Technologies/Motif-Video-2B"
MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/lucataco/motif-video-2b/model.tar"

# Built-in negative prompt the model was trained against.
DEFAULT_NEGATIVE_PROMPT = (
    "text overlay, graphic overlay, watermark, logo, subtitles, timestamp, "
    "broadcast graphics, UI elements, random letters, frozen pose, rigid, "
    "static expression, jerky motion, mechanical motion, discontinuous motion, "
    "flat framing, depthless, dull lighting, monotone, crushed shadows, "
    "blown-out highlights, shifting background, fading background, poor "
    "continuity, identity drift, deformation, flickering, ghosting, smearing, "
    "duplication, mutated proportions, inconsistent clothing, flat colors, "
    "desaturated, tonally compressed, poor background separation, exposure "
    "shift, uneven brightness, color balance shift"
)

WEIGHT_PATTERNS = [
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def _coerce(val, expected_type, default):
    return val if isinstance(val, expected_type) else default


def download_weights() -> None:
    start = time.time()
    print(f"[setup] pget {MODEL_URL} -> {MODEL_CACHE}")
    # pget -x refuses if destination dir already exists; clean it first.
    if os.path.exists(MODEL_CACHE):
        import shutil
        shutil.rmtree(MODEL_CACHE)
    subprocess.check_call(
        ["pget", "-x", MODEL_URL, MODEL_CACHE],
        close_fds=False,
    )
    print(f"[setup] download done in {time.time() - start:.1f}s")


class Predictor(BasePredictor):
    def setup(self) -> None:
        from diffusers import AdaptiveProjectedGuidance, MotifVideoPipeline
        from diffusers.utils import export_to_video  # noqa: F401  (validates install)

        missing = [
            p for p in WEIGHT_PATTERNS
            if not os.path.exists(os.path.join(MODEL_CACHE, p))
        ]
        if missing:
            print(f"[setup] missing {len(missing)} files, downloading from R2")
            download_weights()

        guider = AdaptiveProjectedGuidance(
            guidance_scale=8.0,
            adaptive_projected_guidance_rescale=12.0,
            adaptive_projected_guidance_momentum=0.1,
            use_original_formulation=True,
            adaptive_projected_guidance_norm_dim=[-1, -2, -4],
        )

        t0 = time.time()
        self.pipe = MotifVideoPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            guider=guider,
        ).to("cuda")

        # CRITICAL: Motif-Video 2B is trained for flow_shift=15.0 sigma schedule.
        # The shipped FlowMatchEulerDiscreteScheduler reads the `shift` field, not
        # `flow_shift` (which is a DPMSolver-specific param the upstream PR stores
        # in the same config). Without this override, shift defaults to 1.0 and
        # the model produces correct subjects on top of distorted/noisy backgrounds.
        # See model card "Recommended Settings": flow_shift=15.0.
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=15.0,
            use_dynamic_shifting=False,
        )
        print(
            f"[setup] pipeline loaded in {time.time() - t0:.1f}s, "
            f"VRAM resident: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
            f"scheduler shift=15.0"
        )

        # Track compile state so we only compile once per container.
        self._compiled = False
        self._uncompiled_transformer = self.pipe.transformer

    def _maybe_compile(self, enable: bool) -> None:
        if enable and not self._compiled:
            print("[setup] torch.compile(transformer, mode='reduce-overhead') ...")
            t0 = time.time()
            self.pipe.transformer = torch.compile(
                self._uncompiled_transformer,
                mode="reduce-overhead",
                fullgraph=False,
            )
            self._compiled = True
            print(f"[setup] compile wrapper installed in {time.time() - t0:.1f}s "
                  f"(first prediction will pay ~80-90s compile pass)")
        elif not enable and self._compiled:
            print("[setup] reverting to uncompiled transformer")
            self.pipe.transformer = self._uncompiled_transformer
            self._compiled = False

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the video to generate.",
            default=(
                "A vibrant blue jay perches gracefully on a slender branch, "
                "feathers shimmering in the soft morning light, lush forest "
                "canopy in the background with rays of sunlight filtering through."
            ),
        ),
        negative_prompt: str = Input(
            description=(
                "Negative prompt. Leave empty to use the built-in negative "
                "prompt the model was trained with (recommended)."
            ),
            default="",
        ),
        width: int = Input(
            description="Video width in pixels. 832 is fast 480p test default; 1280 is native 720p.",
            default=832, ge=320, le=1280,
        ),
        height: int = Input(
            description="Video height in pixels. 480 is fast 480p test default; 736 is native 720p.",
            default=480, ge=320, le=736,
        ),
        num_frames: int = Input(
            description="Number of frames. 49 frames @ 24 fps = ~2s test clip; 121 = ~5s full clip.",
            default=49, ge=25, le=121,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. 30 is a good quality/speed tradeoff; bump to 50 for max quality.",
            default=30, ge=10, le=80,
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale.",
            default=8.0, ge=1.0, le=15.0,
        ),
        frame_rate: int = Input(
            description="Output video frame rate (fps).",
            default=24, ge=8, le=30,
        ),
        enable_compile: bool = Input(
            description=(
                "Enable torch.compile on the transformer for ~15% faster inference. "
                "Adds ~90s overhead on the first prediction in this container; "
                "subsequent predictions reuse the compiled graph as long as resolution "
                "and frame count stay the same. Off by default."
            ),
            default=False,
        ),
        seed: int = Input(
            description="Random seed. Set 0 to randomize.",
            default=0,
        ),
    ) -> Path:
        from diffusers import AdaptiveProjectedGuidance
        from diffusers.utils import export_to_video

        prompt = _coerce(prompt, str, "")
        negative_prompt = _coerce(negative_prompt, str, "")
        if not negative_prompt.strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        width = _coerce(width, int, 832)
        height = _coerce(height, int, 480)
        num_frames = _coerce(num_frames, int, 49)
        num_inference_steps = _coerce(num_inference_steps, int, 30)
        guidance_scale = _coerce(guidance_scale, (int, float), 8.0)
        frame_rate = _coerce(frame_rate, int, 24)
        enable_compile = _coerce(enable_compile, bool, False)
        seed = _coerce(seed, int, 0)

        if seed <= 0:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"[predict] seed={seed} steps={num_inference_steps} "
              f"size={width}x{height} frames={num_frames} compile={enable_compile}")

        self._maybe_compile(enable_compile)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Rebuild guider so the user's guidance_scale is honored per-call.
        self.pipe.guider = AdaptiveProjectedGuidance(
            guidance_scale=float(guidance_scale),
            adaptive_projected_guidance_rescale=12.0,
            adaptive_projected_guidance_momentum=0.1,
            use_original_formulation=True,
            adaptive_projected_guidance_norm_dim=[-1, -2, -4],
        )

        t0 = time.time()
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
        )
        print(f"[predict] generation done in {time.time() - t0:.1f}s")

        out_path = "/tmp/output.mp4"
        export_to_video(output.frames[0], out_path, fps=int(frame_rate))
        return Path(out_path)

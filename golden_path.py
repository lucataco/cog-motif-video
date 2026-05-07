"""Golden-path test: model-card-exact settings, then progressively swap in our configs to isolate the distortion."""
import os, time
import torch
torch.backends.cudnn.enabled = False
from diffusers import AdaptiveProjectedGuidance, MotifVideoPipeline
from diffusers.utils import export_to_video

CACHE = "checkpoints"

guider = AdaptiveProjectedGuidance(
    guidance_scale=8.0,
    adaptive_projected_guidance_rescale=12.0,
    adaptive_projected_guidance_momentum=0.1,
    use_original_formulation=True,
    adaptive_projected_guidance_norm_dim=[-1, -2, -4],
)
pipe = MotifVideoPipeline.from_pretrained(CACHE, torch_dtype=torch.bfloat16, guider=guider).to("cuda")
print(f"loaded vram={torch.cuda.memory_allocated()/1e9:.2f} GB, scheduler={type(pipe.scheduler).__name__}")
print(f"scheduler config: {dict(pipe.scheduler.config)}")

NEG = (
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

PROMPT = "a cat walking on grass"

cases = [
    # (label, h, w, frames, steps, cudnn_enabled)
    ("A_lowres_30steps_cudnnoff", 480, 832, 49, 30, False),
    ("B_lowres_50steps_cudnnoff", 480, 832, 49, 50, False),
    ("D_lowres_30steps_cudnnON",  480, 832, 49, 30, True),
]

for label, h, w, nf, st, cudnn_on in cases:
    torch.backends.cudnn.enabled = cudnn_on
    g = torch.Generator(device="cuda").manual_seed(42)
    t0 = time.time()
    try:
        out = pipe(prompt=PROMPT, negative_prompt=NEG, height=h, width=w,
                   num_frames=nf, num_inference_steps=st, generator=g)
        torch.cuda.synchronize()
        path = f"outputs/golden_{label}.mp4"
        os.makedirs("outputs", exist_ok=True)
        export_to_video(out.frames[0], path, fps=24)
        print(f"OK  {label}  {time.time()-t0:.1f}s  -> {path}")
    except Exception as e:
        print(f"ERR {label}: {type(e).__name__}: {str(e)[:200]}")

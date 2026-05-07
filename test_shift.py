"""Test fix: set FlowMatchEuler shift=15.0 to match flow_shift the model was trained at."""
import os, time
import torch
torch.backends.cudnn.enabled = False
from diffusers import AdaptiveProjectedGuidance, MotifVideoPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

CACHE = "checkpoints"

guider = AdaptiveProjectedGuidance(
    guidance_scale=8.0, adaptive_projected_guidance_rescale=12.0,
    adaptive_projected_guidance_momentum=0.1, use_original_formulation=True,
    adaptive_projected_guidance_norm_dim=[-1, -2, -4],
)
pipe = MotifVideoPipeline.from_pretrained(CACHE, torch_dtype=torch.bfloat16, guider=guider).to("cuda")

# CRITICAL FIX: model is flow-matching trained at flow_shift=15.0 but the shipped
# FlowMatchEulerDiscreteScheduler reads `shift` not `flow_shift`. Default shift=1.0
# gives a flat sigma schedule -> distorted output.
pipe.scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000,
    shift=15.0,                # <-- the actual fix
    use_dynamic_shifting=False,
)
print(f"FIXED scheduler shift={pipe.scheduler.config.shift}")

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

os.makedirs("outputs", exist_ok=True)

# Test the fix at multiple shifts to find best
for shift in [1.0, 7.0, 15.0]:
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    g = torch.Generator(device="cuda").manual_seed(42)
    t0 = time.time()
    out = pipe(prompt=PROMPT, negative_prompt=NEG, height=480, width=832,
               num_frames=49, num_inference_steps=30, generator=g)
    torch.cuda.synchronize()
    path = f"outputs/cat_shift{shift}.mp4"
    export_to_video(out.frames[0], path, fps=24)
    print(f"OK shift={shift} {time.time()-t0:.1f}s -> {path}")

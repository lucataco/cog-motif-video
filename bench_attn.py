"""Benchmark attention backends on Motif-Video 2B (L40S).

Strategy: Motif uses two processors -- self-attn (no mask) and cross-attn (with mask).
sage/flash don't support attn_mask, so we only swap self-attn.
"""
import os
import time
import torch
torch.backends.cudnn.enabled = False
from diffusers import AdaptiveProjectedGuidance, MotifVideoPipeline
from diffusers.models.transformers.transformer_motif_video import (
    MotifVideoAttnProcessor2_0,
    MotifVideoCrossAttnProcessor2_0,
)

CACHE = "checkpoints"

guider = AdaptiveProjectedGuidance(
    guidance_scale=8.0,
    adaptive_projected_guidance_rescale=12.0,
    adaptive_projected_guidance_momentum=0.1,
    use_original_formulation=True,
    adaptive_projected_guidance_norm_dim=[-1, -2, -4],
)

t0 = time.time()
pipe = MotifVideoPipeline.from_pretrained(CACHE, torch_dtype=torch.bfloat16, guider=guider).to("cuda")
print(f"loaded in {time.time()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

# Inspect attention modules
n_self = n_cross = 0
for name, mod in pipe.transformer.named_modules():
    proc = getattr(mod, "processor", None)
    if isinstance(proc, MotifVideoCrossAttnProcessor2_0):
        n_cross += 1
    elif isinstance(proc, MotifVideoAttnProcessor2_0):
        n_self += 1
print(f"self-attn modules: {n_self}, cross-attn modules: {n_cross}")

PROMPT = "A vibrant blue jay perches on a slender branch in a sunlit forest"
NEG = "watermark, logo, blurry, distorted, jittery"
KW = dict(prompt=PROMPT, negative_prompt=NEG, height=480, width=832,
          num_frames=49, num_inference_steps=10)

print("warmup...")
g = torch.Generator(device="cuda").manual_seed(0)
_ = pipe(**KW, generator=g)
torch.cuda.synchronize()

def set_self_attn_backend(name):
    """Set _attention_backend ONLY on self-attn processors (cross-attn keeps native because it needs attn_mask)."""
    for mod in pipe.transformer.modules():
        proc = getattr(mod, "processor", None)
        if isinstance(proc, MotifVideoAttnProcessor2_0) and not isinstance(proc, MotifVideoCrossAttnProcessor2_0):
            proc._attention_backend = name

for backend in ["native", "sage_hub", "flash_hub"]:
    try:
        set_self_attn_backend(None if backend == "native" else backend)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        g = torch.Generator(device="cuda").manual_seed(42)
        t0 = time.time()
        out = pipe(**KW, generator=g)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"BACKEND={backend:18s}  {dt:6.1f}s total  {dt/10:5.2f}s/it  peak={peak:.2f}GB")
    except Exception as e:
        print(f"BACKEND={backend:18s}  ERROR: {type(e).__name__}: {str(e)[:140]}")
    finally:
        set_self_attn_backend(None)

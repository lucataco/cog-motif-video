"""Bench torch.compile + step reduction on Motif-Video 2B."""
import os, time
import torch
torch.backends.cudnn.enabled = False
from diffusers import AdaptiveProjectedGuidance, MotifVideoPipeline

CACHE = "checkpoints"
guider = AdaptiveProjectedGuidance(
    guidance_scale=8.0, adaptive_projected_guidance_rescale=12.0,
    adaptive_projected_guidance_momentum=0.1, use_original_formulation=True,
    adaptive_projected_guidance_norm_dim=[-1, -2, -4],
)
pipe = MotifVideoPipeline.from_pretrained(CACHE, torch_dtype=torch.bfloat16, guider=guider).to("cuda")
print(f"vram after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

PROMPT = "A vibrant blue jay perches on a slender branch in a sunlit forest"
NEG = "watermark, logo, blurry, distorted, jittery"
def kw(steps): return dict(prompt=PROMPT, negative_prompt=NEG, height=480, width=832,
                            num_frames=49, num_inference_steps=steps)

def run(label, steps):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device="cuda").manual_seed(42)
    t0 = time.time()
    pipe(**kw(steps), generator=g)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label:40s}  {dt:6.1f}s total  {dt/steps:5.2f}s/it  peak={peak:.2f}GB")

# Warmup
print("warmup...")
g = torch.Generator(device="cuda").manual_seed(0)
pipe(**kw(5), generator=g)
torch.cuda.synchronize()

# Baseline
run("baseline 10 steps", 10)

# Compile transformer
print("compiling transformer (reduce-overhead)...")
try:
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False)
    # Warmup (compile pass)
    t0 = time.time()
    pipe(**kw(3), generator=torch.Generator(device="cuda").manual_seed(1))
    torch.cuda.synchronize()
    print(f"compile warmup: {time.time()-t0:.1f}s")
    run("torch.compile reduce-overhead 10 steps", 10)
except Exception as e:
    print(f"compile failed: {type(e).__name__}: {e}")

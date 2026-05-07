Motif-Video is a 2-billion-parameter open-weight text-to-video model from Motif Technologies. Give it a sentence describing a scene and it generates a short video clip — a few seconds of motion at up to 720p, brought to life by a flow-matching diffusion transformer trained against a built-in negative prompt that suppresses common video artifacts.

The model is fast for its quality tier: a 2-second 480p test clip takes about half a minute on an L40S, and a full 5-second 720p clip lands in roughly three to five minutes. The default settings are tuned for the smaller resolution so you can iterate cheaply on a prompt before committing to the full-quality render.

Three things to know if you are using this model:

First, leave the negative prompt empty unless you really know what you are changing. Motif-Video was trained with a long, specific negative prompt covering watermarks, jerky motion, identity drift, and a dozen other common failure modes. Replicate uses that exact built-in negative prompt whenever you leave the field blank, which is almost always what you want.

Second, prompt with motion. Static descriptions ("a forest at sunset") produce static-feeling clips. Describing what is moving and how it is moving ("morning mist drifting through tall pines, sunlight slowly breaking through the canopy") gives the model something to animate.

Third, for the cheapest exploration, keep the defaults. Width 832, height 480, 49 frames, 30 inference steps. When you find a prompt you love, push width to 1280, height to 736, frames to 121, and steps to 50 for the full-quality version.

The model weights are released under the Apache 2.0 license. Source code for this Replicate wrapper lives at https://github.com/lucataco/cog-motif-video, and the original model is at https://huggingface.co/Motif-Technologies/Motif-Video-2B.
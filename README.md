
# Flux Fill ControlNet Pipeline

This implementation of Flux Fill + ControlNet Inpaint combines the fill-style masked editing of FLUX.1-Fill-dev with full ControlNet conditioning. The base image is processed through the Fill model while the ControlNet receives the corresponding conditioning input (depth, canny, pose, etc.), and both outputs are fused during denoising to guide structure and composition.

While FLUX.1-Fill-dev is designed for mask-based edits, it was not originally trained to operate jointly with ControlNet. In practice, this combined setup works well for structured inpainting tasks, though results may vary depending on the conditioning strength and the alignment between the mask and the control input.

## Diffusers Community Pipeline

This project is implemented as a **Hugging Face Diffusers community pipeline** for combining **FLUX Fill** with **ControlNet-based inpainting**. It extends the original FLUX fill pipeline by integrating ControlNet conditioning, enabling more structured and controllable masked edits using inputs such as depth, canny edges, or pose.

The implementation follows the community pipeline approach from Diffusers, where custom pipelines can be added separately and loaded using the `custom_pipeline` interface without modifying the core library. 

### Original Community Pipeline
- https://github.com/huggingface/diffusers/blob/main/examples/community/pipline_flux_fill_controlnet_Inpaint.py

### Related Documentation
- Community Pipelines README: https://github.com/huggingface/diffusers/tree/main/examples/community#flux-fill-controlnet-pipeline
- FLUX Pipelines: https://huggingface.co/docs/diffusers/api/pipelines/flux
- ControlNet in Diffusers: https://huggingface.co/docs/diffusers/api/pipelines/controlnet_flux

## Example Usage


```python
import torch
from diffusers import (
    FluxControlNetModel,
    FluxPriorReduxPipeline,
)
from diffusers.utils import load_image

# NEW PIPELINE (updated name)
from pipline_flux_fill_controlnet_Inpaint import  FluxControlNetFillInpaintPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# Models
base_model = "black-forest-labs/FLUX.1-Fill-dev"
controlnet_model = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
prior_model = "black-forest-labs/FLUX.1-Redux-dev"

# Load ControlNet
controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=dtype,
)

# Load Fill + ControlNet Pipeline
fill_pipe = FluxControlNetFillInpaintPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=dtype,
).to(device)

# OPTIONAL FP8
# fill_pipe.transformer.enable_layerwise_casting(
#     storage_dtype=torch.float8_e4m3fn,
#     compute_dtype=torch.bfloat16
# )

#  OPTIONAL Prior Redux
#pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
#    prior_model,
#    torch_dtype=dtype,
#).to(device)

# Inputs

# combined_image = load_image("person_input.png")


# 1. Prior conditioning
#prior_out = pipe_prior_redux(
#    image=cloth_image,
#    prompt=cloth_prompt,
#)

# 2. Fill Inpaint with ControlNet

# canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6).

img = load_image(r"imgs/background.jpg")
mask = load_image(r"imgs/mask.png")

control_image_depth = load_image(r"imgs/dog_depth _2.png")

result = fill_pipe(
    prompt="a dog on a bench",
    image=img,
    mask_image=mask,

    control_image=control_image_depth,
    control_mode=[2],  # union mode
    control_guidance_start=0.0,
    control_guidance_end=0.8,
    controlnet_conditioning_scale=0.9,

    height=1024,
    width=1024,

    strength=1.0,
    guidance_scale=50.0,
    num_inference_steps=60,
    max_sequence_length=512,

#    **prior_out,
)

# result.images[0].save("flux_fill_controlnet_inpaint.png")

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result.images[0].save(f"flux_fill_controlnet_inpaint_depth{timestamp}.jpg")
```

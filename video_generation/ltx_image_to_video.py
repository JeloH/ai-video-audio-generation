
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.pipelines.ltx.modeling_latent_upsampler import (
    LTXLatentUpsamplerModel,
)
from diffusers.utils import export_to_video, load_image, load_video

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
UPSCALER_ID = "a-r-r-o-w/LTX-0.9.8-Latent-Upsampler"

OUTPUT_PATH = "output.mp4"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Check your PyTorch installation.")

device = "cuda"
dtype = torch.bfloat16

# ============================================================
# 1. Load the main video-generation model
# ============================================================
print("Loading LTX video model...")

pipe = LTXConditionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)

# ============================================================
# 2. Load the latent upsampler model
# ============================================================
print("Loading latent upsampler...")

latent_upsampler = LTXLatentUpsamplerModel.from_pretrained(
    UPSCALER_ID,
    torch_dtype=dtype,
)

pipe_upsample = LTXLatentUpsamplePipeline(
    vae=pipe.vae,
    latent_upsampler=latent_upsampler,
)

pipe.to(device)
pipe_upsample.to(device)
pipe.vae.enable_tiling()

# ============================================================
# 3. Resolution helper
# ============================================================
def round_to_nearest_resolution_acceptable_by_vae(height, width):
    ratio = pipe.vae_spatial_compression_ratio
    height -= height % ratio
    width -= width % ratio
    return height, width

# ============================================================
# 4. Load the conditioning image
# ============================================================
print("Loading conditioning image...")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
)

# Convert the image into a one-frame video for conditioning.
# export_to_video normally returns a filename, which load_video reads.
conditioning_video_path = export_to_video(
    [image],
    output_video_path="conditioning.mp4",
    fps=24,
)

conditioning_video = load_video(conditioning_video_path)

condition1 = LTXVideoCondition(
    video=conditioning_video,
    frame_index=0,
)

# ============================================================
# 5. Generation settings
# ============================================================
prompt = (
    "A cute little penguin takes out a book and starts reading it. "
    "The penguin moves naturally, opens the book with its flippers, "
    "looks down at the pages, and turns a page. "
    "Smooth continuous motion, stable character appearance, "
    "cinematic lighting, detailed feathers, natural animation."
)

negative_prompt = (
    "worst quality, blurry, jittery, distorted anatomy, "
    "deformed flippers, flickering, inconsistent character, "
    "warped book, sudden scene changes"
)

expected_height = 480
expected_width = 832

downscale_factor = 2 / 3
num_frames = 256
seed = 0

# ============================================================
# 6. Generate video at lower resolution
# ============================================================
downscaled_height = int(expected_height * downscale_factor)
downscaled_width = int(expected_width * downscale_factor)

downscaled_height, downscaled_width = (
    round_to_nearest_resolution_acceptable_by_vae(
        downscaled_height,
        downscaled_width,
    )
)

print(
    f"Generating initial video at "
    f"{downscaled_width}x{downscaled_height}..."
)

generator = torch.Generator(device="cpu").manual_seed(seed)

with torch.inference_mode():
    latents = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=30,
        generator=generator,
        output_type="latent",
    ).frames

# ============================================================
# 7. Upscale the generated video latents
# ============================================================
upscaled_height = downscaled_height * 2
upscaled_width = downscaled_width * 2

print(
    f"Upscaling latents to "
    f"{upscaled_width}x{upscaled_height}..."
)

with torch.inference_mode():
    upscaled_latents = pipe_upsample(
        latents=latents,
        output_type="latent",
    ).frames

# ============================================================
# 8. Refine the upscaled video
# ============================================================
print("Refining video details...")

generator = torch.Generator(device="cpu").manual_seed(seed)

with torch.inference_mode():
    result = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=upscaled_width,
        height=upscaled_height,
        num_frames=num_frames,
        denoise_strength=0.4,
        num_inference_steps=10,
        latents=upscaled_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=generator,
        output_type="pil",
    )

video = result.frames[0]

# ============================================================
# 9. Resize to the target resolution and export
# ============================================================
print("Resizing frames and exporting video...")

video = [
    frame.resize((expected_width, expected_height))
    for frame in video
]

export_to_video(
    video,
    output_video_path=OUTPUT_PATH,
    fps=24,
)

print(f"Video saved to: {OUTPUT_PATH}")

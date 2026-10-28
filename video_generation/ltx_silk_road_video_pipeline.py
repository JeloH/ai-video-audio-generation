
# ============================================================
# ANCIENT SILK ROAD — CINEMATIC STORY VIDEO
# LTX-Video 0.9.8 + Latent Upsampler
#
# Input:
#   silk_road_refs/scene1.jpg ... scene5.jpg
#
# Output:
#   silk_road_story.mp4
#
# IMPORTANT:
# This 13B model is intended for a high-memory GPU/server.
# RTX 3050 Laptop GPU with 4 GB VRAM may not be sufficient.
# Test one short scene before generating the complete video.
# ============================================================

import gc
from pathlib import Path

import torch
from PIL import Image

from diffusers import (
    LTXConditionPipeline,
    LTXLatentUpsamplePipeline,
)
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXVideoCondition,
)
from diffusers.pipelines.ltx.modeling_latent_upsampler import (
    LTXLatentUpsamplerModel,
)
from diffusers.utils import export_to_video, load_video


# ============================================================
# 1. Configuration
# ============================================================

MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
UPSCALER_ID = "a-r-r-o-w/LTX-0.9.8-Latent-Upsampler"

REFERENCE_DIR = Path("silk_road_refs")
OUTPUT_PATH = "silk_road_story.mp4"
TEMP_DIR = Path("silk_road_temp")

TEMP_DIR.mkdir(exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check your PyTorch and GPU installation."
    )

device = "cuda"

# float16 is generally a safer choice for consumer NVIDIA GPUs.
# The 13B model can still require more memory than a 4 GB GPU has.
dtype = torch.float16

FPS = 24

# Begin with 49 frames (~2 seconds per scene).
# After a successful test, try 73 or 97 frames per scene.
NUM_FRAMES = 49

# Keep the first generation relatively small.
BASE_WIDTH = 384
BASE_HEIGHT = 224

# Upsampler doubles the latent spatial dimensions.
FINAL_WIDTH = 832
FINAL_HEIGHT = 480

SEED = 2026

# Set to True only after the initial low-resolution generation works.
USE_UPSAMPLER = True


# ============================================================
# 2. Historical scenes and narration
# ============================================================

scenes = [
    {
        "name": "01_departure_from_persia",
        "image": "scene1.jpg",
        "prompt": """
        Historically grounded cinematic reconstruction of a seventh-century
        Persian and Sogdian merchant caravan departing an ancient Persian city.
        A dozen Bactrian camels carry folded silk, woven carpets, pottery,
        leather bags and trade goods. Persian and Sogdian merchants wear
        historically inspired Central Asian robes and head coverings.
        Ancient mudbrick architecture and distant mountains.
        The caravan slowly begins moving forward at sunrise.
        Wide establishing shot, warm natural light, realistic materials,
        documentary filmmaking, subtle camera tracking, believable camel gait.
        Consistent historical atmosphere, 16:9 cinematic composition.
        No modern objects, no fantasy costumes, no text, no subtitles.
        """,
    },
    {
        "name": "02_crossing_the_desert",
        "image": "scene2.jpg",
        "prompt": """
        A Persian and Sogdian Silk Road caravan travels through a vast
        Central Asian desert. Loaded Bactrian camels walk in a long line
        across golden sand dunes. Fine sand blows across the ground.
        Merchants protect themselves from the wind with cloth head coverings.
        Distant rocky mountains shimmer in the heat.
        The camera slowly follows the caravan from a wide side angle.
        Natural camel movement, subtle wind, realistic sand and cloth,
        epic historical documentary, cinematic natural lighting.
        Seventh-century historical atmosphere, no modern objects,
        no text, no subtitles, no sudden scene changes.
        """,
    },
    {
        "name": "03_caravanserai_exchange",
        "image": "scene3.jpg",
        "prompt": """
        Inside a historically inspired Silk Road caravanserai in Central Asia,
        Persian and Sogdian merchants meet travelers from distant lands.
        Merchants carefully exchange folded silk textiles, spices,
        ceramics and small trade goods on a wooden table.
        Travelers communicate through gestures and friendly conversation.
        Camels rest in the courtyard behind them.
        The camera moves slowly through the busy courtyard.
        Respectful historical reconstruction, natural human gestures,
        realistic clothing and architecture, warm afternoon sunlight,
        cinematic documentary realism, no modern objects, no text.
        """,
    },
    {
        "name": "04_arrival_in_china",
        "image": "scene4.jpg",
        "prompt": """
        A Persian and Sogdian camel caravan arrives at a prosperous
        Tang-dynasty Chinese city inspired by Chang'an in the seventh century.
        Traditional Chinese tiled roofs, timber buildings, busy market streets
        and merchants displaying silk fabrics and imported trade goods.
        Chinese and Central Asian merchants greet one another in the market.
        The camels move slowly through the open market entrance.
        Wide cinematic establishing shot followed by gentle camera movement.
        Historically inspired architecture and costumes, realistic people,
        natural daylight, rich but believable colors, no modern objects,
        no fantasy elements, no text, no subtitles.
        """,
    },
    {
        "name": "05_human_connection",
        "image": "scene5.jpg",
        "prompt": """
        At sunset, a long Persian and Sogdian camel caravan travels
        along an ancient Silk Road route toward distant mountains.
        The sky glows with soft amber and rose-colored light.
        Merchants and camels appear as natural silhouettes against the horizon.
        The camera slowly pulls back to reveal the vast landscape
        and the caravan continuing its journey.
        Reflective, peaceful and hopeful ending about human connection,
        curiosity, cultural exchange and cooperation between civilizations.
        Historically grounded cinematic documentary, realistic motion,
        subtle wind, beautiful natural atmosphere, no modern objects,
        no text, no subtitles, no sudden scene changes.
        """,
    },
]

negative_prompt = """
low quality, blurry, excessive motion blur, flickering, jitter,
unstable camera, deformed camels, extra legs, duplicated animals,
floating objects, distorted faces, malformed hands, unnatural walking,
warped architecture, modern buildings, cars, roads with asphalt,
electric lights, modern clothing, fantasy armor, science fiction,
oversaturated colors, abrupt cuts, text, captions, watermark
"""


# ============================================================
# 3. Check the reference images
# ============================================================

for scene in scenes:
    image_path = REFERENCE_DIR / scene["image"]

    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing reference image: {image_path}\n"
            "Create five historically appropriate reference images "
            "named scene1.jpg through scene5.jpg."
        )

print("All five reference images are available.")


# ============================================================
# 4. Load the main model
# ============================================================

print("Loading LTX-Video model...")

pipe = LTXConditionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)

pipe.vae.enable_tiling()

# Offload model components where supported.
# This reduces VRAM usage but can substantially slow generation.
pipe.enable_model_cpu_offload()


# ============================================================
# 5. Load the latent upsampler
# ============================================================

pipe_upsample = None

if USE_UPSAMPLER:
    print("Loading latent upsampler...")

    latent_upsampler = LTXLatentUpsamplerModel.from_pretrained(
        UPSCALER_ID,
        torch_dtype=dtype,
    )

    pipe_upsample = LTXLatentUpsamplePipeline(
        vae=pipe.vae,
        latent_upsampler=latent_upsampler,
    )

    pipe_upsample.enable_model_cpu_offload()


# ============================================================
# 6. Resolution helper
# ============================================================

def round_to_vae_resolution(height, width):
    ratio = pipe.vae_spatial_compression_ratio

    height = (height // ratio) * ratio
    width = (width // ratio) * ratio

    if height <= 0 or width <= 0:
        raise ValueError("Resolution is too small for the VAE.")

    return height, width


height, width = round_to_vae_resolution(
    BASE_HEIGHT, BASE_WIDTH
)

print(f"Base generation resolution: {width}x{height}")


# ============================================================
# 7. Generate each scene
# ============================================================

all_frames = []

for index, scene in enumerate(scenes):
    print()
    print("=" * 65)
    print(f"Generating scene {index + 1}/{len(scenes)}")
    print(scene["name"])
    print("=" * 65)

    # Load this scene's own reference image.
    image_path = REFERENCE_DIR / scene["image"]

    image = Image.open(image_path).convert("RGB")

    # Match the reference image to the generation aspect ratio.
    image = image.resize((width, height))

    # Convert the image to a one-frame conditioning video.
    conditioning_path = TEMP_DIR / f"{scene['name']}_condition.mp4"

    export_to_video(
        [image],
        output_video_path=str(conditioning_path),
        fps=FPS,
    )

    conditioning_video = load_video(str(conditioning_path))

    condition = LTXVideoCondition(
        video=conditioning_video,
        frame_index=0,
    )

    generator = torch.Generator(device="cpu").manual_seed(
        SEED + index
    )

    # --------------------------------------------------------
    # A. Initial latent generation
    # --------------------------------------------------------

    print("Generating scene motion...")

    with torch.inference_mode():
        result = pipe(
            conditions=[condition],
            prompt=scene["prompt"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=NUM_FRAMES,
            num_inference_steps=30,
            generator=generator,
            output_type="latent",
        )

        latents = result.frames

    del result
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # B. Optional latent upscaling and refinement
    # --------------------------------------------------------

    if USE_UPSAMPLER:
        print("Upscaling latent frames...")

        with torch.inference_mode():
            upscaled = pipe_upsample(
                latents=latents,
                output_type="latent",
            )

            upscaled_latents = upscaled.frames

        del upscaled
        del latents
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        up_height, up_width = round_to_vae_resolution(
            height * 2,
            width * 2,
        )

        print("Refining upscaled video...")

        generator = torch.Generator(device="cpu").manual_seed(
            SEED + index
        )

        with torch.inference_mode():
            refined = pipe(
                conditions=[condition],
                prompt=scene["prompt"],
                negative_prompt=negative_prompt,
                width=up_width,
                height=up_height,
                num_frames=NUM_FRAMES,
                denoise_strength=0.4,
                num_inference_steps=10,
                latents=upscaled_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=generator,
                output_type="pil",
            )

            frames = refined.frames[0]

        del refined
        del upscaled_latents

    else:
        # Decode the initial latent video without upsampling.
        print("Decoding initial video...")

        with torch.inference_mode():
            frames = pipe.decode_latents(
                latents,
                decode_timestep=0.05,
            )

        del latents

    # --------------------------------------------------------
    # C. Normalize frame format and resolution
    # --------------------------------------------------------

    # The pipeline's decoded output may be a tensor or PIL frames,
    # depending on the Diffusers version.
    if isinstance(frames, torch.Tensor):
        frames = pipe.video_processor.postprocess_video(
            frames, output_type="pil"
        )[0]

    normalized_frames = []

    for frame in frames:
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)

        frame = frame.convert("RGB")
        frame = frame.resize(
            (FINAL_WIDTH, FINAL_HEIGHT),
            Image.Resampling.LANCZOS,
        )

        normalized_frames.append(frame)

    # Save an individual scene for troubleshooting.
    scene_output = TEMP_DIR / f"{scene['name']}.mp4"

    export_to_video(
        normalized_frames,
        output_video_path=str(scene_output),
        fps=FPS,
    )

    print(f"Saved scene: {scene_output}")

    all_frames.extend(normalized_frames)

    del frames
    del normalized_frames

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 8. Export the complete silent video
# ============================================================

print()
print("Combining all scenes...")

export_to_video(
    all_frames,
    output_video_path=OUTPUT_PATH,
    fps=FPS,
)

print(f"Finished! Video saved to: {OUTPUT_PATH}")
print(f"Resolution: {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"Frame rate: {FPS} FPS")
print(f"Total frames: {len(all_frames)}")
print(f"Approximate duration: {len(all_frames) / FPS:.1f} seconds")

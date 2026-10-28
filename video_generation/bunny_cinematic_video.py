
# ============================================================
# BUNNY'S HAPPY MORNING — CINEMATIC STORY VIDEO
# LTX-Video 0.9.8 + Optional Latent Upsampler
#
# Test:
#   python test3.py
#
# Input:
#   bunny_refs/scene1.jpg ... scene5.jpg
#
# Output:
#   bunny_morning_story.mp4
#
# IMPORTANT:
# First run with USE_UPSAMPLER=False.
# The 13B model can be too demanding for a 4 GB RTX 3050.
# ============================================================

import gc
import shutil
import subprocess
from pathlib import Path

import torch
from PIL import Image

from diffusers import LTXConditionPipeline

from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXVideoCondition,
)
from diffusers.utils import export_to_video, load_video


# ============================================================
# 1. Configuration
# ============================================================

MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
UPSCALER_ID = "a-r-r-o-w/LTX-0.9.8-Latent-Upsampler"

REFERENCE_DIR = Path("bunny_refs")
TEMP_DIR = Path("bunny_temp")
OUTPUT_PATH = "bunny_morning_story.mp4"

TEMP_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check your PyTorch installation."
    )

dtype = torch.float16

FPS = 24

# 97 frames / 24 FPS = approximately 4.04 seconds per scene.
# Use 49 frames for a shorter initial test if needed.
NUM_FRAMES = 97

BASE_WIDTH = 384
BASE_HEIGHT = 224

FINAL_WIDTH = 832
FINAL_HEIGHT = 480

SEED = 2026

USE_UPSAMPLER = False
TEST_FIRST_SCENE_ONLY = False

NUM_INFERENCE_STEPS = 30


# ============================================================
# 2. Short character description
# Keep prompts concise for the text encoder.
# ============================================================

CHARACTER = (
    "A cute fluffy white baby rabbit with large brown eyes, "
    "upright ears with pink inner ears, a tiny pink nose, "
    "and light-blue pajamas. Consistent character design, "
    "detailed soft fur, polished 3D children's animation, "
    "natural movement, warm pastel colors."
)


# ============================================================
# 3. Story scenes
# ============================================================

scenes = [
    {
        "name": "01_bunny_wakes_up",
        "image": "scene1.jpg",
        "prompt": (
            f"{CHARACTER} "
            "The rabbit sleeps in a small wooden bed in a cozy "
            "woodland cottage. Golden morning sunlight enters "
            "through the window. The rabbit slowly opens its "
            "eyes, blinks, lifts its head, gives a tiny sleepy "
            "yawn, and smiles. Medium shot, gentle camera push-in, "
            "peaceful atmosphere. The rabbit stays in bed. "
            "One continuous shot, no text."
        ),
    },
    {
        "name": "02_bunny_stretches",
        "image": "scene2.jpg",
        "prompt": (
            f"{CHARACTER} "
            "In the same cozy bedroom, the rabbit sits up, "
            "stretches both front paws, and raises its ears. "
            "It carefully climbs down from the bed and walks "
            "toward the open bathroom door. Full-body side view, "
            "gentle tracking camera, clear natural movements. "
            "One continuous shot, no teleporting, no text."
        ),
    },
    {
        "name": "03_bunny_prepares_toothbrush",
        "image": "scene3.jpg",
        "prompt": (
            f"{CHARACTER} "
            "In a bright pastel bathroom, the rabbit stands on "
            "a small secure stool in front of a sink and mirror. "
            "It reaches for one colorful toothbrush and carefully "
            "picks it up with its front paws. A toothpaste tube "
            "rests beside the sink. Medium shot, soft daylight, "
            "clear object interaction. Not brushing yet, no text."
        ),
    },
    {
        "name": "04_bunny_brushes_teeth",
        "image": "scene4.jpg",
        "prompt": (
            f"{CHARACTER} "
            "In the same bathroom, the rabbit gently brushes "
            "its teeth with a small toothbrush while looking "
            "in the mirror. The brush moves back and forth in "
            "small controlled motions. A tiny amount of toothpaste "
            "foam appears around the brush. Steady medium close-up, "
            "natural paw movement, bright morning light, no text."
        ),
    },
    {
        "name": "05_bunny_happy_smile",
        "image": "scene5.jpg",
        "prompt": (
            f"{CHARACTER} "
            "In the same pastel bathroom, the rabbit finishes "
            "brushing and lowers the toothbrush. It places the "
            "brush in a holder beside the sink, looks in the "
            "mirror, and smiles proudly. It turns toward the "
            "door and takes a few happy steps. Warm golden light, "
            "gentle camera pull-back, joyful ending, no text."
        ),
    },
]


negative_prompt = (
    "blurry, low quality, flickering, jitter, inconsistent rabbit, "
    "changing clothes, changing fur color, deformed face, extra limbs, "
    "extra ears, malformed paws, duplicated toothbrush, floating objects, "
    "unnatural movement, teleporting, excessive motion blur, scary face, "
    "text, subtitles, watermark"
)


# ============================================================
# 4. Validate reference images
# ============================================================

selected_scenes = (
    scenes[:1] if TEST_FIRST_SCENE_ONLY else scenes
)

for scene in selected_scenes:
    image_path = REFERENCE_DIR / scene["image"]

    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing reference image: {image_path}\n"
            "Create bunny_refs/scene1.jpg through scene5.jpg."
        )

if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "FFmpeg was not found. Install FFmpeg and add it to PATH."
    )

print(f"Preparing {len(selected_scenes)} scene(s).")


# ============================================================
# 5. Load the main model
# ============================================================

print("Loading LTX-Video model...")

pipe = LTXConditionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)

pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()


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
    BASE_HEIGHT,
    BASE_WIDTH,
)

print(f"Generation resolution: {width}x{height}")


# ============================================================
# 7. Generate each scene
# ============================================================

for index, scene in enumerate(selected_scenes):

    print("\n" + "=" * 60)
    print(f"Generating scene {index + 1}/{len(selected_scenes)}")
    print(scene["name"])
    print("=" * 60)

    image_path = REFERENCE_DIR / scene["image"]

    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    # Create a one-frame conditioning video.
    conditioning_path = (
        TEMP_DIR / f"{scene['name']}_condition.mp4"
    )

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

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(SEED + index)

    print("Generating and decoding video...")

    # IMPORTANT:
    # Request PIL frames directly. This avoids calling the
    # nonexistent pipe.decode_latents() method.
    with torch.inference_mode():
        result = pipe(
            conditions=[condition],
            prompt=scene["prompt"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            output_type="pil",
        )

    # Diffusers LTX pipeline returns a batch of frame sequences.
    frames = result.frames[0]

    del result
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Resize and normalize frames
    # --------------------------------------------------------

    normalized_frames = []

    for frame in frames:

        if isinstance(frame, Image.Image):
            pil_frame = frame
        elif isinstance(frame, torch.Tensor):
            tensor = frame.detach().float().cpu()

            # Convert channel-first images to channel-last.
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)

            # Convert normalized [-1, 1] tensors if necessary.
            if tensor.min().item() < 0:
                tensor = (tensor + 1.0) / 2.0

            array = (
                tensor.clamp(0, 1).numpy() * 255
            ).round().astype("uint8")

            pil_frame = Image.fromarray(array)
        else:
            pil_frame = Image.fromarray(frame)

        pil_frame = pil_frame.convert("RGB")
        pil_frame = pil_frame.resize(
            (FINAL_WIDTH, FINAL_HEIGHT),
            Image.Resampling.LANCZOS,
        )

        normalized_frames.append(pil_frame)

    # --------------------------------------------------------
    # Export the scene
    # --------------------------------------------------------

    scene_output = TEMP_DIR / f"{scene['name']}.mp4"

    export_to_video(
        normalized_frames,
        output_video_path=str(scene_output),
        fps=FPS,
    )

    print(f"Saved scene: {scene_output}")
    print(f"Frames: {len(normalized_frames)}")
    print(f"Duration: {len(normalized_frames) / FPS:.2f} seconds")

    del frames
    del normalized_frames
    del condition
    del conditioning_video
    del image

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 8. Combine scenes using FFmpeg
# ============================================================

scene_files = [
    (TEMP_DIR / f"{scene['name']}.mp4").resolve()
    for scene in selected_scenes
]

concat_file = TEMP_DIR / "concat_list.txt"

with open(concat_file, "w", encoding="utf-8") as file:
    for path in scene_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Generated scene not found: {path}"
            )

        safe_path = str(path).replace("'", "'\\''")
        file.write(f"file '{safe_path}'\n")


command = [
    "ffmpeg",
    "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(concat_file),
    "-vf", f"fps={FPS},scale={FINAL_WIDTH}:{FINAL_HEIGHT}",
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-an",
    OUTPUT_PATH,
]

print("\nCombining scenes...")

subprocess.run(command, check=True)


# ============================================================
# 9. Summary
# ============================================================

approx_duration = (
    len(selected_scenes) * NUM_FRAMES / FPS
)

print("\n" + "=" * 60)
print("VIDEO COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Output: {OUTPUT_PATH}")
print(f"Resolution: {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"Frame rate: {FPS} FPS")
print(f"Scenes generated: {len(selected_scenes)}")
print(f"Approximate duration: {approx_duration:.1f} seconds")
print("Audio: none; add narration and music separately.")

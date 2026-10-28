
# ============================================================
# BENNY THE BLUE MONKEY HELPS MOMMY
# LTX-Video 0.9.8 — 13-minute story generator
#
# Platform: Linux / NVIDIA GPU
#
# Input:
#   bunny_refs/scene1.jpg ... scene5.jpg
#   (Replace these with BLUE MONKEY reference images.)
#
# Output:
#   blue_monkey_helps_mommy.mp4
#
# Design:
#   15 story scenes
#   13 short clips per scene = 195 clips
#   97 frames per clip at 24 FPS
#   Final video trimmed to 13 minutes
#
# Resume support:
#   Existing clips are skipped.
#
# WARNING:
#   195 generations can take a long time.
#   This script does not generate narration or music.
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

ROOT = Path(__file__).resolve().parent

MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"

REFERENCE_DIR = ROOT / "bunny_refs"
TEMP_DIR = ROOT / "blue_monkey_temp"
CLIP_DIR = TEMP_DIR / "clips"

OUTPUT_PATH = ROOT / "blue_monkey_helps_mommy.mp4"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check your PyTorch installation."
    )

if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "FFmpeg is missing. Install it using: "
        "sudo apt update && sudo apt install -y ffmpeg"
    )

FPS = 24

# 97 frames / 24 FPS = approximately 4.04 seconds per clip.
NUM_FRAMES = 97

# 15 scenes x 13 clips x 97 frames / 24 FPS
# = approximately 788 seconds before trimming.
CLIPS_PER_SCENE = 13

TARGET_DURATION_SECONDS = 13 * 60

BASE_WIDTH = 384
BASE_HEIGHT = 224

FINAL_WIDTH = 832
FINAL_HEIGHT = 480

NUM_INFERENCE_STEPS = 30
SEED = 2026

# Start with 1 scene for a test, then set to 15.
# Existing clips will be skipped when you rerun.
SCENES_TO_GENERATE = 15

# Set True to regenerate clips even if they already exist.
REGENERATE_EXISTING_CLIPS = False


# ============================================================
# 2. Consistent character description
# ============================================================

CHARACTER = (
    "Benny is the same adorable small blue baby monkey with "
    "soft sky-blue fur, a light-blue face and belly, large "
    "expressive brown eyes, small rounded ears and childlike "
    "proportions. His mommy is an adult blue monkey with matching "
    "blue fur, a gentle face and a simple pink scarf. Maintain "
    "consistent character appearance, fur colors, facial features "
    "and proportions. High-quality cinematic 3D family animation, "
    "detailed soft fur, expressive emotions, natural movement, "
    "beautiful lighting."
)


# ============================================================
# 3. Fifteen story scenes
# ============================================================

scenes = [
    {
        "name": "01_benny_wakes_up",
        "image": "scene1.jpg",
        "prompt": (
            "Benny wakes up in his cozy jungle treehouse. Warm "
            "morning sunlight enters through the window. He slowly "
            "opens his eyes, stretches, yawns and smiles before "
            "getting out of bed. Peaceful family morning."
        ),
    },
    {
        "name": "02_mommy_is_busy",
        "image": "scene2.jpg",
        "prompt": (
            "Benny watches his mommy preparing breakfast and "
            "organizing their cozy treehouse kitchen. She looks "
            "a little tired but remains loving and patient. Benny "
            "notices how much work she is doing."
        ),
    },
    {
        "name": "03_flower_for_mommy",
        "image": "scene3.jpg",
        "prompt": (
            "Benny picks a small colorful tropical flower and "
            "carefully gives it to his mommy. She smiles with "
            "surprise, kneels down and hugs him warmly. A tender "
            "moment showing love and appreciation."
        ),
    },
    {
        "name": "04_benny_tidies_room",
        "image": "scene4.jpg",
        "prompt": (
            "Benny tidies his bedroom. He picks up colorful toys, "
            "puts them into a small basket and straightens his "
            "blanket. He works carefully and smiles proudly when "
            "the room looks neat."
        ),
    },
    {
        "name": "05_carrying_towels",
        "image": "scene5.jpg",
        "prompt": (
            "Benny carefully carries a small stack of clean towels "
            "through the wooden treehouse to help his mommy. He "
            "walks slowly, balances the towels and places them "
            "neatly in a basket."
        ),
    },
    {
        "name": "06_little_mess",
        "image": "scene1.jpg",
        "prompt": (
            "Benny accidentally knocks over a small basket of "
            "clean socks and toys. He looks surprised and a little "
            "embarrassed. His mommy approaches calmly and reassures "
            "him with a gentle smile."
        ),
    },
    {
        "name": "07_cleaning_together",
        "image": "scene2.jpg",
        "prompt": (
            "Benny and his mommy pick up the scattered items "
            "together. They sort the socks and toys, smile and "
            "laugh gently. Benny learns that mistakes can be fixed "
            "and helping can be fun."
        ),
    },
    {
        "name": "08_preparing_lunch",
        "image": "scene3.jpg",
        "prompt": (
            "Benny helps his mommy prepare lunch in their treehouse "
            "kitchen. He carefully washes colorful tropical fruit "
            "in a bowl while mommy supervises. The hot stove stays "
            "safely away from Benny. Bright, cheerful atmosphere."
        ),
    },
    {
        "name": "09_setting_the_table",
        "image": "scene4.jpg",
        "prompt": (
            "Benny sets the little dining table. He places plates, "
            "napkins and fruit carefully, then steps back to admire "
            "his work. His mommy smiles proudly. Cozy wooden kitchen."
        ),
    },
    {
        "name": "10_mommy_needs_rest",
        "image": "scene5.jpg",
        "prompt": (
            "Mommy sits on a comfortable sofa, looking tired after "
            "her work. Benny brings her a soft cushion and sits "
            "beside her. She thanks him and gently hugs him. A quiet "
            "and loving family moment."
        ),
    },
    {
        "name": "11_caring_for_garden",
        "image": "scene1.jpg",
        "prompt": (
            "Benny and his mommy water flowers and small plants "
            "outside their jungle treehouse. Benny holds a little "
            "watering can while mommy tends the garden. Butterflies "
            "fly nearby in the warm sunshine."
        ),
    },
    {
        "name": "12_thank_you_card",
        "image": "scene2.jpg",
        "prompt": (
            "Benny sits at a small wooden table making a handmade "
            "thank-you card for his mommy. He draws a large heart "
            "and colorful flowers with crayons. He concentrates "
            "carefully, then smiles at his creation. No readable text."
        ),
    },
    {
        "name": "13_special_surprise",
        "image": "scene3.jpg",
        "prompt": (
            "Benny gives his handmade heart card to his mommy. "
            "She opens it, looks touched and embraces him warmly. "
            "Benny smiles and hugs her back. Beautiful golden "
            "afternoon sunlight and expressive faces."
        ),
    },
    {
        "name": "14_reading_together",
        "image": "scene4.jpg",
        "prompt": (
            "Benny sits beside his mommy on a cozy sofa as they "
            "read a colorful picture book together. Mommy points "
            "at an illustration and Benny listens happily. The "
            "house is peaceful and tidy. Warm evening light."
        ),
    },
    {
        "name": "15_love_is_helping",
        "image": "scene5.jpg",
        "prompt": (
            "At bedtime in their cozy jungle treehouse, Benny hugs "
            "his mommy and thanks her for caring for him. She kisses "
            "his forehead and tucks him into bed. Benny closes his "
            "eyes with a peaceful smile. Moonlight enters the window. "
            "Heartwarming cinematic ending."
        ),
    },
]


# Different framing helps clips feel less repetitive.
CAMERA_VARIATIONS = [
    "Medium shot, gentle camera push-in.",
    "Wide establishing shot, slow camera movement.",
    "Close-up of expressive faces and natural emotions.",
    "Side view, gentle tracking camera.",
    "Medium-wide shot, subtle cinematic camera movement.",
    "Close-up of the character's careful actions.",
    "Over-the-shoulder view, natural movement.",
    "Eye-level camera, soft depth of field.",
    "Slow camera pull-back, peaceful atmosphere.",
    "Medium shot, subtle changes in facial expression.",
    "Wide shot with beautiful environmental details.",
    "Gentle camera pan, warm cinematic lighting.",
    "Close-up followed by a gradual wider composition.",
]


negative_prompt = (
    "blurry, low quality, flickering, jitter, inconsistent blue "
    "monkey, changing fur color, changing face, changing clothes, "
    "deformed face, extra limbs, extra fingers, duplicated "
    "characters, floating objects, unnatural movement, teleporting, "
    "scary face, text, subtitles, watermark"
)


# ============================================================
# 4. Validate references
# ============================================================

if not 1 <= SCENES_TO_GENERATE <= len(scenes):
    raise ValueError("SCENES_TO_GENERATE must be between 1 and 15.")

selected_scenes = scenes[:SCENES_TO_GENERATE]

for scene in selected_scenes:
    image_path = REFERENCE_DIR / scene["image"]

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Missing reference image: {image_path}\n"
            "Create the reference images in bunny_refs/."
        )

print(f"Scenes selected: {len(selected_scenes)}")
print(f"Clips planned: {len(selected_scenes) * CLIPS_PER_SCENE}")
print(f"Target duration: {TARGET_DURATION_SECONDS // 60} minutes")


# ============================================================
# 5. Check whether generation is needed
# ============================================================

expected_clips = [
    CLIP_DIR / f"{scene['name']}_clip_{clip_index + 1:02d}.mp4"
    for scene in selected_scenes
    for clip_index in range(CLIPS_PER_SCENE)
]

missing_clips = [
    path for path in expected_clips
    if REGENERATE_EXISTING_CLIPS or not path.is_file()
]

pipe = None

if missing_clips:
    print(f"Clips to generate: {len(missing_clips)}")
    print("Loading LTX-Video model...")

    pipe = LTXConditionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    ratio = pipe.vae_spatial_compression_ratio
    height = (BASE_HEIGHT // ratio) * ratio
    width = (BASE_WIDTH // ratio) * ratio

    if height <= 0 or width <= 0:
        raise ValueError("Base resolution is too small for the VAE.")

    print(f"Generation resolution: {width}x{height}")

else:
    print("All expected clips already exist. Skipping generation.")


# ============================================================
# 6. Generate short clips
# ============================================================

if pipe is not None:

    for scene_index, scene in enumerate(selected_scenes):

        print("\n" + "=" * 65)
        print(f"SCENE {scene_index + 1}/{len(selected_scenes)}")
        print(scene["name"])
        print("=" * 65)

        image_path = REFERENCE_DIR / scene["image"]

        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

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

        for clip_index in range(CLIPS_PER_SCENE):

            clip_path = (
                CLIP_DIR
                / f"{scene['name']}_clip_{clip_index + 1:02d}.mp4"
            )

            if clip_path.exists() and not REGENERATE_EXISTING_CLIPS:
                print(f"Skipping existing clip: {clip_path.name}")
                continue

            global_clip_number = (
                scene_index * CLIPS_PER_SCENE + clip_index + 1
            )

            total_clips = len(selected_scenes) * CLIPS_PER_SCENE

            print(
                f"\nClip {global_clip_number}/{total_clips}: "
                f"{clip_path.name}"
            )

            camera = CAMERA_VARIATIONS[
                clip_index % len(CAMERA_VARIATIONS)
            ]

            prompt = (
                f"{CHARACTER} "
                f"Story action: {scene['prompt']} "
                f"Camera and composition: {camera} "
                "Keep the same character design and environment. "
                "One coherent short shot, natural movement, "
                "no sudden scene changes, no text."
            )

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(
                SEED + scene_index * 1000 + clip_index
            )

            try:
                with torch.inference_mode():
                    result = pipe(
                        conditions=[condition],
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_frames=NUM_FRAMES,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        generator=generator,
                        output_type="pil",
                    )

                frames = result.frames[0]

                if not frames:
                    raise RuntimeError("The model returned no frames.")

                normalized_frames = []

                for frame in frames:
                    if isinstance(frame, Image.Image):
                        pil_frame = frame.convert("RGB")
                    elif isinstance(frame, torch.Tensor):
                        tensor = frame.detach().float().cpu()

                        if (
                            tensor.ndim == 3
                            and tensor.shape[0] in (1, 3, 4)
                        ):
                            tensor = tensor.permute(1, 2, 0)

                        if tensor.max().item() <= 1.0 and tensor.min().item() < 0:
                            tensor = (tensor + 1.0) / 2.0

                        array = (
                            tensor.clamp(0, 1).numpy() * 255
                        ).round().astype("uint8")

                        pil_frame = Image.fromarray(array).convert("RGB")
                    else:
                        pil_frame = Image.fromarray(frame).convert("RGB")

                    pil_frame = pil_frame.resize(
                        (FINAL_WIDTH, FINAL_HEIGHT),
                        Image.Resampling.LANCZOS,
                    )

                    normalized_frames.append(pil_frame)

                export_to_video(
                    normalized_frames,
                    output_video_path=str(clip_path),
                    fps=FPS,
                )

                print(
                    f"Saved {clip_path.name} "
                    f"({len(normalized_frames)} frames)"
                )

                del result
                del frames
                del normalized_frames

            except Exception:
                print(f"Generation failed for {clip_path.name}.")
                print(
                    "Fix the reported error and rerun the script; "
                    "completed clips will be skipped."
                )
                raise

            finally:
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del condition
        del conditioning_video
        del image

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del pipe
    pipe = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 7. Verify all clips exist
# ============================================================

missing_after_generation = [
    path for path in expected_clips if not path.is_file()
]

if missing_after_generation:
    raise FileNotFoundError(
        f"{len(missing_after_generation)} clip(s) are missing. "
        "The final video was not assembled."
    )


# ============================================================
# 8. Create FFmpeg concat list
# ============================================================

concat_file = TEMP_DIR / "concat_list.txt"

with open(concat_file, "w", encoding="utf-8") as file:
    for path in expected_clips:
        safe_path = str(path.resolve()).replace("'", "'\\''")
        file.write(f"file '{safe_path}'\n")


# ============================================================
# 9. Combine clips and trim to exactly 13 minutes
# ============================================================

command = [
    "ffmpeg",
    "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(concat_file),
    "-t", str(TARGET_DURATION_SECONDS),
    "-vf", (
        f"fps={FPS},"
        f"scale={FINAL_WIDTH}:{FINAL_HEIGHT}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={FINAL_WIDTH}:{FINAL_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    ),
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-an",
    "-movflags", "+faststart",
    str(OUTPUT_PATH),
]

print("\n" + "=" * 65)
print("Combining clips into the final video...")
print("=" * 65)

subprocess.run(command, check=True)


# ============================================================
# 10. Final report
# ============================================================

probe_command = [
    "ffprobe",
    "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    str(OUTPUT_PATH),
]

probe_result = subprocess.run(
    probe_command,
    check=True,
    capture_output=True,
    text=True,
)

duration = float(probe_result.stdout.strip())

print("\n" + "=" * 65)
print("VIDEO COMPLETED")
print("=" * 65)
print(f"Output: {OUTPUT_PATH}")
print(f"Resolution: {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"Frame rate: {FPS} FPS")
print(f"Scenes: {len(selected_scenes)}")
print(f"Clips: {len(expected_clips)}")
print(f"Actual duration: {duration:.2f} seconds")
print(f"Actual duration: {duration / 60:.2f} minutes")
print("Audio: none")
print("Next step: add narration, background music and sound effects.")

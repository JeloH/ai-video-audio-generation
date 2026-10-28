cd ~/diffusion

cat > README.md <<'EOF'
# AI Video and Audio Generation

Python pipelines for AI-powered video generation, cinematic storytelling, and multilingual speech synthesis.

## Project Structure

### Audio Generation
- `audio_generation/moss_tts_multilingual_voice_cloning.py` — MOSS-TTS multilingual speech synthesis and voice cloning.

### Video Generation
- `video_generation/ltx_image_to_video.py` — Image-to-video generation using LTX-Video.
- `video_generation/benny_longform_video_generator.py` — Long-form AI video generation.
- `video_generation/ltx_silk_road_video_pipeline.py` — Cinematic Silk Road storytelling.
- `video_generation/bunny_cinematic_video.py` — Cinematic animated storytelling.

## Requirements

Dependencies vary by pipeline. Common requirements may include:
- Python
- PyTorch
- Transformers
- Diffusers
- TorchAudio
- FFmpeg

Check each script for its specific model and dependency requirements.

## Usage

Run a pipeline from the project root. For example:

```bash
python video_generation/ltx_image_to_video.py

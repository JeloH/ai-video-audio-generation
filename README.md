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





AI Video and Audio Generation

GPU-accelerated video and audio generation pipelines using PyTorch, Hugging Face Diffusers, Transformers, and Docker.

GitHub repository: https://github.com/JeloH/ai-video-audio-generation

Features
LTX image-to-video generation
Cinematic and long-form video generation pipelines
Silk Road video generation pipeline
MOSS multilingual text-to-speech and voice cloning
Docker environment with NVIDIA GPU support
Persistent Hugging Face model cache
Requirements
Linux system
Docker Engine
Docker Compose v2
NVIDIA GPU
Compatible NVIDIA drivers
NVIDIA Container Toolkit for Docker GPU access
Sufficient disk space for model downloads and generated videos

The tested configuration uses Python 3.12, PyTorch 2.13.0+cu129, Diffusers 0.42.0, Transformers 5.15.0, and CUDA 12.9 runtime libraries.

GPU memory requirements vary by model, resolution, and generation settings. Large models can require tens of gigabytes of disk space.

1. Clone the repository

On another directory or machine, run:

bash
git clone https://github.com/JeloH/ai-video-audio-generation.git
cd ai-video-audio-generation

2. Build the Docker image
bash
sudo docker compose build


This builds the Python environment, installs the configured dependencies, and prepares the application container.

3. Test GPU access

Run this command from the repository directory:

bash
sudo docker compose run --rm ai-video python -c \
"import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"


Expected output on a machine with two H100 GPUs:

text
PyTorch: 2.13.0+cu129
CUDA available: True
GPUs: ['NVIDIA H100 NVL', 'NVIDIA H100 NVL']


The GPU names will differ depending on your hardware. If CUDA is unavailable, check your NVIDIA driver, NVIDIA Container Toolkit, and Docker GPU configuration.

4. Generate a test video

Create directories for inputs and outputs:

bash
mkdir -p inputs outputs


Run the LTX image-to-video pipeline:

bash
sudo docker

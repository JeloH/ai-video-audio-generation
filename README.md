cd ~/diffusion

cat > README.md <<'EOF'
# AI Video and Audio Generation

GPU-accelerated AI video generation, cinematic storytelling, and multilingual speech synthesis using PyTorch, Hugging Face Diffusers, Transformers, and Docker.

**GitHub Repository:** [JeloH/ai-video-audio-generation](https://github.com/JeloH/ai-video-audio-generation)

## Features

- LTX image-to-video generation
- Cinematic and long-form video generation
- Silk Road cinematic storytelling
- Animated cinematic storytelling
- MOSS-TTS multilingual speech synthesis and voice cloning
- Docker-based environment with NVIDIA GPU acceleration
- Persistent model caching

## Project Structure

\`\`\`text
ai-video-audio-generation/
├── audio_generation/
│   └── moss_tts_multilingual_voice_cloning.py
├── video_generation/
│   ├── ltx_image_to_video.py
│   ├── benny_longform_video_generator.py
│   ├── bunny_cinematic_video.py
│   └── ltx_silk_road_video_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
\`\`\`

## Requirements

- Linux
- Docker Engine and Docker Compose v2
- NVIDIA GPU and compatible NVIDIA driver
- NVIDIA Container Toolkit
- Sufficient disk space for model weights, caches, and generated videos

Tested configuration: Python 3.12, PyTorch 2.13.0+cu129, Diffusers 0.42.0, Transformers 5.15.0, TorchAudio 2.11.0+cu129, and CUDA 12.9 runtime libraries.

Hardware requirements vary by model and generation settings. Large video models can require tens of gigabytes of disk space and substantial GPU memory.

## Quick Start: Docker

### 1. Clone the repository

\`\`\`bash
git clone https://github.com/JeloH/ai-video-audio-generation.git
cd ai-video-audio-generation
\`\`\`

### 2. Build the Docker image

\`\`\`bash
sudo docker compose build
\`\`\`

### 3. Verify GPU access

\`\`\`bash
sudo docker compose run --rm ai-video python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
\`\`\`

Expected output on a two-H100 server:

\`\`\`text
PyTorch: 2.13.0+cu129
CUDA available: True
GPUs: ['NVIDIA H100 NVL', 'NVIDIA H100 NVL']
\`\`\`

GPU names vary by system. If CUDA is unavailable, check the NVIDIA driver and NVIDIA Container Toolkit.

### 4. Generate a test video

\`\`\`bash
mkdir -p inputs outputs

sudo docker compose run --rm \\
  --workdir /outputs \\
  ai-video \\
  python /app/video_generation/ltx_image_to_video.py
\`\`\`

The first run downloads the required model weights unless they are already cached. The script normally reports `output.mp4` when generation completes.

**Output persistence:** Check the volume mappings in `docker-compose.yml` to determine where generated videos are stored on the host. Files inside a removed container may be lost if the output directory is not mounted.

## Available Pipelines

### Video Generation

| Script | Description |
|---|---|
| `ltx_image_to_video.py` | LTX image-to-video generation |
| `benny_longform_video_generator.py` | Long-form AI video generation |
| `ltx_silk_road_video_pipeline.py` | Silk Road cinematic storytelling |
| `bunny_cinematic_video.py` | Cinematic animated storytelling |

Run another video pipeline, for example:

\`\`\`bash
sudo docker compose run --rm \\
  --workdir /outputs \\
  ai-video \\
  python /app/video_generation/bunny_cinematic_video.py
\`\`\`

Review each script's configuration for required assets, model settings, and output paths.

### Audio Generation

MOSS-TTS multilingual speech synthesis and voice cloning:

\`\`\`bash
sudo docker compose run --rm ai-video \\
  python /app/audio_generation/moss_tts_multilingual_voice_cloning.py
\`\`\`

Review the script configuration and provide any required reference audio or other inputs.

## Running Without Docker

Install dependencies in a compatible Python environment:

\`\`\`bash
python -m pip install -r requirements.txt
python video_generation/ltx_image_to_video.py
\`\`\`

Local execution may require additional dependencies or configuration.

## Model Downloads and Caching

- Model weights download from Hugging Face when required.
- Initial downloads can be large.
- Persistent Docker cache volumes can avoid repeated downloads.
- Some models require authentication or acceptance of their license terms.

Provide any required Hugging Face token securely. Never commit tokens or credentials to GitHub.

## Troubleshooting

### GPU access

\`\`\`bash
nvidia-smi

sudo docker run --rm --gpus all \\
  nvidia/cuda:12.9.1-base-ubuntu24.04 \\
  nvidia-smi
\`\`\`

### Dependency errors

\`\`\`bash
sudo docker compose build --no-cache
\`\`\`

### Disk space

\`\`\`bash
df -h
\`\`\`

Model weights, caches, and generated videos can require substantial storage.

### Missing output files

Check the script's output path and the volume mappings in `docker-compose.yml`. Files in a removed container may not persist on the host.

## Security and Licensing

- Never commit API keys, passwords, tokens, or private credentials.
- Check each model's license before redistribution or commercial use.
- Model weights may have separate usage restrictions.
- Use voice cloning only with appropriate authorization and consent.

## Contributing

Issues, bug reports, and pull requests are welcome. When reporting a problem, include the relevant error message, operating system, GPU model, and Docker version.
EOF

git diff --check
git add README.md
git diff --cached --check
git commit -m "Improve README documentation and Docker instructions"
git push origin main

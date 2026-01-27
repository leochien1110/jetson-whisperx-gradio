# WhisperX Jetson Gradio UI

A GPU-accelerated web UI for OpenAI's WhisperX, optimized for NVIDIA Jetson (Orin/Xavier) devices.

## Features
- **Fast Transcription**: Uses `faster-whisper` and `CTranslate2` with CUDA acceleration.
- **Diarization**: Speaker identification using `pyannote.audio`.
- **Large Models**: Supports `large-v3-turbo` model.
- **Jetson Optimized**: Built on top of `dustynv/l4t-pytorch` with custom compiled CTranslate2.
- **System Service**: Auto-starts on boot.

## Setup & Installation

### 1. Prerequisites
- NVIDIA Jetson Orin / Xavier
- JetPack 6.x (L4T R36.x)
- Docker & Docker Compose

### 2. Build the Container
The Dockerfile is configured to build all necessary dependencies, including CTranslate2 (with CUDA) and patched Pyannote libraries.

```bash
docker compose build
```

*Note: The first build will take significant time (15-30 mins) to compile CTranslate2.*

### 3. Run the Service
```bash
docker compose up -d
```

### 4. Access the UI
Open your browser and navigate to:
**http://<JETSON_IP>:7861**

## Configuration

- **Models**: Select from `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`.
- **Language**: Supports multiple languages (zh, en, ja, ko, etc).
- **Diarization**: Enable to separate speakers (requires valid Hugging Face Token).

## Troubleshooting

- **"CTranslate2 not compiled with CUDA"**: Ensure you rebuilt the image using the provided Dockerfile which compiles it from source.
- **"hf_hub_download unexpected keyword"**: The Dockerfile includes patches for `pyannote.audio` compatibility. Rebuild the image.
- **"numpy conflicts"**: The Dockerfile explicitly pins `numpy<2.0` to ensure stability.

## Systemd Service (Auto-Start)
To enable auto-start on boot:
```bash
sudo ./install-service.sh
```

import torch
import torch.serialization
import functools
import typing
import collections
import omegaconf
import gc
import logging
import warnings
import os
from datetime import timedelta

# --- SUPPRESS HARMLESS WARNINGS ---
# Suppress specific UserWarnings and informational messages for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Model was trained with.*")
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing`.*")
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0.*")

# Suppress Lightning and ONNX Runtime logs
os.environ["PYTORCH_LIGHTNING_LOG_LEVEL"] = "ERROR"
os.environ["ORT_LOGGING_LEVEL"] = "3" 
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Explicitly set TF32 to avoid ReproducibilityWarning and potentially improve Orin performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- CRITICAL FIX FOR PYTORCH 2.6+ WEIGHT LOADING ---
# PyTorch 2.6+ defaults to weights_only=True, which breaks loading of many
# models (Whisper, Pyannote) because they use common Python types not in the
# default allowlist. We force weights_only=False for these trusted models.

original_load = torch.load

@functools.wraps(original_load)
def patched_load(*args, **kwargs):
    # Force weights_only=False to allow loading all types in trusted models
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

# Patch both the main entry point and the serialization module's entry point
torch.load = patched_load
torch.serialization.load = patched_load

# Also add common types to the safe globals as a secondary defense
safe_globals = [
    list, dict, set, tuple, str, int, float, bool,
    typing.Any, typing.Dict, typing.List, typing.Tuple, typing.Optional, typing.Union,
    collections.defaultdict, collections.OrderedDict, collections.Counter, collections.deque,
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.nodes.AnyNode,
    omegaconf.base.ContainerMetadata,
]
# Add all omegaconf node types
for name in dir(omegaconf.nodes):
    attr = getattr(omegaconf.nodes, name)
    if isinstance(attr, type):
        safe_globals.append(attr)

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals(safe_globals)

# Now it's safe to import external libraries
import gradio as gr
import whisperx

# Hugging Face Token (Read from environment variable HF_TOKEN)
HF_TOKEN = os.getenv("HF_TOKEN")

def format_timestamp(seconds: float):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def write_srt(segments, diarization=False):
    srt_content = ""
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        speaker = f"[{seg.get('speaker', 'Unknown')}] " if diarization and 'speaker' in seg else ""
        text = seg['text'].strip()
        srt_content += f"{i}\n{start} --> {end}\n{speaker}{text}\n\n"
    return srt_content

def transcribe(audio_files, model_name, language, diarization):
    try:
        if audio_files is None:
            return "No audio files uploaded.", None
            
        if not isinstance(audio_files, list):
            audio_files = [audio_files]
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Device: {device}")
        print(f"Loading model: {model_name}...")
        
        # Load model once for all files
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
        
        # Load alignment model once if language is specified and not auto
        model_a, metadata = None, None
        if language != "auto":
            print(f"Pre-loading alignment model for {language}...")
            model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        
        all_full_texts = []
        all_srt_files = []
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        for audio_file in audio_files:
            audio_basename = os.path.basename(audio_file)
            print(f"\n--- Processing: {audio_basename} ---")
            
            # Load audio
            audio = whisperx.load_audio(audio_file)
            
            # Transcription
            print("Starting transcription...")
            result = model.transcribe(audio, batch_size=16)
            
            # Alignment
            print("Aligning...")
            curr_language = result["language"]
            if model_a is None or (language == "auto"):
                # Load or switch alignment model if needed
                model_a, metadata = whisperx.load_align_model(language_code=curr_language, device=device)
            
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            # Diarization
            if diarization:
                print("Diarizing...")
                # Modern huggingface-hub uses 'token' instead of 'use_auth_token'
                diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Process results
            file_text = f"FILE: {audio_basename}\n" + "="*len(audio_basename) + "\n"
            for seg in result['segments']:
                speaker = f"[{seg.get('speaker', 'Unknown')}] " if diarization else ""
                file_text += f"{speaker}{seg['text']}\n"
            
            all_full_texts.append(file_text)
            
            # Save SRT
            srt_output = write_srt(result['segments'], diarization)
            audio_name = os.path.splitext(audio_basename)[0]
            srt_path = os.path.join(output_dir, f"{audio_name}.srt")
            
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_output)
            
            all_srt_files.append(srt_path)
            print(f"Saved: {srt_path}")
            
        # Combine texts for display
        combined_text = "\n\n".join(all_full_texts)
            
        # Cleanup
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return combined_text, all_srt_files
        
    except Exception as e:
        return f"Error: {str(e)}", None

def get_jetson_stats():
    """Get Jetson GPU stats including utilization, power, and VRAM usage."""
    try:
        import subprocess
        import re
        
        # Run tegrastats once to get current stats
        result = subprocess.run(['tegrastats', '--interval', '100'], 
                              capture_output=True, text=True, timeout=0.5)
        output = result.stdout
        
        stats = {
            "gpu_util": "N/A",
            "gpu_freq": "N/A", 
            "vram_used": "N/A",
            "vram_total": "N/A",
            "power_total": "N/A",
            "power_gpu": "N/A",
            "temp_gpu": "N/A"
        }
        
        if not output:
            return stats
            
        # GPU utilization (GR3D_FREQ)
        gr3d_match = re.search(r'GR3D_FREQ\s+(\d+)%', output)
        if gr3d_match:
            stats["gpu_util"] = f"{gr3d_match.group(1)}%"
        
        # GPU frequency
        gr3d_freq_match = re.search(r'GR3D_FREQ\s+\d+%@(\d+)', output)
        if gr3d_freq_match:
            stats["gpu_freq"] = f"{gr3d_freq_match.group(1)} MHz"
        
        # VRAM usage (Unified memory on Jetson)
        ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', output)
        if ram_match:
            stats["vram_used"] = f"{ram_match.group(1)} MB"
            stats["vram_total"] = f"{ram_match.group(2)} MB"
        
        # Total Power (VDD_IN in mW)
        power_match = re.search(r'VDD_IN\s+(\d+)/(\d+)', output)
        if power_match:
            power_mw = int(power_match.group(1))
            stats["power_total"] = f"{power_mw/1000:.2f} W"
        
        # GPU+CPU Power (VDD_CPU_GPU_CV)
        gpu_power_match = re.search(r'VDD_CPU_GPU_CV\s+(\d+)/(\d+)', output)
        if gpu_power_match:
            gpu_power_mw = int(gpu_power_match.group(1))
            stats["power_gpu"] = f"{gpu_power_mw/1000:.2f} W"
        
        # GPU Temperature
        temp_match = re.search(r'GPU@([\d.]+)C', output)
        if temp_match:
            stats["temp_gpu"] = f"{temp_match.group(1)}°C"
        
        return stats
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # Fallback to PyTorch CUDA info
        try:
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
                return {
                    "gpu_util": "N/A",
                    "gpu_freq": "N/A",
                    "vram_used": f"{mem_allocated:.0f} MB",
                    "vram_total": f"{mem_reserved:.0f} MB",
                    "power_total": "N/A",
                    "power_gpu": "N/A",
                    "temp_gpu": "N/A"
                }
        except:
            pass
        
        return {
            "gpu_util": "Error",
            "gpu_freq": "Error",
            "vram_used": "Error",
            "vram_total": "Error",
            "power_total": "Error",
            "power_gpu": "Error",
            "temp_gpu": "Error"
        }

def format_stats_display():
    """Format Jetson stats for display in Gradio."""
    stats = get_jetson_stats()
    
    display_text = f"""**🖥️ Jetson GPU Monitor**
    
📊 **GPU Utilization:** {stats['gpu_util']}
⚡ **GPU Frequency:** {stats['gpu_freq']}
🌡️ **GPU Temperature:** {stats['temp_gpu']}

💾 **VRAM Usage:** {stats['vram_used']} / {stats['vram_total']}
⚡ **Total Power:** {stats['power_total']}
🔋 **CPU+GPU Power:** {stats['power_gpu']}
"""
    return display_text

# Gradio UI components
with gr.Blocks(title="WhisperX Jetson UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ WhisperX Batch Transcription UI")
    gr.Markdown("GPU-accelerated transcription for **multiple files** with speaker diarization.")
    
    # GPU Stats Panel
    with gr.Row():
        gpu_stats = gr.Markdown(value=format_stats_display(), every=2)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.File(file_count="multiple", label="Upload Audio Files", file_types=["audio", "video"])
            model_select = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
                value="large-v3-turbo",
                label="Model Size"
            )
            lang_select = gr.Dropdown(
                choices=["zh", "en", "ja", "ko", "auto"],
                value="zh",
                label="Primary Language"
            )
            diarize_toggle = gr.Checkbox(label="Enable Speaker Diarization", value=True)
            run_btn = gr.Button("🚀 Transcribe All", variant="primary")
            
        with gr.Column():
            text_output = gr.Textbox(label="Combined Results", lines=20)
            file_output = gr.File(label="Download All SRT Files")

    run_btn.click(
        fn=transcribe,
        inputs=[audio_input, model_select, lang_select, diarize_toggle],
        outputs=[text_output, file_output]
    )
    
    # Auto-refresh GPU stats every 2 seconds
    timer = gr.Timer(value=2)
    timer.tick(fn=format_stats_display, outputs=gpu_stats)

if __name__ == "__main__":
    # Check CUDA status on startup
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    demo.launch(server_name="0.0.0.0", server_port=7861)

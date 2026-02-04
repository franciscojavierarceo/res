#!/usr/bin/env python3
"""
Helper script to start vLLM with common models.
"""

import subprocess
import sys
import argparse


COMMON_MODELS = {
    "gpt2": "openai-community/gpt2",
    "dialogpt": "microsoft/DialoGPT-medium",
    "phi": "microsoft/phi-2",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gpt-oss-20b": "microsoft/DialoGPT-large",  # placeholder, update with actual model
}


def start_vllm(model_name: str, port: int = 8001, gpu: bool = False):
    """Start vLLM server with specified model."""

    if model_name in COMMON_MODELS:
        model_path = COMMON_MODELS[model_name]
    else:
        model_path = model_name  # assume it's a full model path

    print(f"🚀 Starting vLLM server with model: {model_path}")
    print(f"📡 Port: {port}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", "0.0.0.0"
    ]

    # Add GPU-specific settings
    if gpu:
        print("⚡ GPU mode enabled")
        cmd.extend([
            "--tensor-parallel-size", "1",  # adjust based on your GPU setup
        ])
    else:
        print("💻 CPU mode (slower)")
        cmd.extend([
            "--enforce-eager",  # helps with CPU inference
        ])

    print(f"🔧 Command: {' '.join(cmd)}")
    print("\n" + "="*50)
    print("vLLM will start below. Press Ctrl+C to stop.")
    print("="*50 + "\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 vLLM stopped")


def main():
    parser = argparse.ArgumentParser(description="Start vLLM server")
    parser.add_argument(
        "model",
        nargs="?",
        default="dialogpt",
        help=f"Model to use. Options: {', '.join(COMMON_MODELS.keys())} or full model path"
    )
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU mode")
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        print("Available model shortcuts:")
        for key, value in COMMON_MODELS.items():
            print(f"  {key:10} -> {value}")
        return

    start_vllm(args.model, args.port, args.gpu)


if __name__ == "__main__":
    main()
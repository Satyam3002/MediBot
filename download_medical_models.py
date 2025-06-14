#!/usr/bin/env python3
"""
Skin Model Download Script
Downloads and saves a skin disease AI model
"""

import os
import sys

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"📁 Created/verified directory: {path}")
    return path

def download_model(model_name, save_path, model_type="model"):
    print(f"🔄 Downloading {model_type}: {model_name}...")
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoProcessor, AutoModelForImageTextToText
        # Try to use the right processor/model class for multimodal models
        if model_name == "google/derm-foundation":
            # This is a TensorFlow/Keras model, not directly supported by transformers
            from huggingface_hub import from_pretrained_keras
            from pathlib import Path
            keras_model = from_pretrained_keras(model_name)
            keras_model.save(Path(save_path) / "keras_model.h5")
            print(f"✅ {model_type} downloaded successfully: {model_name}")
            print(f"   Model card: https://huggingface.co/{model_name}")
            return True, model_name
        elif model_name == "Eraly-ml/Skin-AI":
            # Download the PyTorch scripted model file
            import requests
            url = "https://huggingface.co/Eraly-ml/Skin-AI/resolve/main/skinconvnext_scripted.pt"
            local_path = os.path.join(save_path, "skinconvnext_scripted.pt")
            r = requests.get(url)
            with open(local_path, "wb") as f:
                f.write(r.content)
            print(f"✅ {model_type} downloaded successfully: {model_name}")
            print(f"   Model card: https://huggingface.co/{model_name}")
            return True, model_name
        elif model_name == "Brucze-wayne/DermaAI":
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForImageTextToText.from_pretrained(model_name)
            processor.save_pretrained(save_path)
            model.save_pretrained(save_path)
            print(f"✅ {model_type} downloaded successfully: {model_name}")
            print(f"   Model card: https://huggingface.co/{model_name}")
            return True, model_name
        else:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            processor.save_pretrained(save_path)
            model.save_pretrained(save_path)
            print(f"✅ {model_type} downloaded successfully: {model_name}")
            print(f"   Model card: https://huggingface.co/{model_name}")
            return True, model_name
    except Exception as e:
        msg = str(e)
        print(f"❌ Failed to download {model_name}: {msg[:200]}...")
        if "must accept" in msg or "gated repo" in msg or "review and agree" in msg:
            print(f"⚠️  You may need to log in to Hugging Face and accept the terms for this model: https://huggingface.co/{model_name}")
        return False, None

def main():
    print("🚀 Starting skin model download...")
    print("=" * 50)
    skin_save_path = create_directory("./backend/saved_skin_model")

    # List of specialized skin disease models
    skin_models = [
        "google/derm-foundation",        # Foundation model for dermatology (embeddings)
        "Eraly-ml/Skin-AI",             # Direct classifier, PyTorch scripted
        "Brucze-wayne/DermaAI",         # Image-to-text, SkinCAP
        "mattmdjaga/dermnet",           # DermNet trained model
        "mattmdjaga/effnetb0-dermnet",  # EfficientNet trained on DermNet
        "mattmdjaga/vit-dermnet"        # ViT trained on DermNet
    ]
    
    skin_success = False
    skin_model_used = None
    for model_name in skin_models:
        success, used_model = download_model(model_name, skin_save_path, "Skin model")
        if success:
            skin_success = True
            skin_model_used = used_model
            break

    print("\n" + "=" * 50)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 50)
    if skin_success:
        print(f"✅ Skin model: {skin_model_used}")
        print(f"   Model card: https://huggingface.co/{skin_model_used}")
        print(f"   Saved to: {skin_save_path}")
        print("\n🎉 Skin model downloaded successfully!")
    else:
        print("❌ All skin models failed to download")
        print(f"   Directory: {skin_save_path}")
        print("\n💥 Download failed - check your internet connection, model availability, or terms acceptance and try again")

if __name__ == "__main__":
    main()
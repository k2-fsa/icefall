#!/usr/bin/env python3
"""
Script to upload icefall conformer CTC model to Hugging Face Hub
"""

import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any
import json
import shutil

# Hugging Face imports
try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    exit(1)

def create_model_card(model_info: Dict[str, Any]) -> str:
    """Create a model card for the Hugging Face model"""
    
    model_card = f"""---
language: en
license: apache-2.0
tags:
- speech
- audio
- automatic-speech-recognition
- icefall
- conformer
- ctc
library_name: icefall
datasets:
- librispeech_asr
metrics:
- wer
---

# {model_info['model_name']}

This is a Conformer CTC model trained with icefall on LibriSpeech dataset.

## Model Description

- **Architecture**: Conformer with CTC loss
- **Training Framework**: icefall
- **Dataset**: LibriSpeech ASR
- **Language**: English
- **Sample Rate**: 16kHz

## Model Details

- **Model Size**: {model_info.get('num_params', 'Unknown')} parameters
- **Feature Dimension**: {model_info.get('feature_dim', 80)}
- **Attention Dimension**: {model_info.get('attention_dim', 256)}
- **Number of Heads**: {model_info.get('nhead', 4)}
- **Subsampling Factor**: {model_info.get('subsampling_factor', 4)}

## Training Information

- **Best Valid Loss**: {model_info.get('best_valid_loss', 'Unknown')}
- **Training Epochs**: {model_info.get('epoch', 'Unknown')}
- **Optimizer**: Adam
- **Framework**: icefall + k2 + lhotse

## Usage

```python
# Load model with icefall
from icefall.checkpoint import load_checkpoint
from conformer import Conformer
import torch

# Model configuration
model = Conformer(
    num_features=80,
    nhead=4,
    d_model=256,
    num_classes=5000,  # Adjust based on your vocab size
    subsampling_factor=4,
    num_decoder_layers=0,
    vgg_frontend=False,
    use_feat_batchnorm=True,
)

# Load checkpoint
load_checkpoint("best-valid-loss.pt", model)
model.eval()
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{icefall2021,
  title={{Icefall: A speech recognition toolkit with PyTorch}},
  author={{The icefall development team}},
  howpublished={{\\url{{https://github.com/k2-fsa/icefall}}}},
  year={{2021}}
}}
```

## License

This model is released under the Apache 2.0 License.
"""
    return model_card

def extract_model_info(checkpoint_path: Path) -> Dict[str, Any]:
    """Extract model information from checkpoint"""
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model_info = {
            'model_name': 'icefall-conformer-ctc-librispeech',
            'checkpoint_path': str(checkpoint_path)
        }
        
        # Extract information from checkpoint
        if 'epoch' in checkpoint:
            model_info['epoch'] = checkpoint['epoch']
        
        if 'best_valid_loss' in checkpoint:
            model_info['best_valid_loss'] = checkpoint['best_valid_loss']
            
        if 'model' in checkpoint:
            # Count parameters
            num_params = sum(p.numel() for p in checkpoint['model'].values())
            model_info['num_params'] = f"{num_params:,}"
            
        # Model architecture info (you might need to adjust these)
        model_info.update({
            'feature_dim': 80,
            'attention_dim': 256,
            'nhead': 4,
            'subsampling_factor': 4
        })
        
        return model_info
        
    except Exception as e:
        logging.error(f"Error extracting model info: {e}")
        return {'model_name': 'icefall-conformer-ctc-librispeech'}

def create_config_json(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a config.json file for the model"""
    
    config = {
        "architectures": ["Conformer"],
        "model_type": "conformer_ctc",
        "framework": "icefall",
        "feature_dim": model_info.get('feature_dim', 80),
        "attention_dim": model_info.get('attention_dim', 256),
        "nhead": model_info.get('nhead', 4),
        "subsampling_factor": model_info.get('subsampling_factor', 4),
        "num_decoder_layers": 0,
        "vgg_frontend": False,
        "use_feat_batchnorm": True,
        "sample_rate": 16000,
        "language": "en"
    }
    
    return config

def upload_to_huggingface(
    checkpoint_path: Path,
    repo_name: str,
    token: str = None,
    private: bool = False
):
    """Upload icefall model to Hugging Face Hub"""
    
    # Create temporary directory for upload
    temp_dir = Path("./hf_upload_temp")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Extract model information
        print("Extracting model information...")
        model_info = extract_model_info(checkpoint_path)
        
        # Copy model file
        print("Copying model file...")
        shutil.copy2(checkpoint_path, temp_dir / "best-valid-loss.pt")
        
        # Create model card
        print("Creating model card...")
        model_card = create_model_card(model_info)
        with open(temp_dir / "README.md", "w") as f:
            f.write(model_card)
        
        # Create config.json
        print("Creating config.json...")
        config = create_config_json(model_info)
        with open(temp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create additional files
        print("Creating additional files...")
        
        # Create inference example
        inference_example = '''#!/usr/bin/env python3
"""
Example inference script for icefall Conformer CTC model
"""

import torch
from pathlib import Path

def load_model(model_path: str):
    """Load the icefall Conformer model"""
    
    # You'll need to have icefall installed and import the Conformer class
    # from conformer import Conformer
    # from icefall.checkpoint import load_checkpoint
    
    # model = Conformer(
    #     num_features=80,
    #     nhead=4,
    #     d_model=256,
    #     num_classes=5000,  # Adjust based on vocab
    #     subsampling_factor=4,
    #     num_decoder_layers=0,
    #     vgg_frontend=False,
    #     use_feat_batchnorm=True,
    # )
    
    # load_checkpoint(model_path, model)
    # model.eval()
    # return model
    
    pass

if __name__ == "__main__":
    model = load_model("best-valid-loss.pt")
    print("Model loaded successfully!")
'''
        
        with open(temp_dir / "inference_example.py", "w") as f:
            f.write(inference_example)
        
        # Create requirements.txt
        requirements = """torch>=1.9.0
torchaudio>=0.9.0
k2
lhotse
icefall
"""
        with open(temp_dir / "requirements.txt", "w") as f:
            f.write(requirements)
        
        # Initialize Hugging Face API
        api = HfApi(token=token)
        
        # Create repository
        print(f"Creating repository: {repo_name}")
        try:
            create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                repo_type="model"
            )
            print(f"✅ Repository {repo_name} created successfully!")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Repository {repo_name} already exists, continuing...")
            else:
                raise e
        
        # Upload files
        print("Uploading files to Hugging Face Hub...")
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            token=token,
            commit_message="Upload icefall Conformer CTC model"
        )
        
        print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        raise e
        
    finally:
        # Clean up
        print("Cleaning up temporary files...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    """Main function"""
    
    # Configuration
    checkpoint_path = Path("/home/hdd2/jenny/ASRToolkit/icefall/egs/librispeech/ASR/conformer_ctc/exp-cleanASR/models/best-valid-loss.pt")
    
    # Get user input
    repo_name = input("Enter repository name (e.g., username/model-name): ").strip()
    if not repo_name:
        print("Repository name is required!")
        return
    
    token = input("Enter your Hugging Face token (or press Enter to use saved token): ").strip()
    if not token:
        token = None  # Will use saved token from huggingface-cli login
    
    private = input("Make repository private? (y/N): ").strip().lower() == 'y'
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"📁 Checkpoint path: {checkpoint_path}")
    print(f"🔗 Repository: {repo_name}")
    print(f"🔒 Private: {private}")
    
    confirm = input("\\nProceed with upload? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Upload cancelled.")
        return
    
    # Upload model
    upload_to_huggingface(
        checkpoint_path=checkpoint_path,
        repo_name=repo_name,
        token=token,
        private=private
    )

if __name__ == "__main__":
    main()

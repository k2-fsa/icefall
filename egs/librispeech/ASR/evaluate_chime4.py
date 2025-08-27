#!/usr/bin/env python3
"""
Evaluate trained conformer_ctc model on CHiME-4 dataset.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
from conformer_ctc.asr_datamodule import LibriSpeechAsrDataModule
from conformer_ctc.conformer import Conformer


def setup_logging(args):
    """Setup logging configuration."""
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained conformer model from checkpoint."""
    logging.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    params = checkpoint.get('params', {})
    
    # Create model with parameters from checkpoint
    model = Conformer(
        num_features=params.get('num_features', 80),
        nhead=params.get('nhead', 8),
        d_model=params.get('d_model', 512),
        num_classes=params.get('num_classes', 5000),  # Adjust based on your vocab
        subsampling_factor=params.get('subsampling_factor', 4),
        num_decoder_layers=params.get('num_decoder_layers', 0),
        vgg_frontend=params.get('vgg_frontend', False),
        num_encoder_layers=params.get('num_encoder_layers', 12),
        att_rate=params.get('att_rate', 0.0),
        # Add other parameters as needed
    )
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logging.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def evaluate_chime4(model, datamodule, device: torch.device):
    """Evaluate model on CHiME-4 test sets."""
    from conformer_ctc.decode import greedy_search
    
    # Get CHiME-4 test dataloaders
    test_loaders = datamodule.chime4_test_dataloaders()
    
    if not test_loaders:
        logging.error("No CHiME-4 test dataloaders found!")
        return {}
    
    results = {}
    
    for test_set_name, test_loader in test_loaders.items():
        logging.info(f"Evaluating on CHiME-4 {test_set_name}")
        
        total_num_tokens = 0
        total_num_errors = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    logging.info(f"Processing batch {batch_idx} of {test_set_name}")
                
                feature = batch["inputs"].to(device)
                # Convert supervisions to expected format
                supervisions = batch["supervisions"]
                
                # Forward pass
                encoder_out, encoder_out_lens = model.encode(feature, supervisions)
                
                # Greedy decoding
                hyps = greedy_search(
                    model=model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                )
                
                # Calculate WER (simplified - you may want to use proper WER calculation)
                for i, hyp in enumerate(hyps):
                    ref_tokens = supervisions["text"][i].split()
                    hyp_tokens = hyp.split()
                    
                    total_num_tokens += len(ref_tokens)
                    # Simple edit distance calculation (you may want to use proper edit distance)
                    errors = abs(len(ref_tokens) - len(hyp_tokens))
                    total_num_errors += errors
                    
                    if batch_idx == 0 and i == 0:  # Print first example
                        logging.info(f"Reference: {supervisions['text'][i]}")
                        logging.info(f"Hypothesis: {hyp}")
        
        # Calculate WER
        wer = total_num_errors / total_num_tokens if total_num_tokens > 0 else 1.0
        results[test_set_name] = {"WER": wer, "total_tokens": total_num_tokens}
        
        logging.info(f"{test_set_name} WER: {wer:.4f} ({total_num_errors}/{total_num_tokens})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate conformer CTC on CHiME-4")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/fbank"),
        help="Path to directory with manifests",
    )
    parser.add_argument(
        "--max-duration", type=float, default=200.0, help="Max duration for batching"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create data module
    datamodule = LibriSpeechAsrDataModule(args)
    
    # Evaluate on CHiME-4
    results = evaluate_chime4(model, datamodule, device)
    
    # Print summary
    logging.info("=" * 50)
    logging.info("CHiME-4 Evaluation Results:")
    for test_set, result in results.items():
        logging.info(f"{test_set}: WER = {result['WER']:.4f}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()

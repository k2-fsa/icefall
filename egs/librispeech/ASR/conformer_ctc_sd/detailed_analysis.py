#!/usr/bin/env python3
"""
Detailed test to understand attention map size changes and encoder output structure
"""

import torch
import torch.nn as nn
from conformer_ctc import ConformerCTC

def test_attention_map_sizes():
    """Test attention map sizes at different layers"""
    print("=" * 60)
    print("ATTENTION MAP SIZE ANALYSIS")
    print("=" * 60)
    
    # Create model
    model = ConformerCTC(
        num_features=80,
        num_classes=500,
        d_model=256,
        num_encoder_layers=6,
        nhead=4,
        distill_layers=[0, 1, 2, 3, 4, 5],  # All layers
        knowledge_type='attention-map',
    )
    model.eval()
    
    # Create test input with different sequence lengths
    batch_size = 2
    seq_lens = [100, 80]  # Different lengths to see padding effect
    max_len = max(seq_lens)
    
    # Create input
    x = torch.randn(batch_size, max_len, 80)
    x_lens = torch.tensor(seq_lens)
    
    # Create targets
    y = torch.randint(0, 500, (batch_size, 50))
    y_lens = torch.tensor([50, 40])
    
    print(f"Input shape: {x.shape}")
    print(f"Input lengths: {x_lens}")
    print(f"Target shape: {y.shape}")
    print(f"Target lengths: {y_lens}")
    print()
    
    # Forward pass
    with torch.no_grad():
        # ConformerCTC forward only takes x and supervisions
        supervisions = {
            'sequence_idx': torch.arange(batch_size),
            'start_frame': torch.zeros(batch_size),
            'num_frames': x_lens,
        }
        outputs = model(x, supervisions)
    
    print("Attention maps from different layers:")
    for i, attn_map in enumerate(outputs['distill_outputs']):
        print(f"Layer {model.distill_layers[i]}: {attn_map.shape}")
        # Check for NaN or inf
        if torch.isnan(attn_map).any():
            print(f"  ⚠️  WARNING: NaN detected in layer {i}")
        if torch.isinf(attn_map).any():
            print(f"  ⚠️  WARNING: Inf detected in layer {i}")
    
    print()
    print("Analysis:")
    print("- All attention maps should have same batch_size and num_heads")
    print("- Sequence length dimension may vary due to subsampling in conformer")
    print("- Later layers typically have shorter sequences due to subsampling")

def test_encoder_output_structure():
    """Test encoder output structure and understand distill_outputs vs distill_hidden"""
    print("=" * 60)
    print("ENCODER OUTPUT STRUCTURE ANALYSIS") 
    print("=" * 60)
    
    # Create model
    model = ConformerCTC(
        num_features=80,
        num_classes=500,
        d_model=256,
        num_encoder_layers=6,
        nhead=4,
        distill_layers=[1, 3, 5],  # Selected layers
        knowledge_type='encoder-output',
    )
    model.eval()
    
    # Create test input
    batch_size = 2
    seq_lens = [100, 80]
    max_len = max(seq_lens)
    
    x = torch.randn(batch_size, max_len, 80)
    x_lens = torch.tensor(seq_lens)
    y = torch.randint(0, 500, (batch_size, 50))
    y_lens = torch.tensor([50, 40])
    
    print(f"Input shape: {x.shape}")
    print(f"Selected distillation layers: {model.distill_layers}")
    print()
    
    # Forward pass
    with torch.no_grad():
        # ConformerCTC forward only takes x and supervisions
        supervisions = {
            'sequence_idx': torch.arange(batch_size),
            'start_frame': torch.zeros(batch_size),
            'num_frames': x_lens,
        }
        outputs = model(x, supervisions)
    
    print("Encoder outputs from selected layers:")
    print(f"distill_outputs type: {type(outputs['distill_outputs'])}")
    print(f"distill_outputs length: {len(outputs['distill_outputs'])}")
    
    for i, enc_output in enumerate(outputs['distill_outputs']):
        layer_idx = model.distill_layers[i]
        print(f"Layer {layer_idx}: {enc_output.shape}")
        print(f"  Mean: {enc_output.mean().item():.4f}")
        print(f"  Std: {enc_output.std().item():.4f}")
    
    print()
    if 'distill_hidden' in outputs:
        print("distill_hidden structure:")
        print(f"distill_hidden type: {type(outputs['distill_hidden'])}")
        if isinstance(outputs['distill_hidden'], (list, tuple)):
            print(f"distill_hidden length: {len(outputs['distill_hidden'])}")
            for i, hidden in enumerate(outputs['distill_hidden']):
                print(f"Hidden {i}: {hidden.shape if torch.is_tensor(hidden) else type(hidden)}")
        else:
            print(f"distill_hidden shape: {outputs['distill_hidden'].shape}")
    else:
        print("❌ distill_hidden not found in outputs")
    
    print()
    print("Key differences:")
    print("- distill_outputs: Contains the actual hidden states from selected encoder layers")
    print("- distill_hidden: May contain additional context or processed versions")
    print("- For encoder-output distillation, we primarily use distill_outputs")

def test_subsampling_effect():
    """Test how subsampling affects sequence lengths through layers"""
    print("=" * 60)
    print("SUBSAMPLING EFFECT ANALYSIS")
    print("=" * 60)
    
    from conformer import ConformerEncoder
    
    # Create standalone encoder to track subsampling
    encoder = ConformerEncoder(
        num_features=80,
        d_model=256,
        num_layers=6,
        nhead=4,
        distill_layers=[0, 1, 2, 3, 4, 5],
        knowledge_type='attention-map'
    )
    encoder.eval()
    
    # Test input
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 80)
    x_lens = torch.tensor([seq_len, seq_len])
    
    print(f"Original input: {x.shape}")
    
    with torch.no_grad():
        encoder_out, encoder_out_lens, distill_outputs, distill_hidden = encoder(x, x_lens)
    
    print(f"Final encoder output: {encoder_out.shape}")
    print(f"Final encoder lengths: {encoder_out_lens}")
    print()
    
    if encoder.knowledge_type == 'attention-map':
        print("Attention map sizes through layers:")
        for i, attn_map in enumerate(distill_outputs):
            # Attention map shape: [batch, num_heads, seq_len, seq_len]
            seq_len_at_layer = attn_map.shape[-1]
            print(f"Layer {encoder.distill_layers[i]}: attention [{attn_map.shape[0]}, {attn_map.shape[1]}, {attn_map.shape[2]}, {attn_map.shape[3]}] -> seq_len = {seq_len_at_layer}")
    
    print()
    print("Observations:")
    print("- Sequence length typically reduces due to subsampling in early layers")
    print("- This affects attention map sizes (they become smaller)")
    print("- All layers after subsampling will have the same reduced sequence length")

if __name__ == "__main__":
    print("🔍 DETAILED DISTILLATION ANALYSIS")
    print("=" * 80)
    
    try:
        test_attention_map_sizes()
        print("\n" + "="*80 + "\n")
        
        test_encoder_output_structure()
        print("\n" + "="*80 + "\n")
        
        test_subsampling_effect()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

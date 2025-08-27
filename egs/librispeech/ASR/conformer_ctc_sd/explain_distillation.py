#!/usr/bin/env python3
"""
Simple explanation of attention map sizes and encoder outputs
"""

import torch
from conformer_ctc import ConformerCTC

def explain_attention_map_sizes():
    """Explain how attention map sizes are determined"""
    print("🔍 ATTENTION MAP SIZE EXPLANATION")
    print("=" * 50)
    
    print("Attention Map Size는 다음과 같이 결정됩니다:")
    print()
    print("1. 입력 시퀀스 길이:")
    print("   - 원본 오디오 프레임 수에 따라 결정")
    print("   - 예: 100 프레임 -> 100 길이")
    print()
    print("2. Subsampling 효과:")
    print("   - Conformer는 초기 레이어에서 subsampling 수행")
    print("   - 보통 4배 또는 6배 압축")
    print("   - 예: 100 -> 25 (4배), 100 -> 16 (6배)")
    print()
    print("3. Attention Map 형태:")
    print("   - [batch_size, num_heads, seq_len, seq_len]")
    print("   - seq_len은 해당 레이어에서의 시퀀스 길이")
    print("   - subsampling 후에는 모든 레이어가 같은 seq_len")
    print()
    
    # Create test model
    model = ConformerCTC(
        num_features=80,
        num_classes=500,
        d_model=256,
        num_encoder_layers=6,
        nhead=4,
        distill_layers=[2, 4],
        knowledge_type='attention-map',
    )
    
    # Test with different input sizes
    test_cases = [
        {"seq_len": 50, "name": "Short audio"},
        {"seq_len": 100, "name": "Medium audio"},
        {"seq_len": 200, "name": "Long audio"}
    ]
    
    print("실제 테스트:")
    for case in test_cases:
        seq_len = case["seq_len"]
        batch_size = 2
        
        x = torch.randn(batch_size, seq_len, 80)
        supervisions = {
            'sequence_idx': torch.arange(batch_size),
            'start_frame': torch.zeros(batch_size),
            'num_frames': torch.tensor([seq_len, seq_len]),
        }
        
        with torch.no_grad():
            outputs = model(x, supervisions)
        
        print(f"\n{case['name']} (입력 길이: {seq_len}):")
        if 'distill_outputs' in outputs and len(outputs['distill_outputs']) > 0:
            for i, attn_map in enumerate(outputs['distill_outputs']):
                layer_idx = model.distill_layers[i]
                attn_seq_len = attn_map.shape[-1]
                compression_ratio = seq_len / attn_seq_len
                print(f"  Layer {layer_idx}: {attn_map.shape} (압축비: {compression_ratio:.1f}x)")

def explain_encoder_outputs():
    """Explain encoder output structure"""
    print("\n🔍 ENCODER OUTPUT vs DISTILL OUTPUT")
    print("=" * 50)
    
    print("Encoder-Output 모드에서:")
    print()
    print("1. distill_outputs:")
    print("   - 선택된 레이어들의 실제 hidden states")
    print("   - 각 레이어의 인코더 출력 (feature representations)")
    print("   - 형태: [batch_size, seq_len, d_model]")
    print("   - Self-distillation에 직접 사용되는 정보")
    print()
    print("2. distill_hidden:")
    print("   - 현재 구현에서는 distill_outputs와 동일")
    print("   - 향후 확장을 위한 placeholder")
    print("   - 추가적인 컨텍스트나 처리된 정보를 담을 수 있음")
    print()
    
    # Test encoder output mode
    model = ConformerCTC(
        num_features=80,
        num_classes=500,
        d_model=256,
        num_encoder_layers=6,
        nhead=4,
        distill_layers=[1, 3, 5],
        knowledge_type='encoder-output',
    )
    
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 80)
    supervisions = {
        'sequence_idx': torch.arange(batch_size),
        'start_frame': torch.zeros(batch_size),
        'num_frames': torch.tensor([seq_len, seq_len]),
    }
    
    with torch.no_grad():
        outputs = model(x, supervisions)
    
    print("실제 예시:")
    print(f"입력 크기: {x.shape}")
    print(f"선택된 레이어: {model.distill_layers}")
    print()
    
    if 'distill_outputs' in outputs:
        print("distill_outputs (각 레이어의 hidden states):")
        for i, enc_out in enumerate(outputs['distill_outputs']):
            layer_idx = model.distill_layers[i]
            print(f"  Layer {layer_idx}: {enc_out.shape}")
    
    print()
    print("📝 요약:")
    print("- Attention Map: 시퀀스 길이는 subsampling에 의해 결정")
    print("- Encoder Output: 각 레이어의 feature representation")
    print("- distill_outputs가 self-distillation의 핵심 데이터")

if __name__ == "__main__":
    try:
        explain_attention_map_sizes()
        explain_encoder_outputs()
        print("\n✅ 설명 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

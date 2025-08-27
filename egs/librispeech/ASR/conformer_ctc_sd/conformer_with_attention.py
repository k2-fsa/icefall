#!/usr/bin/env python3

"""
Conformer with attention map extraction support for multi-GPU training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from conformer import Conformer, ConformerEncoder, ConformerEncoderLayer


class ConformerEncoderLayerWithAttention(ConformerEncoderLayer):
    """ConformerEncoderLayer that can optionally return attention weights."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_attention = False
        
    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional attention weight return."""
        
        # Store original parameters
        src_orig = src
        
        warmup_scale = min(0.1 + warmup, 1.0)
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0

        # macaron style feed forward module
        src = src + self.dropout(self.feed_forward_macaron(src))

        # multi-headed self-attention module
        src_att, attn_weights = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attention or self.return_attention,
        )
        
        src = src + self.dropout(src_att)

        # convolution module
        src = src + self.dropout(
            self.conv_module(src, src_key_padding_mask=src_key_padding_mask)
        )

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        return src, attn_weights if (return_attention or self.return_attention) else None


class ConformerEncoderWithAttention(ConformerEncoder):
    """ConformerEncoder that can extract attention maps from specified layers."""
    
    def __init__(self, *args, attention_layers: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_layers = attention_layers or []
        
        # Replace layers with attention-capable versions
        new_layers = []
        for i, layer in enumerate(self.layers):
            # Create new layer with same parameters
            new_layer = ConformerEncoderLayerWithAttention(
                d_model=layer.d_model,
                nhead=layer.self_attn.num_heads,
                dim_feedforward=layer.feed_forward.w_1.in_features,
                dropout=layer.dropout.p,
                activation=layer.feed_forward.activation,
                layer_dropout=layer.layer_dropout,
                cnn_module_kernel=layer.conv_module.pointwise_conv2.out_channels // layer.conv_module.pointwise_conv1.out_channels,
            )
            
            # Copy weights
            new_layer.load_state_dict(layer.state_dict())
            
            # Enable attention return for specified layers
            if i in self.attention_layers:
                new_layer.return_attention = True
                
            new_layers.append(new_layer)
        
        self.layers = nn.ModuleList(new_layers)
    
    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[int, torch.Tensor]]:
        """Forward pass returning layer outputs and attention maps."""
        
        output = src
        layer_outputs = []
        attention_maps = {}
        
        for i, layer in enumerate(self.layers):
            output, attn_weights = layer(
                output,
                pos_emb,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                warmup=warmup,
                return_attention=i in self.attention_layers,
            )
            
            # Store layer output if needed
            if i in self.output_layers:
                layer_outputs.append(output)
                
            # Store attention weights if available
            if attn_weights is not None:
                attention_maps[i] = attn_weights
        
        # Ensure we always have the final layer output
        if len(self.output_layers) == 0 or (len(self.layers) - 1) not in self.output_layers:
            layer_outputs.append(output)
            
        # Calculate output lengths (assuming subsampling was applied earlier)
        output_lens = src_key_padding_mask.size(1) - src_key_padding_mask.sum(dim=1)
        
        return layer_outputs, output_lens, attention_maps


class ConformerWithAttention(Conformer):
    """Conformer with built-in attention map extraction support."""
    
    def __init__(self, *args, attention_layers: Optional[List[int]] = None, **kwargs):
        # Initialize parent without calling its __init__ to avoid double initialization
        nn.Module.__init__(self)
        
        # Store parameters
        self.num_features = kwargs.get('num_features')
        self.subsampling_factor = kwargs.get('subsampling_factor', 4)
        self.d_model = kwargs.get('d_model', 256)
        self.attention_layers = attention_layers or []
        
        # Create subsampling layer
        from subsampling import Conv2dSubsampling
        self.subsampling = Conv2dSubsampling(
            in_channels=1,
            out_channels=self.d_model,
            subsampling_factor=self.subsampling_factor,
        )
        
        # Create encoder with attention support
        self.encoder = ConformerEncoderWithAttention(
            *args,
            attention_layers=attention_layers,
            **kwargs
        )
    
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[int, torch.Tensor]]:
        """Forward pass returning layer outputs and attention maps."""
        
        # Subsampling
        x, pos_emb = self.subsampling(x)
        
        # Create padding mask
        max_len = x.size(1)
        batch_size = x.size(0)
        lengths_after_subsampling = ((x_lens - 1) // self.subsampling_factor) + 1
        
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths_after_subsampling[:, None]
        
        # Encoder forward pass
        layer_outputs, output_lens, attention_maps = self.encoder(
            x, pos_emb, src_key_padding_mask=padding_mask
        )
        
        return layer_outputs, output_lens, attention_maps

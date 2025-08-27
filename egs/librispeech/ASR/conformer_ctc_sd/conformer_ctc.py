#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Conformer CTC model with support for self-distillation on encoder outputs and attention maps.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import Conformer
from conformer_with_attention import ConformerWithAttention

# from icefall.utils import add_sos_eos, is_jit_tracing, is_jit_scripting


class ConformerCTC(nn.Module):
    """Conformer CTC model with self-distillation support.
    
    This model extends the basic Conformer encoder to support extraction of 
    intermediate layer outputs and attention maps for self-distillation.
    
    Args:
        num_features: Number of input features (e.g., 80 for Fbank)
        num_classes: Number of output classes (vocabulary size)
        subsampling_factor: Subsampling factor of encoder 
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Feedforward dimension
        num_encoder_layers: Number of encoder layers
        dropout: Dropout rate
        cnn_module_kernel: Convolution module kernel size
        distill_layers: List of layer indices for distillation (0-based)
        knowledge_type: Type of knowledge to extract ('encoder-output' or 'attention-map')
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        distill_layers: Optional[List[int]] = None,
        knowledge_type: str = "encoder-output",
    ) -> None:
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.d_model = d_model
        self.distill_layers = distill_layers or []
        self.knowledge_type = knowledge_type
        
        # Determine which layers need to output intermediate results
        output_layers = []
        if distill_layers:
            output_layers.extend(distill_layers)
        
        # Create conformer encoder with support for intermediate outputs and attention maps
        if knowledge_type == "attention-map":
            self.encoder = ConformerWithAttention(
                num_features=num_features,
                subsampling_factor=subsampling_factor,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel,
                attention_layers=distill_layers,
            )
        else:
            self.encoder = Conformer(
                num_features=num_features,
                subsampling_factor=subsampling_factor,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel,
            )
        
        # Modify the encoder to output intermediate layers
        if distill_layers and knowledge_type != "attention-map":
            # Add the last layer to always be included
            output_layers = list(set(distill_layers + [num_encoder_layers - 1]))
            output_layers.sort()
            self.encoder.encoder.output_layers = output_layers
        
        # CTC output projection
        self.ctc_output = nn.Linear(d_model, num_classes)
        
    def _modify_encoder_for_attention_maps(self):
        """Modify the encoder layers to extract attention maps using forward hooks."""
        # Store attention weights during forward pass
        self._attention_storage = {}
        self._hooks = []
        
        # Register forward hooks instead of modifying methods directly
        for layer_idx in self.distill_layers:
            if layer_idx < len(self.encoder.encoder.layers):
                layer = self.encoder.encoder.layers[layer_idx]
                
                # Create hook function that captures attention weights
                def create_attention_hook(idx):
                    def attention_hook(module, input, output):
                        # This hook will be called after the layer's forward pass
                        # We need to modify the layer to store attention weights
                        pass
                    return attention_hook
                
                # Register the hook
                hook = layer.register_forward_hook(create_attention_hook(layer_idx))
                self._hooks.append(hook)
        
        # Alternative: Monkey patch the self_attn module specifically
        for layer_idx in self.distill_layers:
            if layer_idx < len(self.encoder.encoder.layers):
                layer = self.encoder.encoder.layers[layer_idx]
                original_self_attn = layer.self_attn
                
                def create_patched_attention(orig_attn, idx):
                    class PatchedAttention(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.original_attn = orig_attn
                            self.layer_idx = idx
                        
                        def forward(self, query, key, value, pos_emb=None, attn_mask=None, 
                                  key_padding_mask=None, need_weights=False):
                            # Always request attention weights for distillation layers
                            output, attn_weights = self.original_attn(
                                query, key, value, pos_emb=pos_emb, attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask, need_weights=True
                            )
                            
                            # Store attention weights in parent module
                            if attn_weights is not None and hasattr(self, '_parent_storage'):
                                self._parent_storage[self.layer_idx] = attn_weights
                            
                            # Return in expected format
                            if need_weights:
                                return output, attn_weights
                            else:
                                return output
                    
                    patched = PatchedAttention()
                    patched._parent_storage = self._attention_storage
                    return patched
                
                # Replace the self_attn module
                layer.self_attn = create_patched_attention(original_self_attn, layer_idx)
    
    def cleanup_hooks(self):
        """Clean up forward hooks to prevent memory leaks."""
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
        
    def forward(
        self, 
        x: torch.Tensor, 
        supervisions: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with support for distillation knowledge extraction.
        
        Args:
            x: Input tensor of shape (N, T, num_features)
            supervisions: Supervision information (optional)
            
        Returns:
            Dictionary containing:
            - ctc_output: CTC output logits (N, T', num_classes)
            - encoder_out: Final encoder output (N, T', d_model)
            - encoder_out_lens: Sequence lengths after subsampling
            - distill_outputs: Intermediate layer outputs for distillation
            - attention_maps: Attention maps if knowledge_type is 'attention-map'
        """
        # Get sequence lengths
        x_lens = x.new_zeros(x.size(0)).long() + x.size(1)
        
        if self.knowledge_type == "attention-map":
            # Use ConformerWithAttention
            layer_outputs, output_lens, attention_maps = self.encoder(x, x_lens)
        else:
            # Use regular Conformer
            layer_outputs, output_lens = self.encoder(x, x_lens)
            attention_maps = {}
        
        # The last output is the final encoder output
        encoder_out = layer_outputs[-1]  # (N, T', d_model)
        
        # CTC output projection
        ctc_output = self.ctc_output(encoder_out)  # (N, T', num_classes)
        
        # Prepare return dictionary
        result = {
            'ctc_output': ctc_output,
            'encoder_out': encoder_out,
            'encoder_out_lens': output_lens,
        }
        
        # Extract distillation knowledge based on type
        if self.distill_layers:
            if self.knowledge_type == "encoder-output":
                # Extract encoder outputs from specified layers
                distill_outputs = {}
                for i, layer_idx in enumerate(self.distill_layers):
                    if i < len(layer_outputs):
                        distill_outputs[layer_idx] = layer_outputs[i]
                result['distill_outputs'] = distill_outputs
                
                # For backward compatibility, also provide single distill_hidden
                if len(self.distill_layers) == 1:
                    result['distill_hidden'] = layer_outputs[0]
                    
            elif self.knowledge_type == "attention-map":
                # Return attention maps from ConformerWithAttention
                result['attention_maps'] = attention_maps
        
        return result


def compute_distillation_loss(
    teacher_knowledge: torch.Tensor,
    student_knowledge: torch.Tensor,
    knowledge_lens: torch.Tensor,
    loss_type: str = "mse",
    knowledge_type: str = "encoder-output",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute distillation loss between teacher and student knowledge.
    
    Args:
        teacher_knowledge: Teacher knowledge tensor
        student_knowledge: Student knowledge tensor  
        knowledge_lens: Sequence lengths for masking
        loss_type: Type of loss ('mse', 'cosine' for encoder outputs; 'kl' for attention maps)
        knowledge_type: Type of knowledge ('encoder-output' or 'attention-map')
        temperature: Temperature for softmax (used with KL divergence)
        
    Returns:
        Computed distillation loss
    """
    if knowledge_type == "encoder-output":
        # Handle encoder output distillation
        if loss_type == "mse":
            return _compute_mse_loss(teacher_knowledge, student_knowledge, knowledge_lens)
        elif loss_type == "cosine":
            return _compute_cosine_loss(teacher_knowledge, student_knowledge, knowledge_lens)
        else:
            raise ValueError(f"Unsupported loss type for encoder outputs: {loss_type}")
            
    elif knowledge_type == "attention-map":
        # Handle attention map distillation
        if loss_type == "kl":
            return _compute_kl_divergence_loss(teacher_knowledge, student_knowledge, knowledge_lens, temperature)
        else:
            raise ValueError(f"Unsupported loss type for attention maps: {loss_type}")
    else:
        raise ValueError(f"Unsupported knowledge type: {knowledge_type}")


def _compute_mse_loss(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor, 
    hidden_lens: torch.Tensor
) -> torch.Tensor:
    """Compute MSE loss between teacher and student hidden states."""
    # teacher_hidden, student_hidden: (N, T, d_model)
    # hidden_lens: (N,)
    
    batch_size, max_len, _ = teacher_hidden.shape
    
    # Create mask for valid positions
    mask = torch.arange(max_len, device=hidden_lens.device)[None, :] < hidden_lens[:, None]
    mask = mask.float()  # (N, T)
    
    # Compute MSE loss element-wise
    mse_loss = F.mse_loss(student_hidden, teacher_hidden, reduction='none')  # (N, T, d_model)
    mse_loss = mse_loss.mean(dim=-1)  # (N, T)
    
    # Apply mask and compute mean
    masked_loss = mse_loss * mask
    total_loss = masked_loss.sum()
    total_tokens = mask.sum()
    
    return total_loss / (total_tokens + 1e-8)


def _compute_cosine_loss(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    hidden_lens: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity loss between teacher and student hidden states."""
    # teacher_hidden, student_hidden: (N, T, d_model)
    # hidden_lens: (N,)
    
    batch_size, max_len, _ = teacher_hidden.shape
    
    # Create mask for valid positions
    mask = torch.arange(max_len, device=hidden_lens.device)[None, :] < hidden_lens[:, None]
    mask = mask.float()  # (N, T)
    
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(teacher_hidden, student_hidden, dim=-1)  # (N, T)
    
    # Convert to loss (1 - cosine_similarity)
    cosine_loss = 1.0 - cosine_sim  # (N, T)
    
    # Apply mask and compute mean
    masked_loss = cosine_loss * mask
    total_loss = masked_loss.sum()
    total_tokens = mask.sum()
    
    return total_loss / (total_tokens + 1e-8)


def _compute_kl_divergence_loss(
    teacher_attention: torch.Tensor,
    student_attention: torch.Tensor,
    attention_lens: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute KL divergence loss between teacher and student attention maps.
    
    Args:
        teacher_attention: Teacher attention weights (N, H, T, T) or (N, T, T)
        student_attention: Student attention weights (N, H, T, T) or (N, T, T)
        attention_lens: Sequence lengths for masking (N,)
        temperature: Temperature for softmax smoothing
        
    Returns:
        KL divergence loss
    """
    # Handle different attention map shapes
    if teacher_attention.dim() == 4:
        # (N, H, T, T) -> average over heads to get (N, T, T)
        teacher_attention = teacher_attention.mean(dim=1)
        student_attention = student_attention.mean(dim=1)
    
    batch_size, seq_len, _ = teacher_attention.shape
    
    # Create attention mask
    mask = torch.arange(seq_len, device=attention_lens.device)[None, :] < attention_lens[:, None]
    mask = mask.float()  # (N, T)
    
    # Create 2D mask for attention matrix
    mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (N, T, T)
    
    # Apply temperature and compute log probabilities
    teacher_log_probs = F.log_softmax(teacher_attention / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_attention / temperature, dim=-1)
    
    # Convert student to probabilities for KL divergence
    student_probs = F.softmax(student_attention / temperature, dim=-1)
    
    # Compute KL divergence: KL(student || teacher) = sum(student * (log(student) - log(teacher)))
    kl_div = student_probs * (student_log_probs - teacher_log_probs)  # (N, T, T)
    kl_div = kl_div.sum(dim=-1)  # (N, T)
    
    # Apply mask and compute mean
    masked_kl = kl_div * mask
    total_loss = masked_kl.sum()
    total_tokens = mask.sum()
    
    return total_loss / (total_tokens + 1e-8)


def compute_multi_layer_distillation_loss(
    teacher_knowledge: Dict[int, torch.Tensor],
    student_knowledge: Dict[int, torch.Tensor],
    knowledge_lens: torch.Tensor,
    layer_indices: List[int],
    loss_type: str = "mse",
    knowledge_type: str = "encoder-output",
    aggregation: str = "layer_avg",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute multi-layer distillation loss with specified aggregation strategy.
    
    Args:
        teacher_knowledge: Dictionary mapping layer indices to teacher knowledge tensors
        student_knowledge: Dictionary mapping layer indices to student knowledge tensors
        knowledge_lens: Sequence lengths for masking
        layer_indices: List of layer indices to compute loss for
        loss_type: Type of loss computation
        knowledge_type: Type of knowledge being distilled
        aggregation: Aggregation strategy ('layer_avg' or 'output_avg')
        temperature: Temperature for softmax (attention maps)
        
    Returns:
        Aggregated distillation loss
    """
    if aggregation == "layer_avg":
        # Compute loss for each layer and average them
        total_loss = torch.tensor(0.0, device=knowledge_lens.device)
        valid_layers = 0
        
        for layer_idx in layer_indices:
            if layer_idx in teacher_knowledge and layer_idx in student_knowledge:
                layer_loss = compute_distillation_loss(
                    teacher_knowledge[layer_idx],
                    student_knowledge[layer_idx],
                    knowledge_lens,
                    loss_type,
                    knowledge_type,
                    temperature,
                )
                total_loss += layer_loss
                valid_layers += 1
        
        if valid_layers > 0:
            return total_loss / valid_layers
        else:
            return torch.tensor(0.0, device=knowledge_lens.device)
        
    elif aggregation == "output_avg":
        # Average the layer outputs first, then compute a single loss
        if knowledge_type == "encoder-output":
            # Stack and average encoder outputs
            teacher_outputs = []
            student_outputs = []
            
            for layer_idx in layer_indices:
                if layer_idx in teacher_knowledge and layer_idx in student_knowledge:
                    teacher_outputs.append(teacher_knowledge[layer_idx])
                    student_outputs.append(student_knowledge[layer_idx])
            
            if not teacher_outputs:
                return torch.tensor(0.0, device=knowledge_lens.device)
            
            # Average the outputs
            avg_teacher = torch.stack(teacher_outputs).mean(dim=0)
            avg_student = torch.stack(student_outputs).mean(dim=0)
            
            return compute_distillation_loss(
                avg_teacher, avg_student, knowledge_lens, loss_type, knowledge_type, temperature
            )
            
        elif knowledge_type == "attention-map":
            # Average attention maps and compute KL divergence
            teacher_attentions = []
            student_attentions = []
            
            for layer_idx in layer_indices:
                if layer_idx in teacher_knowledge and layer_idx in student_knowledge:
                    teacher_attentions.append(teacher_knowledge[layer_idx])
                    student_attentions.append(student_knowledge[layer_idx])
            
            if not teacher_attentions:
                return torch.tensor(0.0, device=knowledge_lens.device)
            
            # Average the attention maps
            avg_teacher_attention = torch.stack(teacher_attentions).mean(dim=0)
            avg_student_attention = torch.stack(student_attentions).mean(dim=0)
            
            return compute_distillation_loss(
                avg_teacher_attention, avg_student_attention, knowledge_lens, 
                loss_type, knowledge_type, temperature
            )
    else:
        raise ValueError(f"Unsupported aggregation strategy: {aggregation}")

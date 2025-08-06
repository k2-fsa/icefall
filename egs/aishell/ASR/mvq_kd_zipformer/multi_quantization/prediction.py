import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch import optim
from typing import Tuple, Optional, List
from .checkpoint import checkpoint # from current directory.. could not get relative import to work..

# functional version of joint codebook loss, added so that we can more easily implement
# checkpointing to save memory.
def joint_codebook_loss(predictor: Tensor,
                        codebook_indexes: Tensor,
                        teacher_weights: List[float],
                        linear1_weight: Tensor,
                        linear1_bias: Optional[Tensor],
                        codebook_embedding_weight: Tensor,
                        linear2_weight: Tensor,
                        linear2b_weight: Tensor,
                        linear2_bias: Tensor,
                        ignore_index: int,
                        is_joint: bool,
                        enable_dynamic_temp: bool,
                        reduction: str) -> Tensor:
    """
    Args:
       predictor: predictor tensor of shape (*, predictor_channels)
       codebook_indexes: codebook indexes of shape (*, num_codebooks)
       linear1_weight: weight of shape (hidden_channels, predictor_channels)
       linear1_bias: optional bias of shape (hidden_channels,)
       codebook_embedding_weight: weight of shape ((num_codebooks - 1) * codebook_size,
                                                   hidden_channels)
       linear2_weight: weight of shape (num_codebooks, codebook_size,
                                                hidden_channels)
       linear2b_weight: weight of shape (num_codebooks, codebook_size,
                                                predictor_dim)
       linear2_bias: bias of shape (num_codebooks, codebook_size)
       ignore_index: index to ignore in cross entropy loss, e.g. -100
       is_joint: if false, this function becomes a standard CE loss.
       reduction: reduction in cross entropy loss, e.g. 'sum'
    """
    # single teacher
    if len(codebook_indexes.shape) == 3:
        num_codebooks = codebook_indexes.shape[-1]
        predictor_channels = predictor.shape[-1]
        hidden_channels = linear1_weight.shape[0]
        codebook_size = codebook_embedding_weight.shape[0] // (num_codebooks - 1)

        codebook_indexes = codebook_indexes.to(torch.int64)
        assert list(predictor.shape[:-1]) == list(codebook_indexes.shape[:-1])
        predictor = predictor.reshape(-1, predictor.shape[-1])  # (N, predictor_channels)
        codebook_indexes = codebook_indexes.reshape(-1, codebook_indexes.shape[-1])

        logprobs = torch.matmul(predictor, # (N, predictor_channels)
                                linear2b_weight.transpose(1, 2) # (num_codebooks, predictor_channels, codebook_size)
                                ).transpose(0, 1) # (N, num_codebooks, codebook_size)

        logprobs += linear2_bias
        if is_joint:
            first_indexes = codebook_indexes[:,:-1] # all but last codebook indexes; (N, num_codebooks-1)

            # do clamp(min=0) to avoid errors on padding (-100).. these frames will
            # later be ignored in the loss, so the value can be treated as a don't-care.
            first_indexes = first_indexes.clamp(min=0) + torch.arange(0, (num_codebooks - 1) * codebook_size,
                                                                    step=codebook_size,
                                                                    device=first_indexes.device)  # (N, num_codebooks-1)

            first_embeddings_scale = 0.5 * ((hidden_channels / num_codebooks) ** 0.5)
            first_embeddings = torch.nn.functional.embedding(first_indexes,
                                                            codebook_embedding_weight) * first_embeddings_scale # (N, num_codebooks-1, hidden_channels)


            hidden_predictor = torch.nn.functional.linear(predictor, linear1_weight, linear1_bias)
            all_embeddings = torch.cat((hidden_predictor.unsqueeze(1),
                                        first_embeddings),
                                    dim=1) # (N, num_codebooks, hidden_channels)

            # after cumsum, all positions will contain a contribution from 'hidden_predictor'; and
            # will also contain contributions from all *previous* codebooks.  Here, "position" means
            # a position in {0..num_codebooks-1}
            all_embeddings = torch.cumsum(all_embeddings, dim=1) # (N, num_codebooks, hidden_channels)

            all_embeddings = torch.nn.functional.relu(all_embeddings)

            logprobs += torch.matmul(all_embeddings.transpose(0, 1), # (num_codebooks, N, hidden_channels)
                                    linear2_weight.transpose(1, 2)   #  (num_codebooks, hidden_channels, codebook_size)
                                    ).transpose(0, 1)  # (N, num_codebooks, codebook_size)

        logits_student = logprobs.reshape(-1, codebook_size)
        targets_teacher = codebook_indexes.reshape(-1)
        # print(f"logits_student shape :  {logits_student.shape}       targets_teacher shape :  {targets_teacher.shape}")

        return torch.nn.functional.cross_entropy(logits_student,targets_teacher,ignore_index=ignore_index,reduction=reduction)
    # multiple teachers
    if len(codebook_indexes.shape) == 4:
        num_codebooks = codebook_indexes.shape[-1]
        predictor_channels = predictor.shape[-1]
        hidden_channels = linear1_weight.shape[0]
        codebook_size = codebook_embedding_weight.shape[0] // (num_codebooks - 1)
        codebook_indexes = codebook_indexes.to(torch.int64)
        assert list(predictor.shape[:-1]) == list(codebook_indexes.shape[1:3])
        predictor = predictor.reshape(-1, predictor.shape[-1])  # (N, predictor_channels)
        logprobs = torch.matmul(predictor, # (N, predictor_channels)
                                linear2b_weight.transpose(1, 2) # (num_codebooks, predictor_channels, codebook_size)
                                ).transpose(0, 1) # (N, num_codebooks, codebook_size)
        logprobs += linear2_bias
        logits_student = logprobs.reshape(-1, codebook_size)
        # teacher_weights = [0.5, 0.3, 0.2]    #示例
        # teacher_weights = torch.tensor(teacher_weights).to(codebook_indexes.device)
        teacher_weights = teacher_weights.to(codebook_indexes.device)
        print("teacher_weights is : ", teacher_weights)

        soft_group = hard_labels_to_soft(codebook_indexes, weights=teacher_weights, temperature=1.0, epsilon=0.05)  # shape [46, 322, 16, 257]
        targets_teacher = soft_group.reshape(-1, codebook_size + 1)
        processed_targets, valid_mask = process_soft_targets(targets_teacher) # [236992,256], [236992]

        return masked_cross_entropy(logits_student, processed_targets, valid_mask,reduction=reduction)



def masked_cross_entropy(logits, processed_targets, mask, reduction='sum'):
    """
    Args:
        logits: [N,256]
        processed_targets: [N,256]
        mask: [N] (True 表示有效样本)
        reduction: 'mean'/'sum'/'none'
    """
    # 计算逐样本交叉熵
    log_probs = F.log_softmax(logits, dim=1)                          # [N,256]
    per_sample_loss = -(processed_targets * log_probs).sum(dim=1)     # [N]
    # 应用掩码
    masked_loss = per_sample_loss * mask.float()
    # 聚合方式
    if reduction == 'mean':
        return masked_loss.sum() / mask.float().sum().clamp(min=1e-6)  # 
    elif reduction == 'sum':
        return masked_loss.sum()
    else: # 'none'
        return masked_loss

def process_soft_targets(soft_targets):
    """
    Args:
        soft_targets: shape [N, 257]
        
    Returns:
        processed_targets: shape [N, 256]
        mask: shape [N] (True 表示有效样本)
    """
    # 步骤1：生成忽略掩码（排除最大值为第257类的样本）
    max_values, max_indices = torch.max(soft_targets, dim=1)          # 获取每行最大值及其索引
    mask = (max_indices != (soft_targets.size(1) - 1))                # 当最大值不是最后一列时标记为有效 
    # 步骤2：概率重分配
    pad_class_probs = soft_targets[:, -1:]                            # 提取第257类概率 [N,1]
    redistributed = pad_class_probs / 256                             # 平均分配到前256类
    adjusted_probs = soft_targets[:, :-1] + redistributed             # 叠加到前256类
    # 归一化（确保概率和为1）
    adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)  # 
    
    return adjusted_probs, mask

def hard_labels_to_soft(
    hard_labels_group: torch.Tensor,  # shape [N_teachers, B, T, F]
    num_classes: int = 257,  # 256正常类 + 1个填充类
    weights: list = None,            
    temperature: float = 1.0,        
    epsilon: float = 0.0,
    pad_value: int = -100
) -> torch.Tensor:
    """将多教师硬标签转为软标签（支持填充值处理）
    
    Args:
        pad_value: 原始填充值（默认-100），将映射到num_classes-1位置
        最终生成的第256维（索引256）为填充类
    """
    # 0. 填充值替换（核心修改点）
    mapped_labels = torch.where(
        hard_labels_group == pad_value,
        torch.tensor(num_classes-1, device=hard_labels_group.device),
        hard_labels_group
    )
    # 验证替换后的标签范围（确保所有值∈[0, num_classes-1]）
    assert (mapped_labels >= 0).all() and (mapped_labels < num_classes).all(), \
        f"标签越界！有效范围[0, {num_classes-1}]，检测到最小值{mapped_labels.min()}, 最大值{mapped_labels.max()}"
    # 1. One-Hot编码（包含填充类）
    one_hot = F.one_hot(mapped_labels, num_classes).float()  # [N, B, T, F, 257]
    # 2. 融合教师预测
    if weights is not None:
        # weights = torch.tensor(weights).view(-1, 1, 1, 1, 1)
        weights = weights.clone().detach().view(-1, 1, 1, 1, 1)
        soft_labels = (one_hot * weights).sum(dim=0)  # 加权平均
    else:
        soft_labels = one_hot.mean(dim=0)             # 简单平均
    # 3. 温度缩放
    if temperature != 1.0:
        logits = torch.log(soft_labels + 1e-8)
        soft_labels = F.softmax(logits / temperature, dim=-1)
    # 4. 标签平滑
    if epsilon > 0:
        uniform = torch.ones_like(soft_labels) / num_classes
        soft_labels = (1 - epsilon) * soft_labels + epsilon * uniform
    
    return soft_labels  # shape [B, T, F, C]


class JointCodebookLoss(nn.Module):
    """
    This module predicts a group of codebook indexes from a vector.  The idea is that
    you have a number of codebooks (probably jointly trained), from class Quantizer,
    and you want to predict the probabilities of the codebook entries based on some
    predictor that you are training.

    The simplest thing would be to project the vector using nn.Linear, then
    reshape and use logsoftmax to normalize the probabilities within each group,
    then compute the likelihood.  However, this has a constraint that all the
    codebooks are predicted independently of each other.  This module allows you
    to predict them jointly, by regressing each codebook on all previous codebooks.
    This is done with a nonlinearity in which the previous codebook entries are combined
    with the input predictor vector, so that the regression is not purely
    linear.

    Args:
        predictor_dim: the number of features that we use to predict the codebook
               indexes, e.g. 2048 (will depend on your model).
        hidden_dim:  a hidden dimension in the model; should be more than
                codebook_size, but may be less or more than predictor_dim.

        num_codebooks: the number of codebooks that you are predicting;
               will likely be the same as the bytes_per_frame given to the
               QuantizerTrainer that you used to train the Quantizer you
               are predicting.
        codebook_size: number of entries per codebook (often 256)
        self_prediction: you can set this to false to enable prediction of
              codebooks by earlier-numbered codebooks
        hidden_dim: the hidden dimension per codebook (we use a 1-hidden-layer
              network, with a ReLU and then batchnorm).
        is_joint: if false, becomes a standard CE loss.
        checkpoint: if true, reduce backprop memory at the expense of doing
              the computation twice.
    """
    def __init__(self,
                 predictor_channels: int,
                 num_codebooks: int,
                 hidden_channels: int = 512,
                 codebook_size: int = 256,
                 reduction: str = 'sum',
                 ignore_index: int = -100,
                 is_joint: bool = True,
                 enable_dynamic_temp: bool = False,
                 checkpoint: bool = True):
        super(JointCodebookLoss, self).__init__()

        assert num_codebooks > 1 # we may later handle this specially.
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_channels = hidden_channels
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.is_joint = is_joint
        self.enable_dynamic_temp = enable_dynamic_temp
        self.checkpoint = checkpoint

        self.linear1 = nn.Linear(predictor_channels, hidden_channels)

        # codebook_embedding is used to predict each codebook from previous
        # codebooks, so it's a joint, not independent, model.  we'll multiply
        # this by hidden_channels ** 0.5 when we use it; this keeps the magnitude
        # small allows it to train fast enough (relatively speaking).
        self.codebook_embedding = nn.Embedding((num_codebooks - 1) * codebook_size,
                                               hidden_channels,
                                               _weight=torch.randn((num_codebooks - 1) * codebook_size,
                                                                   hidden_channels) * (hidden_channels ** -0.5))

        self.linear2_weight = nn.Parameter(torch.randn(num_codebooks, codebook_size,
                                                hidden_channels) * (hidden_channels ** -0.5))
        self.linear2b_weight = nn.Parameter(torch.randn(num_codebooks, codebook_size,
                                                predictor_channels) * (predictor_channels ** -0.5))
        self.linear2_bias = nn.Parameter(torch.zeros(num_codebooks, codebook_size))


    def forward(self,
                predictor: Tensor,
                codebook_indexes: Tensor,
                teacher_weights: List[float],) -> Tuple[Tensor, Tensor]:
        """
        Forward function.

        Args:
          predictor: a Tensor of some real type, with shape (*, predictor_channels).
          codebook_indexes:  a Tensor of integers, of shape (*, num_codebooks),
             where the '*' should be the same as for `predictor`.  It will be
             converted to type torch.int64.  Should contain indexes of codebook
             entries, in {0..codebook_size-1},
             or negative values which will be interpreted as "no codebook index here"
             (e.g. due to padding); we assume that each frame will either have
             all-negative or all-nonnegative indexes, meaning that (codebook_indexes >= 0)
             should not vary as you change the last index into it.

        Returns:
           cross_entropy_loss, will be a total negated log-probability, assuming
           reduction == 'sum'.
        """

        args = (predictor, codebook_indexes, teacher_weights,
                self.linear1.weight, self.linear1.bias,
                self.codebook_embedding.weight,
                self.linear2_weight,
                self.linear2b_weight,
                self.linear2_bias,
                self.ignore_index,
                self.is_joint,
                self.enable_dynamic_temp,
                self.reduction)
        if self.checkpoint:
            return checkpoint(joint_codebook_loss, *args)
        else:
            return joint_codebook_loss(*args)


# add multitask loss policy
class AutomaticWeightedLoss(nn.Module):
    """
    Params：
        num: int，the number of loss
        x: multi-task loss
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # params = torch.ones(num, requires_grad=True)
        params = torch.full((num,), 3.6, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, losses, opt):
        # loss_sum = 0
        weights = []
        new_losses = []
        if opt == "uncertainty":
            print("uncertainty 1/sigma^2 * loss + log(1 + sigma^2)")
            for i, loss in enumerate(losses):
                # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                loss = 1.0 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                weights.append(self.params[i])
                new_losses.append(loss)

        if opt == "uncertainty1":
            print("uncertainty 1/sigma^2 * loss + log(sigma)")
            for i, loss in enumerate(losses):
                # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                loss = 1.0 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
                weights.append(self.params[i])
                new_losses.append(loss)

        if opt == "uncertainty2":
            # print("uncertainty 0.5/sigma^2 * loss + log(1 + sigma^2)")
            for i, loss in enumerate(losses):
                # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                loss = 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                weights.append(self.params[i])
                new_losses.append(loss)

        return new_losses, weights
"""
awl = AutomaticWeightedLoss(2)	# we have 2 losses

# learnable parameters
optimizer = optim.Adam([
                {'params': model.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0}
            ])

for i in range(epoch):
    for data, label in data_loader:
        # forward and calculate losses
        loss1 = ...
        loss2 = ...
        # weigh losses
        loss_sum = awl(loss1, loss2)
        # backward
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
"""
import torch
from torch import nn
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class EncoderProjector(nn.Module):
    """
    The encoder projector module. It is used to project the encoder outputs to the same dimension as the language model.
    Modified from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/models/projector.py.
    Args:
        encoder_dim (:obj:`int`): The dimension of the encoder outputs.
        llm_dim (:obj:`int`): The dimension of the language model.
        downsample_rate (:obj:`int`, `optional`, defaults to 5): The downsample rate to use.
    """

    def __init__(self, encoder_dim, llm_dim, downsample_rate=5):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * self.downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):

        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SPEECH_LLM(nn.Module):
    """
    The Speech-to-Text model. It consists of an encoder, a language model and an encoder projector.
    The encoder is used to extract speech features from the input speech signal.
    The encoder projector is used to project the encoder outputs to the same dimension as the language model.
    The language model is used to generate the text from the speech features.
    Args:
        encoder (:obj:`nn.Module`): The encoder module.
        llm (:obj:`nn.Module`): The language model module.
        encoder_projector (:obj:`nn.Module`): The encoder projector module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        codec_lm: nn.Module = None,
        codec_lm_padding_side: str = "left",
    ):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.codec_lm = codec_lm
        if self.codec_lm:
            self.speech_token_projector = nn.Linear(
                self.llm.config.hidden_size, self.codec_lm.config.hidden_size
            )
            self.codec_lm_head = nn.Linear(
                self.codec_lm.config.hidden_size, self.codec_lm.config.vocab_size
            )
            self.loss_fct = torch.nn.CrossEntropyLoss()
            self.codec_lm_padding_side = codec_lm_padding_side

    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None
    ):
        """
        Merge the speech features with the input_ids and attention_mask. This is done by replacing the speech tokens
        with the speech features and padding the input_ids to the maximum length of the speech features.
        Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py#L277.
        Args:
            speech_features (:obj:`torch.Tensor`): The speech features to merge with the input_ids.
            inputs_embeds (:obj:`torch.Tensor`): The embeddings of the input_ids.
            input_ids (:obj:`torch.Tensor`): The input ids to merge.
            attention_mask (:obj:`torch.Tensor`): The attention mask to merge.
            labels (:obj:`torch.Tensor`, `optional`): The labels to merge.
        Returns:
            :obj:`Tuple(torch.Tensor)`: The merged embeddings, attention mask, labels and position ids.
        """
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.llm.config.pad_token_id)
        )
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == self.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != self.llm.config.default_speech_token_id
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                IGNORE_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_speech_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_speech_indices
            ]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= speech_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == self.llm.config.pad_token_id
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        fbank: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
    ):
        encoder_outs = self.encoder(fbank)

        speech_features = self.encoder_projector(encoder_outs)

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        (
            inputs_embeds,
            attention_mask,
            labels,
            _,
        ) = self._merge_input_ids_with_speech_features(
            speech_features, inputs_embeds, input_ids, attention_mask, labels
        )

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
        return model_outputs.loss, acc

    def forward_with_speech_output(
        self,
        fbank: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        speech_codec_ids: torch.LongTensor = None,
    ):
        encoder_outs = self.encoder(fbank)

        speech_features = self.encoder_projector(encoder_outs)

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        (
            inputs_embeds,
            attention_mask,
            labels,
            _,
        ) = self._merge_input_ids_with_speech_features(
            speech_features, inputs_embeds, input_ids, attention_mask, labels
        )

        # get the label start_index in inputs_embeds from labels
        text_label_start_index_list = []
        for i in range(labels.shape[0]):
            text_label_start_index = torch.where(labels[i] != IGNORE_TOKEN_ID)[0][0]
            text_label_start_index_list.append(text_label_start_index)

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, output_hidden_states=True
        )
        text_loss = model_outputs.loss

        # prepare codec lm inputs
        audio_codes_lens = torch.tensor(
            [len(x) for x in speech_codec_ids], dtype=torch.int64, device=input_ids.device
        )
        # print(audio_codes_lens, "audio_codes_lens")
        max_len_speech_codec = max(audio_codes_lens)
        delay_step = 2
        audio_codes = torch.full(
            (inputs_embeds.shape[0], max_len_speech_codec + inputs_embeds.shape[1] + 1),
            self.codec_lm.config.pad_token_id,
            dtype=torch.int64,
            device=input_ids.device
        )
        audio_labels = audio_codes.clone()
        total_len = audio_codes.shape[1]

        for i, speech_codec in enumerate(speech_codec_ids):
            text_label_start_index = text_label_start_index_list[i]
            speech_codec = torch.tensor(
                speech_codec, dtype=torch.int64, device=input_ids.device
            )
            speech_codec_len = len(speech_codec)

            # Calculate lengths of non-padding content
            codes_len = text_label_start_index + delay_step + 1 + speech_codec_len
            # Actual label content length (speech codec tokens + eos token)
            labels_actual_content_len = speech_codec_len + 1

            if self.codec_lm_padding_side == "right":
                # Fill audio_codes (right padding)
                codes_end_idx = text_label_start_index + delay_step + 1 + speech_codec_len
                audio_codes[i, :text_label_start_index + delay_step + 1] = self.codec_lm.config.bos_token_id # mask token_id
                audio_codes[i, text_label_start_index + delay_step + 1 : codes_end_idx] = speech_codec

                # Fill audio_labels (right padding)
                labels_start_idx = text_label_start_index + delay_step
                labels_speech_end_idx = labels_start_idx + speech_codec_len
                audio_labels[i, labels_start_idx : labels_speech_end_idx] = speech_codec
                audio_labels[i, labels_speech_end_idx] = self.codec_lm.config.eos_token_id

            elif self.codec_lm_padding_side == "left":
                # Calculate start indices for left padding (shifting content to the right)
                codes_start_idx = total_len - codes_len
                labels_start_idx = total_len - labels_actual_content_len # Start index for the actual label content

                # Fill audio_codes (left padding)
                codes_speech_start_idx = codes_start_idx + text_label_start_index + delay_step + 1
                audio_codes[i, codes_start_idx : codes_speech_start_idx] = self.codec_lm.config.bos_token_id # mask token_id
                audio_codes[i, codes_speech_start_idx : total_len] = speech_codec

                # Fill audio_labels (left padding)
                labels_speech_end_idx = labels_start_idx + speech_codec_len
                # Note: The beginning part remains pad_token_id
                audio_labels[i, labels_start_idx : labels_speech_end_idx] = speech_codec
                audio_labels[i, labels_speech_end_idx] = self.codec_lm.config.eos_token_id
            else:
                 raise ValueError(f"Unsupported padding side: {self.codec_lm_padding_side}")

        audio_attention_mask = audio_codes.ne(self.codec_lm.config.pad_token_id)
        audio_embeddings = self.codec_lm.get_input_embeddings()(audio_codes)
       
        # input_ids: seq_len T1, audio_codec seq_len T2
        text_last_hidden_outputs = model_outputs.hidden_states[-1]
        text_input_embeds = inputs_embeds + text_last_hidden_outputs
        text_input_embeds = self.speech_token_projector(text_input_embeds)

        T_merged = text_input_embeds.shape[1]
        T_audio = audio_embeddings.shape[1]

        if self.codec_lm_padding_side == "right":
            # Add to the beginning for right padding
            audio_embeddings[:, :T_merged] += text_input_embeds
        elif self.codec_lm_padding_side == "left":
            # Need to add to the shifted position for left padding
            # Calculate the length of the non-padded sequence for each item
            seq_lens = audio_attention_mask.sum(dim=1) # Shape (B)
            for i in range(audio_embeddings.shape[0]):
                item_len = seq_lens[i].item() # Get the non-padded length for item i
                start_idx_content = T_audio - item_len # Start index of the content for item i
                end_idx_target = start_idx_content + T_merged # End index of the target slice within the content
                # Add the text_input_embeds to the calculated slice
                audio_embeddings[i, start_idx_content:end_idx_target] += text_input_embeds[i]
        else:
             raise ValueError(f"Unsupported padding side: {self.codec_lm_padding_side}")

        speech_outputs = self.codec_lm(
            attention_mask=audio_attention_mask,
            inputs_embeds=audio_embeddings,
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden_state = speech_outputs.hidden_states[-1].clone()

        audio_logits = self.codec_lm_head(last_hidden_state) # shape, B, T, vocab_size
        audio_logits = audio_logits.contiguous().view(-1, self.codec_lm.config.vocab_size)
        audio_labels = audio_labels.contiguous().view(-1)
        audio_labels = audio_labels.masked_fill(
            audio_labels == self.codec_lm.config.pad_token_id, IGNORE_TOKEN_ID
        )
        codec_loss = self.loss_fct(audio_logits, audio_labels)
        audio_preds = torch.argmax(audio_logits, -1)

        
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
            audio_acc = compute_accuracy(
                audio_preds.detach(),
                audio_labels.detach(),
                ignore_label=IGNORE_TOKEN_ID,
            )


        return text_loss, acc, codec_loss, audio_acc
    
    def decode(
        self,
        fbank: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):

        encoder_outs = self.encoder(fbank)
        speech_features = self.encoder_projector(encoder_outs)
        speech_features = speech_features.to(torch.float16)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        (
            inputs_embeds,
            attention_mask,
            _,
            _,
        ) = self._merge_input_ids_with_speech_features(
            speech_features, inputs_embeds, input_ids, attention_mask
        )
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            num_beams=kwargs.get("num_beams", 1),
            do_sample=kwargs.get("do_sample", True),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 0.5),
            top_k=kwargs.get("top_k", 20),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            temperature=kwargs.get("temperature", 0.7),
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        # generated_ids = self.llm.generate(
        #     inputs_embeds=inputs_embeds,
        #     max_new_tokens=kwargs.get("max_new_tokens", 200),
        #     num_beams=kwargs.get("num_beams", 1),
        #     do_sample=kwargs.get("do_sample", False),
        #     min_length=kwargs.get("min_length", 1),
        #     top_p=kwargs.get("top_p", 1.0),
        #     repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        #     temperature=kwargs.get("temperature", 1.0),
        #     length_penalty=kwargs.get("length_penalty", 1.0),
        #     bos_token_id=self.llm.config.bos_token_id,
        #     eos_token_id=self.llm.config.eos_token_id,
        #     pad_token_id=self.llm.config.pad_token_id,
        # )
        return generated_ids


def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.
    Copied from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/utils/metric.py
    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()

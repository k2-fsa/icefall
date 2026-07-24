from typing import List, Tuple

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import logging


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
        encoder: nn.Module = None,
        llm: nn.Module = None,
        encoder_projector: nn.Module = None,
        codec_lm: nn.Module = None,
        codec_lm_padding_side: str = "left",
        teacher_llm: nn.Module = None,
        kl_temperature: float = 2.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.codec_lm = codec_lm
        if self.codec_lm:
            self.speech_token_projector = nn.Linear(
                self.llm.config.hidden_size + self.llm.config.hidden_size,
                self.codec_lm.config.hidden_size,
            )
            self.codec_lm_head = nn.Linear(
                self.codec_lm.config.hidden_size, self.codec_lm.config.vocab_size
            )
            self.speech_token_projector = self.speech_token_projector.to(
                dtype=torch.float16
            )
            self.codec_lm_head = self.codec_lm_head.to(dtype=torch.float16)
            self.loss_fct = torch.nn.CrossEntropyLoss()
            self.codec_lm_padding_side = codec_lm_padding_side

            self.audio_accuracy_metric = MulticlassAccuracy(
                self.codec_lm.vocab_size,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=IGNORE_TOKEN_ID,
            )
        if teacher_llm is not None:
            self.teacher_llm = teacher_llm
            self.kl_temperature = kl_temperature

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

    def forward_kl_div(
        self,
        fbank: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        teacher_input_ids: torch.LongTensor = None,
        teacher_attention_mask: torch.Tensor = None,
        teacher_labels: torch.LongTensor = None,
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

        teacher_outputs = self.teacher_llm(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
        )

        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(
                model_outputs.logits[labels != -100] / self.kl_temperature,
                dim=-1,
            ),
            torch.nn.functional.softmax(
                teacher_outputs.logits[teacher_labels != -100] / self.kl_temperature,
                dim=-1,
            ),
            reduction="batchmean",
        )

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            teacher_preds = torch.argmax(teacher_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
            acc_teacher = compute_accuracy(
                teacher_preds.detach()[:, :-1],
                teacher_labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
        return kl_loss, acc, acc_teacher

    def forward_with_speech_output(
        self,
        fbank: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        speech_codec_ids: torch.LongTensor = None,
    ):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        if fbank is not None:
            encoder_outs = self.encoder(fbank)
            speech_features = self.encoder_projector(encoder_outs)
            (
                inputs_embeds,
                attention_mask,
                labels,
                _,
            ) = self._merge_input_ids_with_speech_features(
                speech_features, inputs_embeds, input_ids, attention_mask, labels
            )

        input_seq_len = attention_mask.sum(dim=1)  # shape, B
        (
            text_label_start_index_list,
            text_input_start_index_list,
            input_question_len_list,
        ) = ([], [], [])
        for i in range(labels.shape[0]):
            input_embeds_valid_index = torch.where(attention_mask[i] != 0)[0]
            input_embeds_start_index = input_embeds_valid_index[0]
            text_labels_valid_index = torch.where(labels[i] != IGNORE_TOKEN_ID)[0]
            text_labels_start_index = text_labels_valid_index[0]

            assert (
                input_seq_len[i]
                == input_embeds_valid_index[-1] - input_embeds_start_index + 1
            ), f"input_seq_len: {input_seq_len[i]}, input_embeds_valid_index: {input_embeds_valid_index}, input_embeds_start_index: {input_embeds_start_index}"
            assert (
                input_embeds_valid_index[-1] == text_labels_valid_index[-1]
            ), f"input_embeds_valid_index: {input_embeds_valid_index}, text_labels_valid_index: {text_labels_valid_index}"
            input_question_len = text_labels_start_index - input_embeds_start_index
            assert (
                input_question_len
                + text_labels_valid_index[-1]
                - text_labels_start_index
                + 1
                == input_seq_len[i]
            )
            text_label_start_index_list.append(text_labels_start_index)
            text_input_start_index_list.append(input_embeds_start_index)
            input_question_len_list.append(input_question_len)

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        text_loss = model_outputs.loss
        delay_step = 1
        # prepare codec lm inputs
        audio_codes_lens = [
            len(x) + input_question_len_list[i] + delay_step + 1
            for i, x in enumerate(speech_codec_ids)
        ]
        max_len_speech_codec = max(audio_codes_lens)

        if self.codec_lm_padding_side == "right":
            audio_codes = [
                [self.codec_lm.config.mask_token_id]
                * (input_question_len_list[i] + delay_step)
                + [self.codec_lm.config.bos_token_id]
                + x
                + [self.codec_lm.config.pad_token_id]
                * (max_len_speech_codec - audio_codes_lens[i])
                for i, x in enumerate(speech_codec_ids)
            ]
            audio_labels = [
                [self.codec_lm.config.pad_token_id]
                * (input_question_len_list[i] + delay_step)
                + x
                + [self.codec_lm.config.eos_token_id]
                + [self.codec_lm.config.pad_token_id]
                * (max_len_speech_codec - audio_codes_lens[i])
                for i, x in enumerate(speech_codec_ids)
            ]
        elif self.codec_lm_padding_side == "left":
            audio_codes = [
                [self.codec_lm.config.pad_token_id]
                * (max_len_speech_codec - audio_codes_lens[i])
                + [self.codec_lm.config.mask_token_id]
                * (input_question_len_list[i] + delay_step)
                + [self.codec_lm.config.bos_token_id]
                + x
                for i, x in enumerate(speech_codec_ids)
            ]
            audio_labels = [
                [self.codec_lm.config.pad_token_id]
                * (max_len_speech_codec - audio_codes_lens[i])
                + [self.codec_lm.config.pad_token_id]
                * (input_question_len_list[i] + delay_step)
                + x
                + [self.codec_lm.config.eos_token_id]
                for i, x in enumerate(speech_codec_ids)
            ]
        audio_codes = torch.tensor(
            audio_codes, dtype=torch.int64, device=input_ids.device
        )
        audio_labels = torch.tensor(
            audio_labels, dtype=torch.int64, device=input_ids.device
        )

        audio_attention_mask = audio_codes.ne(self.codec_lm.config.pad_token_id)
        audio_embeddings = self.codec_lm.get_input_embeddings()(audio_codes)

        # text_last_hidden_lists, text_embeds_list, text_input_embeds_list = [], [], []
        text_input_embeds_list = []
        for i in range(len(text_label_start_index_list)):
            text_last_hidden = model_outputs.hidden_states[-1][
                i,
                text_input_start_index_list[i] : text_input_start_index_list[i]
                + input_seq_len[i]
                - 1,
            ]
            # text_last_hidden_lists.append(text_last_hidden)
            text_embed = inputs_embeds[
                i,
                text_input_start_index_list[i]
                + 1 : text_input_start_index_list[i]
                + input_seq_len[i],
            ]  # exclude bos
            # text_embeds_list.append(text_embed)

            text_input_embeds = torch.cat(
                [
                    text_last_hidden,
                    text_embed,
                ],
                dim=-1,
            )  # shape, T, D1 + D2
            text_input_embeds = self.speech_token_projector(
                text_input_embeds
            )  # shape, T, D_codec
            text_input_embeds_list.append(text_input_embeds)

        for i in range(audio_embeddings.shape[0]):
            text_input_embeds = text_input_embeds_list[i]
            if self.codec_lm_padding_side == "right":
                audio_embeddings[i, : text_input_embeds.shape[0]] += text_input_embeds
            elif self.codec_lm_padding_side == "left":
                start_idx = torch.where(
                    audio_codes[i] == self.codec_lm.config.mask_token_id
                )[0][0]
                start_idx_re_compute = torch.where(audio_attention_mask[i] != 0)[0][0]
                assert (
                    start_idx == start_idx_re_compute
                ), f"start_idx: {start_idx}, start_idx_re_compute: {start_idx_re_compute}"
                if text_input_embeds.shape[0] > audio_embeddings.shape[1] - start_idx:
                    logging.warning(
                        f"Truncate text_input_embeds: {text_input_embeds.shape} to {audio_embeddings.shape[1] - start_idx}\naudio_codes_lens: {audio_codes_lens[i]}\ninput_question_len_list: {input_question_len_list[i]}\ninput_seq_len: {input_seq_len[i]}\n"
                    )
                    # breakpoint()
                    text_input_embeds = text_input_embeds[
                        : audio_embeddings.shape[1] - start_idx
                    ]
                audio_embeddings[
                    i, start_idx : start_idx + text_input_embeds.shape[0]
                ] += text_input_embeds

        speech_outputs = self.codec_lm(
            attention_mask=audio_attention_mask,
            inputs_embeds=audio_embeddings,
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden_state = speech_outputs.hidden_states[-1].clone()

        audio_logits = self.codec_lm_head(last_hidden_state)  # shape, B, T, vocab_size
        audio_logits = audio_logits.contiguous().view(
            -1, self.codec_lm.config.vocab_size
        )
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
            audio_topk_acc = self.audio_accuracy_metric(
                audio_logits.detach(), audio_labels.detach()
            ).item()

        return text_loss, acc, codec_loss, audio_acc, audio_topk_acc

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

        return generated_ids

    def decode_with_speech_output(
        self,
        fbank: torch.Tensor = None,
        input_ids: torch.LongTensor = None,  # Prompt input_ids
        attention_mask: torch.Tensor = None,  # Prompt attention_mask
        max_text_new_tokens: int = 1024,
        max_speech_new_tokens: int = 2048,  # Max length for speech tokens
        llm_kwargs: dict = None,  # Kwargs for text LLM generate
        codec_lm_kwargs: dict = None,  # Kwargs for codec LM (e.g., temperature for sampling) - NOT IMPLEMENTED YET
    ) -> Tuple[torch.LongTensor, List[List[int]]]:
        """
        Generates text and corresponding speech tokens using the revised logic.

        Args:
            fbank: Input audio features.
            input_ids: Input token IDs for the text prompt.
            attention_mask: Attention mask for the text prompt.
            max_text_new_tokens: Max new tokens for text generation.
            max_speech_new_tokens: Max new tokens for speech generation.
            llm_kwargs: Additional arguments for self.llm.generate.
            codec_lm_kwargs: Additional arguments for self.codec_lm.generate.

        Returns:
            Tuple[torch.LongTensor, List[List[int]]]:
                - generated_text_ids: Tensor of generated text token IDs (including prompt).
                - generated_speech_tokens: List of lists, where each inner list contains
                                           the generated speech codec tokens for a batch item.
        """
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for speech generation."

        device = next(self.parameters()).device  # Use model's device

        prompt_embeds = self.llm.get_input_embeddings()(input_ids)

        # Merge speech features with prompt embeddings
        if fbank is not None:
            encoder_outs = self.encoder(fbank)
            speech_features = self.encoder_projector(encoder_outs)
            speech_features = speech_features.to(self.llm.dtype)  # Ensure matching dtype
            (
                merged_prompt_inputs_embeds,
                merged_prompt_attention_mask,
                _,
                _,
            ) = self._merge_input_ids_with_speech_features(
                speech_features, prompt_embeds, input_ids, attention_mask
            )
        else:
            merged_prompt_inputs_embeds = prompt_embeds
            merged_prompt_attention_mask = attention_mask

        # --- 2. Generate Text using LLM ---
        # Use merged embeds/mask as input to generate
        # Ensure kwargs passed are suitable for llm.generate
        # Note: Using default generation params from `decode` if not provided in kwargs
        final_llm_kwargs = {
            "bos_token_id": self.llm.config.bos_token_id,
            "eos_token_id": self.llm.config.eos_token_id,
            "pad_token_id": self.llm.config.pad_token_id,
            "num_beams": 1,
            "do_sample": True,  # Typically false for S2ST/S2TT tasks unless exploration needed
            "top_p": 0.5,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "temperature": 0.7,
            **(llm_kwargs or {}),  # User-provided kwargs override defaults
        }

        text_outputs = self.llm.generate(
            inputs_embeds=merged_prompt_inputs_embeds,
            attention_mask=merged_prompt_attention_mask,
            max_new_tokens=max_text_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            **final_llm_kwargs,
        )
        delay_step = 1
        generated_text_ids = text_outputs.sequences  # [B, S_full]
        eos_token_id = self.llm.config.eos_token_id
        eos_token_embedding = self.llm.get_input_embeddings()(
            torch.tensor([[eos_token_id]], device=device)
        )
        assert (
            generated_text_ids[0, -1] == eos_token_id
        ), f"Last token is not EOS: {generated_text_ids[0, -1]} != {eos_token_id}"
        thinker_token_embeds_org = [
            token_hidden_states[0].to(self.llm.device)
            for token_hidden_states in text_outputs.hidden_states
        ]

        first_thinker_token_embed = torch.cat(
            [
                thinker_token_embeds_org[0][:, 1:],
                thinker_token_embeds_org[1],
            ],
            dim=1,
        )

        thinker_token_embeds = (
            [first_thinker_token_embed]
            + thinker_token_embeds_org[2:]
            + [eos_token_embedding]
        )
        thinker_hidden_states = [
            token_hidden_states[-1].to(self.llm.device)
            for token_hidden_states in text_outputs.hidden_states
        ]

        thinker_reply_part = [
            torch.cat(
                [
                    thinker_hidden_state,
                    thinker_token_embed,
                ],
                dim=-1,
            )
            for thinker_hidden_state, thinker_token_embed in zip(
                thinker_hidden_states[1:], thinker_token_embeds[1:]
            )
        ]
        thinker_reply_part = torch.cat(thinker_reply_part, dim=1)
        # thinker_prompt_part = thinker_hidden_states[0] + thinker_token_embeds[0]
        thinker_prompt_part = torch.cat(
            [
                thinker_hidden_states[0],
                thinker_token_embeds[0],
            ],
            dim=-1,
        )

        thinker_prompt_part = self.speech_token_projector(thinker_prompt_part)
        thinker_reply_part = self.speech_token_projector(thinker_reply_part)

        thinker_prompt_part_seq_len = thinker_prompt_part.shape[1]
        talker_input_ids = torch.full(
            (batch_size, thinker_prompt_part_seq_len + delay_step + 1),
            self.codec_lm.config.mask_token_id,
            dtype=torch.long,
            device=self.llm.device,
        )
        talker_input_ids[:, -1] = self.codec_lm.config.bos_token_id
        talker_inputs_embeds = self.codec_lm.get_input_embeddings()(talker_input_ids)
        thinker_input_embeds = torch.cat(
            [
                thinker_prompt_part,
                thinker_reply_part[:, : delay_step + 1, :],
            ],
            dim=1,
        )
        talker_inputs_embeds += thinker_input_embeds
        thinker_reply_part = thinker_reply_part[:, delay_step + 1 :, :]

        past_key_values = None

        generated_speech_tokens_list = []
        next_token_ids = None

        for t in range(max_speech_new_tokens):
            if t > 0:
                talker_inputs_embeds = self.codec_lm.get_input_embeddings()(
                    next_token_ids
                )
                if thinker_reply_part.shape[1] > 0:
                    talker_inputs_embeds += thinker_reply_part[:, :1, :]
                    thinker_reply_part = thinker_reply_part[:, 1:, :]

            codec_outputs = self.codec_lm(
                inputs_embeds=talker_inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            last_token_hidden_state = codec_outputs.hidden_states[-1][:, -1, :]
            next_token_logits = self.codec_lm_head(last_token_hidden_state)

            next_token_ids = topk_sampling(
                next_token_logits,
            )
            if next_token_ids[0, 0] == self.codec_lm.config.eos_token_id:
                break

            past_key_values = codec_outputs.past_key_values  # Update KV cache
            generated_speech_tokens_list.append(
                next_token_ids.squeeze(1).cpu().tolist()[0]
            )

        return generated_text_ids, generated_speech_tokens_list


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


def topk_sampling(
    logits,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
):
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits_filtered = top_k_top_p_filtering(
        logits.clone(), top_k=top_k, top_p=top_p, min_tokens_to_keep=2
    )
    # Sample
    probs = torch.nn.functional.softmax(logits_filtered, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1)

    return tokens


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=20, top_p=0.5, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

import torch
import torch.nn as nn
import abc
import copy
from collections import OrderedDict


class Ffn(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, nlayers=1, drop_out=0.1, skip=False) -> None:
        super().__init__()

        layers = []
        for ilayer in range(nlayers):
            _in = hidden_dim if ilayer > 0 else input_dim
            _out = hidden_dim if ilayer < nlayers - 1 else out_dim
            layers.extend([
                nn.Linear(_in, _out),
                # nn.ReLU(),
                # nn.Sigmoid(),
                nn.Tanh(),
                nn.Dropout(p=drop_out),
            ])
        self.ffn = torch.nn.Sequential(
            *layers,
        )

        self.skip = skip
    
    def forward(self, x) -> torch.Tensor:
        x_out = self.ffn(x)
        
        if self.skip:
            x_out = x_out + x

        return x_out


class ContextEncoder(torch.nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

        self.stats_num_distractors_per_utt = 0
        self.stats_num_utt = 0

    @abc.abstractmethod
    def forward(
        self, 
        word_list, 
        word_lengths,
        is_encoder_side=None,
    ):
        pass

    def embed_contexts(
        self, 
        contexts,
        is_encoder_side=None,
    ):
        """
        Args:
            contexts: 
                The contexts, see below for details
        Returns:
            final_h:
                A tensor of shape (batch_size, max(num_words_per_utt) + 1, joiner_dim),
                which is the embedding for each context word.
            mask_h:
                A tensor of shape (batch_size, max(num_words_per_utt) + 1),
                which contains a True/False mask for final_h
        """
        if contexts["mode"] == "get_context_word_list":
            """
            word_list: 
                Option1: A list of words, where each word is a list of token ids.
                The list of tokens for each word has been padded.
                Option2: A list of words, where each word is an embedding.
            word_lengths:
                Option1: The number of tokens per word
                Option2: None
            num_words_per_utt:
                The number of words in the context for each utterance
            """
            word_list, word_lengths, num_words_per_utt = \
                contexts["word_list"], contexts["word_lengths"], contexts["num_words_per_utt"]

            assert word_lengths is None or word_list.size(0) == len(word_lengths)
            batch_size = len(num_words_per_utt)
        elif contexts["mode"] == "get_context_word_list_shared":
            """
            word_list: 
                Option1: A list of words, where each word is a list of token ids.
                The list of tokens for each word has been padded.
                Option2: A list of words, where each word is an embedding.
            word_lengths:
                Option1: The number of tokens per word
                Option2: None
            positive_mask_list:
                For each utterance, it contains a list of indices of the words should be masked
            """
            # word_list, word_lengths, positive_mask_list = \
            #     contexts["word_list"], contexts["word_lengths"], contexts["positive_mask_list"]
            # batch_size = len(positive_mask_list)
            word_list, word_lengths, num_words_per_utt = \
                contexts["word_list"], contexts["word_lengths"], contexts["num_words_per_utt"]
            batch_size = len(num_words_per_utt)
            
            assert word_lengths is None or word_list.size(0) == len(word_lengths)
        else:
            raise NotImplementedError

        # print(f"word_list.shape={word_list.shape}")
        final_h = self.forward(word_list, word_lengths, is_encoder_side=is_encoder_side)

        if contexts["mode"] == "get_context_word_list":
            final_h = torch.split(final_h, num_words_per_utt)
            final_h = torch.nn.utils.rnn.pad_sequence(
                final_h, 
                batch_first=True, 
                padding_value=0.0
            )
            # print(f"final_h.shape={final_h.shape}")

            # add one no-bias token
            no_bias_h = torch.zeros(final_h.shape[0], 1, final_h.shape[-1])
            no_bias_h = no_bias_h.to(final_h.device)
            final_h = torch.cat((no_bias_h, final_h), dim=1)
            # print(final_h)

            # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
            mask_h = torch.arange(max(num_words_per_utt) + 1)
            mask_h = mask_h.expand(len(num_words_per_utt), max(num_words_per_utt) + 1) > torch.Tensor(num_words_per_utt).unsqueeze(1)
            mask_h = mask_h.to(final_h.device)

            num_utt = len(num_words_per_utt)
            self.stats_num_distractors_per_utt = len(word_list) / (num_utt + self.stats_num_utt) + self.stats_num_utt / (num_utt + self.stats_num_utt) * self.stats_num_distractors_per_utt
            self.stats_num_utt += num_utt
        elif contexts["mode"] == "get_context_word_list_shared":
            no_bias_h = torch.zeros(1, final_h.shape[-1])
            no_bias_h = no_bias_h.to(final_h.device)
            final_h = torch.cat((no_bias_h, final_h), dim=0)

            final_h = final_h.expand(batch_size, -1, -1)

            # mask_h = torch.full(False, (batch_size, final_h.shape(1)))  # TODO
            # for i, my_mask in enumerate(positive_mask_list):
            #     if len(my_mask) > 0:
            #         my_mask = torch.Tensor(my_mask, dtype=int)
            #         my_mask += 1
            #         mask_h[i][my_mask] = True
            mask_h = None

            num_utt = len(num_words_per_utt)
            self.stats_num_distractors_per_utt = num_utt / (num_utt + self.stats_num_utt) * len(word_list) + self.stats_num_utt / (num_utt + self.stats_num_utt) * self.stats_num_distractors_per_utt
            self.stats_num_utt += num_utt


        # TODO: validate this shape is correct:
        # final_h:  batch_size * max_num_words_per_utt + 1 * dim
        # mask_h:   batch_size * max_num_words_per_utt + 1
        return final_h, mask_h

    def clustering(self):
        pass

    def cache(self):
        pass


class ContextEncoderLSTM(ContextEncoder):
    def __init__(
        self,
        vocab_size: int = None,
        context_encoder_dim: int = None,
        embedding_layer: nn.Module = None,
        output_dim: int = None,
        num_layers: int = None,
        num_directions: int = None,
        drop_out: float = 0.1,
        bi_encoders: bool = False,
    ):
        super(ContextEncoderLSTM, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.context_encoder_dim = context_encoder_dim

        if embedding_layer is not None:
            # self.embed = embedding_layer
            self.embed = torch.nn.Embedding(
                vocab_size, 
                context_encoder_dim
            )
            self.embed.weight.data = embedding_layer.weight.data
        else:
            self.embed = torch.nn.Embedding(
                vocab_size, 
                context_encoder_dim
            )
        
        self.rnn = torch.nn.LSTM(
            input_size=self.embed.weight.shape[1], 
            hidden_size=context_encoder_dim, 
            num_layers=self.num_layers,
            batch_first=True, 
            bidirectional=(self.num_directions == 2), 
            dropout=0.1 if self.num_layers > 1 else 0
        )
        self.linear = torch.nn.Linear(
            context_encoder_dim * self.num_directions, 
            output_dim
        )
        self.drop_out = torch.nn.Dropout(drop_out)

        # TODO: Do we need some relu layer?
        # https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

        self.bi_encoders = bi_encoders
        if bi_encoders:
            # Create the decoder/predictor side of the context encoder
            self.embed_dec = copy.deepcopy(self.embed)
            self.rnn_dec = copy.deepcopy(self.rnn)
            self.linear_dec = copy.deepcopy(self.linear)
            self.drop_out_dec = copy.deepcopy(self.drop_out)

    def forward(
        self, 
        word_list, 
        word_lengths,
        is_encoder_side=None,
    ):
        if is_encoder_side is None or is_encoder_side is True:
            embed = self.embed
            rnn = self.rnn
            linear = self.linear
            drop_out = self.drop_out
        else:
            embed = self.embed_dec
            rnn = self.rnn_dec
            linear = self.linear_dec
            drop_out = self.drop_out_dec

        out = embed(word_list)
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, 
            batch_first=True, 
            lengths=word_lengths, 
            enforce_sorted=False
        )
        output, (hn, cn) = rnn(out)  # use default all zeros (h_0, c_0)

        # # https://discuss.pytorch.org/t/bidirectional-3-layer-lstm-hidden-output/41336/4
        # final_state = hn.view(
        #     self.num_layers, 
        #     self.num_directions,
        #     word_list.shape[0], 
        #     self.encoder_dim,
        # )[-1]  # Only the last layer
        # h_1, h_2 = final_state[0], final_state[1]
        # # X = h_1 + h_2                     # Add both states (needs different input size for first linear layer)
        # final_h = torch.cat((h_1, h_2), dim=1)  # Concatenate both states
        # final_h = self.linear(final_h)

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer.
        # hidden[-2, :, : ] is the last of the forwards RNN 
        # hidden[-1, :, : ] is the last of the backwards RNN
        h_1, h_2 = hn[-2, :, : ] , hn[-1, :, : ]
        final_h = torch.cat((h_1, h_2), dim=1)  # Concatenate both states
        final_h = linear(final_h)
        # final_h = drop_out(final_h)

        return final_h


class SimpleGLU(nn.Module):
    def __init__(self):
        super(SimpleGLU, self).__init__()
        # Initialize the learnable parameter 'a'
        self.a = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Perform the operation a * x
        return self.a * x


class BiasingModule(torch.nn.Module):
    def __init__(
        self, 
        query_dim,
        qkv_dim=64,
        num_heads=4,
    ):
        super(BiasingModule, self).__init__()
        self.proj_in1 = nn.Linear(query_dim, qkv_dim)
        self.proj_in2 = Ffn(
            input_dim=qkv_dim,
            hidden_dim=qkv_dim,
            out_dim=qkv_dim,
            skip=True,
            drop_out=0.1,
            nlayers=2,
        )
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=qkv_dim,
            num_heads=num_heads, 
            # kdim=64,
            # vdim=64,
            batch_first=True,
        )
        self.proj_out1 = Ffn(
            input_dim=qkv_dim,
            hidden_dim=qkv_dim,
            out_dim=qkv_dim,
            skip=True,
            drop_out=0.1,
            nlayers=2,
        )
        self.proj_out2 = nn.Linear(qkv_dim, query_dim)
        self.glu = nn.GLU()
        # self.glu = SimpleGLU()
        
        self.contexts = None
        self.contexts_mask = None

    # def __init__(
    #     self, 
    #     query_dim,
    #     qkv_dim=64,
    #     num_heads=4,
    # ):
    #     super(BiasingModule, self).__init__()
    #     self.proj_in1 = Ffn(
    #         input_dim=query_dim,
    #         hidden_dim=query_dim,
    #         out_dim=query_dim,
    #         skip=True,
    #         drop_out=0.1,
    #         nlayers=2,
    #     )
    #     self.proj_in2 = nn.Linear(query_dim, qkv_dim)
    #     self.multihead_attn = torch.nn.MultiheadAttention(
    #         embed_dim=qkv_dim,
    #         num_heads=num_heads, 
    #         # kdim=64,
    #         # vdim=64,
    #         batch_first=True,
    #     )
    #     self.proj_out1 = nn.Linear(qkv_dim, query_dim)
    #     self.proj_out2 = Ffn(
    #         input_dim=query_dim,
    #         hidden_dim=query_dim,
    #         out_dim=query_dim,
    #         skip=True,
    #         drop_out=0.1,
    #         nlayers=2,
    #     )
    #     self.glu = nn.GLU()
        
    #     self.contexts = None
    #     self.contexts_mask = None
    #     self.attn_output_weights = None

    def forward(
        self, 
        queries,
        contexts=None,
        contexts_mask=None,
        need_weights=False,
    ):
        """
        Args:
            query: 
                of shape batch_size * seq_length * query_dim
            contexts: 
                of shape batch_size * max_contexts_size * query_dim
            contexts_mask:
                of shape batch_size * max_contexts_size
        Returns:
            attn_output:
                of shape batch_size * seq_length * context_dim
        """
        
        if contexts is None:
            contexts = self.contexts
        if contexts_mask is None:
            contexts_mask = self.contexts_mask

        _queries = self.proj_in1(queries)
        _queries = self.proj_in2(_queries)
        # _queries = _queries / 0.01
        
        attn_output, attn_output_weights = self.multihead_attn(
            _queries,    # query
            contexts,   # key
            contexts,   # value
            key_padding_mask=contexts_mask,
            need_weights=need_weights,
        )
        biasing_output = self.proj_out1(attn_output)
        biasing_output = self.proj_out2(biasing_output)

        # apply the gated linear unit
        biasing_output = self.glu(biasing_output.repeat(1,1,2))
        # biasing_output = self.glu(biasing_output)
        
        # inject contexts here
        output = queries + biasing_output

        # print(f"query={query.shape}")
        # print(f"value={contexts} value.shape={contexts.shape}")
        # print(f"attn_output_weights={attn_output_weights} attn_output_weights.shape={attn_output_weights.shape}")
        
        return output, attn_output_weights


def tuple_to_list(t):
    if isinstance(t, tuple):
        return list(map(tuple_to_list, t))
    return t


def list_to_tuple(l):
    if isinstance(l, list):
        return tuple(map(list_to_tuple, l))
    return l


class ContextualSequential(nn.Sequential):
    def __init__(self, *args):
        super(ContextualSequential, self).__init__(*args)
        
        self.contexts_h = None
        self.contexts_mask = None
    
    def set_contexts(self, contexts_h, contexts_masks):
        self.contexts_h, self.contexts_mask = contexts_h, contexts_masks
       
    def forward(self, *args, **kwargs):
        # print(f"input: {type(args[0])=}, {args[0].shape=}")
        
        is_hf = False
        
        for module in self._modules.values():
            module_name = type(module).__name__
            # if "AudioEncoder" in module_name:
            #     args = (module(*args, **kwargs),)
            # elif "TextDecoder" in module_name:
            #     args = (module(*args, **kwargs),)
            # elif "WhisperDecoderLayer" in module_name:
            #     args = (module(*args, **kwargs),)
            # elif "WhisperEncoderLayer" in module_name:
            #     args = (module(*args, **kwargs),)
            
            if "WhisperDecoderLayer" in module_name or "WhisperEncoderLayer" in module_name:
                is_hf = True

            if "BiasingModule" in module_name:
                x = args[0]
                while isinstance(x, list) or isinstance(x, tuple):
                    x = x[0]
                
                if self.contexts_h is not None:
                    # The contexts are injected here
                    x, attn_output_weights = module(x, contexts=self.contexts_h, contexts_mask=self.contexts_mask, need_weights=True)
                else:
                    # final_h:  batch_size * max_num_words_per_utt + 1 * dim
                    # mask_h:   batch_size * max_num_words_per_utt + 1

                    batch_size = x.size(0)
                    contexts_h = torch.zeros(batch_size, 1, module.multihead_attn.embed_dim)
                    contexts_h = contexts_h.to(x.device)
                    
                    contexts_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
                    contexts_mask = contexts_mask.to(x.device)
                    
                    x, attn_output_weights = module(x, contexts=contexts_h, contexts_mask=contexts_mask, need_weights=True)
                
                args = (x,)
            else:
                x = module(*args, **kwargs)
                
                while isinstance(x, list) or isinstance(x, tuple):
                    x = x[0]
                args = (x,)

        # print(f"output: {type(args[0])=}, {args[0].shape=}")
        if is_hf:
            return args
        else:
            return args[0]


def set_contexts_for_model(model, contexts):
    # check each module in the model, if it is a class of "ContextualSequential",
    # then set the contexts for the module

    # if hasattr(model, "context_encoder") and model.context_encoder is not None:
    contexts_h, contexts_mask = model.context_encoder.embed_contexts(
        contexts
    )
    
    for module in model.modules():
        if isinstance(module, ContextualSequential):
            module.set_contexts(contexts_h, contexts_mask)


def get_contextual_model(model, encoder_biasing_layers="31,", decoder_biasing_layers="31,", context_dim=128) -> nn.Module:
    # context_dim = 128  # 1.5%
    # context_dim = 256  # 5.22% => seems better?
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # [(n, param.requires_grad) for n, param in context_encoder.named_parameters()]
    # [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    
    print("Before neural biasing:")
    print(f"{total_params=}")
    print(f"{trainable_params=} ({trainable_params/total_params*100:.2f}%)")
    
    encoder_biasing_layers = [int(l) for l in encoder_biasing_layers.strip().split(",") if len(l) > 0]
    decoder_biasing_layers = [int(l) for l in decoder_biasing_layers.strip().split(",") if len(l) > 0]

    if len(encoder_biasing_layers) > 0 or len(decoder_biasing_layers) > 0:
        if hasattr(model, "model"):  # Hugegingface models
            embedding_layer = model.model.decoder.embed_tokens
        else:
            embedding_layer = model.decoder.token_embedding
            
        context_encoder = ContextEncoderLSTM(
            # vocab_size=embedding_layer.weight.shape[0],
            embedding_layer=embedding_layer,
            # context_encoder_dim=int(params.encoder_dims.split(",")[-1]),
            context_encoder_dim=context_dim,
            output_dim=context_dim,
            num_layers=2,
            num_directions=2,
            drop_out=0.1,
        )
        model.context_encoder = context_encoder

    # encoder_biasing_adapter = BiasingModule(
    #     query_dim=int(params.encoder_dims.split(",")[-1]),
    #     qkv_dim=context_dim,
    #     num_heads=4,
    # )
    
    if hasattr(model, "model") and hasattr(model.model.encoder, "layers"):  # Huggingface models
        for i, layer in enumerate(model.model.encoder.layers):
            if i in encoder_biasing_layers:
                layer_output_dim = layer.final_layer_norm.normalized_shape[0]
                model.model.encoder.layers[i] = ContextualSequential(OrderedDict([
                    ("layer", layer),
                    ("biasing_adapter", BiasingModule(
                        query_dim=layer_output_dim,
                        qkv_dim=context_dim,
                        num_heads=4,
                    ))
                ]))
    elif hasattr(model.encoder, "blocks"):  # OpenAI models
        for i, layer in enumerate(model.encoder.blocks):
            if i in encoder_biasing_layers:
                layer_output_dim = layer.mlp_ln.normalized_shape[0]
                model.encoder.blocks[i] = ContextualSequential(OrderedDict([
                    ("layer", layer),
                    ("biasing_adapter", BiasingModule(
                        query_dim=layer_output_dim,
                        qkv_dim=context_dim,
                        num_heads=4,
                    ))
                ]))
    else:
        raise NotImplementedError
        
    if hasattr(model, "model") and hasattr(model.model.decoder, "layers"):
        for i, layer in enumerate(model.model.decoder.layers):
            if i in decoder_biasing_layers:
                layer_output_dim = layer.final_layer_norm.normalized_shape[0]
                model.model.decoder.layers[i] = ContextualSequential(OrderedDict([
                    ("layer", layer),
                    ("biasing_adapter", BiasingModule(
                        query_dim=layer_output_dim,
                        qkv_dim=context_dim,
                        num_heads=4,
                    ))
                ]))
    elif hasattr(model.decoder, "blocks"):
        for i, layer in enumerate(model.decoder.blocks):
            if i in decoder_biasing_layers:
                layer_output_dim = layer.mlp_ln.normalized_shape[0]
                model.decoder.blocks[i] = ContextualSequential(OrderedDict([
                    ("layer", layer),
                    ("biasing_adapter", BiasingModule(
                        query_dim=layer_output_dim,
                        qkv_dim=context_dim,
                        num_heads=4,
                    ))
                ]))
    else:
        raise NotImplementedError

    # Freeze the model params
    # exception_types = (BiasingModule, ContextEncoderLSTM)
    for name, param in model.named_parameters():
        # Check if the parameter belongs to a layer of the specified types
        if "biasing_adapter" in name:
            param.requires_grad = True
        elif "context_encoder" in name and "context_encoder.embed" not in name:  # We will not fine-tune the embedding layer, which comes from the original model
            param.requires_grad = True
        elif "context_encoder" in name and "context_encoder.embed" in name:  # Debug
            param.requires_grad = False
        else:
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # [(n, param.requires_grad) for n, param in context_encoder.named_parameters()]
    # [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    
    print("Neural biasing:")
    print(f"{total_params=}")
    print(f"{trainable_params=} ({trainable_params/total_params*100:.2f}%)")
    
    return model


# # Test:
# import whisper, torch; device = torch.device("cuda", 0)
# model = whisper.load_model("large-v2", is_ctx=True, device=device)
# from neural_biasing import get_contextual_model
# model1 = get_contextual_model(model)

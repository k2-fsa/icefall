import torch


def test_emformer_attention_forward():
    from emformer import EmformerAttention

    B, D = 2, 256
    U, R = 12, 2
    chunk_length = 2
    attention = EmformerAttention(embed_dim=D, nhead=8)

    for use_memory in [True, False]:
        if use_memory:
            S = U // chunk_length
            M = S - 1
        else:
            S, M = 0, 0

        Q, KV = R + U + S, M + R + U
        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        summary = torch.randn(S, B, D)
        memory = torch.randn(M, B, D)
        attention_mask = torch.rand(Q, KV) >= 0.5

        output_right_context_utterance, output_memory = attention(
            utterance,
            lengths,
            right_context,
            summary,
            memory,
            attention_mask,
        )
        assert output_right_context_utterance.shape == (R + U, B, D)
        assert output_memory.shape == (M, B, D)


def test_emformer_attention_infer():
    from emformer import EmformerAttention

    B, D = 2, 256
    R, L = 4, 2
    chunk_length = 2
    U = chunk_length
    attention = EmformerAttention(embed_dim=D, nhead=8)

    for use_memory in [True, False]:
        if use_memory:
            S, M = 1, 3
        else:
            S, M = 0, 0

        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        summary = torch.randn(S, B, D)
        memory = torch.randn(M, B, D)
        left_context_key = torch.randn(L, B, D)
        left_context_val = torch.randn(L, B, D)

        (
            output_right_context_utterance,
            output_memory,
            next_key,
            next_val,
        ) = attention.infer(
            utterance,
            lengths,
            right_context,
            summary,
            memory,
            left_context_key,
            left_context_val,
        )
        assert output_right_context_utterance.shape == (R + U, B, D)
        assert output_memory.shape == (S, B, D)
        assert next_key.shape == (L + U, B, D)
        assert next_val.shape == (L + U, B, D)


def test_emformer_layer_forward():
    from emformer import EmformerLayer

    B, D = 2, 256
    U, R, L = 12, 2, 5
    chunk_length = 2

    for use_memory in [True, False]:
        if use_memory:
            S = U // chunk_length
            M = S - 1
        else:
            S, M = 0, 0

        layer = EmformerLayer(
            d_model=D,
            nhead=8,
            dim_feedforward=1024,
            chunk_length=chunk_length,
            cnn_module_kernel=3,
            left_context_length=L,
            max_memory_size=M,
            causal=True,
        )

        Q, KV = R + U + S, M + R + U
        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        memory = torch.randn(M, B, D)
        attention_mask = torch.rand(Q, KV) >= 0.5

        output_utterance, output_right_context, output_memory = layer(
            utterance,
            lengths,
            right_context,
            memory,
            attention_mask,
        )
        assert output_utterance.shape == (U, B, D)
        assert output_right_context.shape == (R, B, D)
        assert output_memory.shape == (M, B, D)


def test_emformer_layer_infer():
    from emformer import EmformerLayer

    B, D = 2, 256
    R, L = 2, 5
    chunk_length = 2
    U = chunk_length
    K = 3

    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0

        layer = EmformerLayer(
            d_model=D,
            nhead=8,
            dim_feedforward=1024,
            chunk_length=chunk_length,
            cnn_module_kernel=K,
            left_context_length=L,
            max_memory_size=M,
            causal=True,
        )

        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        memory = torch.randn(M, B, D)
        state = None
        conv_cache = None
        (
            output_utterance,
            output_right_context,
            output_memory,
            output_state,
            output_conv_cache,
        ) = layer.infer(
            utterance, lengths, right_context, memory, state, conv_cache
        )
        assert output_utterance.shape == (U, B, D)
        assert output_right_context.shape == (R, B, D)
        if use_memory:
            assert output_memory.shape == (1, B, D)
        else:
            assert output_memory.shape == (0, B, D)
        assert len(output_state) == 4
        assert output_state[0].shape == (M, B, D)
        assert output_state[1].shape == (L, B, D)
        assert output_state[2].shape == (L, B, D)
        assert output_state[3].shape == (1, B)
        assert output_conv_cache.shape == (B, D, K - 1)


def test_emformer_encoder_forward():
    from emformer import EmformerEncoder

    B, D = 2, 256
    U, R, L = 12, 2, 5
    chunk_length = 2

    for use_memory in [True, False]:
        if use_memory:
            S = U // chunk_length
            M = S - 1
        else:
            S, M = 0, 0

        encoder = EmformerEncoder(
            chunk_length=chunk_length,
            d_model=D,
            dim_feedforward=1024,
            num_encoder_layers=2,
            cnn_module_kernel=3,
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            causal=True,
        )

        x = torch.randn(U + R, B, D)
        lengths = torch.randint(1, U + R + 1, (B,))
        lengths[0] = U + R

        output, output_lengths = encoder(x, lengths)
        assert output.shape == (U, B, D)
        assert torch.equal(output_lengths, torch.clamp(lengths - R, min=0))


def test_emformer_encoder_infer():
    from emformer import EmformerEncoder

    B, D = 2, 256
    R, L = 2, 5
    chunk_length = 2
    U = chunk_length
    num_chunks = 3
    num_encoder_layers = 2
    K = 3

    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0

        encoder = EmformerEncoder(
            chunk_length=chunk_length,
            d_model=D,
            dim_feedforward=1024,
            num_encoder_layers=num_encoder_layers,
            cnn_module_kernel=K,
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            causal=True,
        )

        states = None
        conv_caches = None
        for chunk_idx in range(num_chunks):
            x = torch.randn(U + R, B, D)
            lengths = torch.randint(1, U + R + 1, (B,))
            lengths[0] = U + R
            output, output_lengths, states, conv_caches = encoder.infer(
                x, lengths, states, conv_caches
            )
            assert output.shape == (U, B, D)
            assert torch.equal(output_lengths, torch.clamp(lengths - R, min=0))
            assert len(states) == num_encoder_layers
            for state in states:
                assert len(state) == 4
                assert state[0].shape == (M, B, D)
                assert state[1].shape == (L, B, D)
                assert state[2].shape == (L, B, D)
                assert torch.equal(
                    state[3], (chunk_idx + 1) * U * torch.ones_like(state[3])
                )
            for conv_cache in conv_caches:
                assert conv_cache.shape == (B, D, K - 1)


def test_emformer_forward():
    from emformer import Emformer

    num_features = 80
    output_dim = 1000
    chunk_length = 8
    L, R = 128, 4
    B, D, U = 2, 256, 80
    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0
        model = Emformer(
            num_features=num_features,
            output_dim=output_dim,
            chunk_length=chunk_length,
            subsampling_factor=4,
            d_model=D,
            cnn_module_kernel=3,
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            vgg_frontend=False,
            causal=True,
        )
        x = torch.randn(B, U + R + 3, num_features)
        x_lens = torch.randint(1, U + R + 3 + 1, (B,))
        x_lens[0] = U + R + 3
        logits, output_lengths = model(x, x_lens)
        assert logits.shape == (B, U // 4, output_dim)
        assert torch.equal(
            output_lengths,
            torch.clamp(((x_lens - 1) // 2 - 1) // 2 - R // 4, min=0),
        )


def test_emformer_infer():
    from emformer import Emformer

    num_features = 80
    output_dim = 1000
    chunk_length = 8
    U = chunk_length
    L, R = 128, 4
    B, D = 2, 256
    num_chunks = 3
    num_encoder_layers = 2
    K = 3
    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0
        model = Emformer(
            num_features=num_features,
            output_dim=output_dim,
            chunk_length=chunk_length,
            subsampling_factor=4,
            d_model=D,
            num_encoder_layers=num_encoder_layers,
            cnn_module_kernel=K,
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            vgg_frontend=False,
            causal=True,
        )
        states = None
        conv_caches = None
        for chunk_idx in range(num_chunks):
            x = torch.randn(B, U + R + 3, num_features)
            x_lens = torch.randint(1, U + R + 3 + 1, (B,))
            x_lens[0] = U + R + 3
            logits, output_lengths, states, conv_caches = model.infer(
                x, x_lens, states, conv_caches
            )
            assert logits.shape == (B, U // 4, output_dim)
            assert torch.equal(
                output_lengths,
                torch.clamp(((x_lens - 1) // 2 - 1) // 2 - R // 4, min=0),
            )
            assert len(states) == num_encoder_layers
            for state in states:
                assert len(state) == 4
                assert state[0].shape == (M, B, D)
                assert state[1].shape == (L // 4, B, D)
                assert state[2].shape == (L // 4, B, D)
                assert torch.equal(
                    state[3],
                    U // 4 * (chunk_idx + 1) * torch.ones_like(state[3]),
                )
            for conv_cache in conv_caches:
                assert conv_cache.shape == (B, D, K - 1)


def test_emformer_encoder_layer_forward_infer_consistency():
    from emformer import EmformerEncoder

    chunk_length = 4
    num_chunks = 3
    U = chunk_length * num_chunks
    L, R = 1, 2
    D = 256
    num_encoder_layers = 1
    memory_sizes = [0, 3]
    K = 3

    for M in memory_sizes:
        encoder = EmformerEncoder(
            chunk_length=chunk_length,
            d_model=D,
            dim_feedforward=1024,
            num_encoder_layers=num_encoder_layers,
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            dropout=0.1,
            cnn_module_kernel=K,
            causal=True,
        )
        encoder.eval()
        encoder_layer = encoder.emformer_layers[0]

        x = torch.randn(U + R, 1, D)
        lengths = torch.tensor([U])
        right_context = encoder._gen_right_context(x)
        utterance = x[: x.size(0) - R]
        attention_mask = encoder._gen_attention_mask(utterance)
        memory = (
            encoder.init_memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[
                :-1
            ]
            if encoder.use_memory
            else torch.empty(0).to(dtype=x.dtype, device=x.device)
        )
        (
            forward_output_utterance,
            forward_output_right_context,
            forward_output_memory,
        ) = encoder_layer(
            utterance,
            lengths,
            right_context,
            memory,
            attention_mask,
        )

        state = None
        conv_cache = None
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_length
            end_idx = start_idx + chunk_length
            chunk = x[start_idx:end_idx]
            chunk_right_context = x[end_idx : end_idx + R]  # noqa
            chunk_length = torch.tensor([chunk_length])
            chunk_memory = (
                encoder.init_memory_op(chunk.permute(1, 2, 0)).permute(2, 0, 1)
                if encoder.use_memory
                else torch.empty(0).to(dtype=x.dtype, device=x.device)
            )
            (
                infer_output_chunk,
                infer_right_context,
                infer_output_memory,
                state,
                conv_cache,
            ) = encoder_layer.infer(
                chunk,
                chunk_length,
                chunk_right_context,
                chunk_memory,
                state,
                conv_cache,
            )
            forward_output_chunk = forward_output_utterance[start_idx:end_idx]
            assert torch.allclose(
                infer_output_chunk,
                forward_output_chunk,
                atol=1e-5,
                rtol=0.0,
            )


def test_emformer_encoder_forward_infer_consistency():
    from emformer import EmformerEncoder

    chunk_length = 4
    num_chunks = 3
    U = chunk_length * num_chunks
    L, R = 1, 2
    D = 256
    num_encoder_layers = 3
    K = 3
    memory_sizes = [0, 3]

    for M in memory_sizes:
        encoder = EmformerEncoder(
            chunk_length=chunk_length,
            d_model=D,
            dim_feedforward=1024,
            num_encoder_layers=num_encoder_layers,
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            dropout=0.1,
            cnn_module_kernel=K,
            causal=True,
        )
        encoder.eval()

        x = torch.randn(U + R, 1, D)
        lengths = torch.tensor([U + R])

        forward_output, forward_output_lengths = encoder(x, lengths)

        states = None
        conv_caches = None
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_length
            end_idx = start_idx + chunk_length
            chunk = x[start_idx : end_idx + R]  # noqa
            chunk_right_context = x[end_idx : end_idx + R]  # noqa
            chunk_length = torch.tensor([chunk_length])
            (
                infer_output_chunk,
                infer_output_lengths,
                states,
                conv_caches,
            ) = encoder.infer(
                chunk,
                chunk_length,
                states,
                conv_caches,
            )
            forward_output_chunk = forward_output[start_idx:end_idx]
            assert torch.allclose(
                infer_output_chunk,
                forward_output_chunk,
                atol=1e-5,
                rtol=0.0,
            )


def test_emformer_forward_infer_consistency():
    from emformer import Emformer

    num_features = 80
    output_dim = 1000
    chunk_length = 8
    num_chunks = 3
    U = chunk_length * num_chunks
    L, R = 128, 4
    D = 256
    num_encoder_layers = 2
    K = 3
    memory_sizes = [0, 3]

    for M in memory_sizes:
        model = Emformer(
            num_features=num_features,
            output_dim=output_dim,
            chunk_length=chunk_length,
            subsampling_factor=4,
            d_model=D,
            num_encoder_layers=num_encoder_layers,
            cnn_module_kernel=K,
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            dropout=0.1,
            vgg_frontend=False,
            causal=True,
        )
        model.eval()

        x = torch.randn(1, U + R + 3, num_features)
        x_lens = torch.tensor([x.size(1)])

        # forward mode
        forward_logits, _ = model(x, x_lens)

        states = None
        conv_caches = None
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_length
            end_idx = start_idx + chunk_length
            chunk = x[:, start_idx : end_idx + R + 3]  # noqa
            lengths = torch.tensor([chunk.size(1)])
            (
                infer_chunk_logits,
                output_lengths,
                states,
                conv_caches,
            ) = model.infer(chunk, lengths, states, conv_caches)
            forward_chunk_logits = forward_logits[
                :, start_idx // 4 : end_idx // 4  # noqa
            ]
            assert torch.allclose(
                infer_chunk_logits,
                forward_chunk_logits,
                atol=1e-5,
                rtol=0.0,
            )


if __name__ == "__main__":
    test_emformer_attention_forward()
    test_emformer_attention_infer()
    test_emformer_layer_forward()
    test_emformer_layer_infer()
    test_emformer_encoder_forward()
    test_emformer_encoder_infer()
    test_emformer_forward()
    test_emformer_infer()
    test_emformer_encoder_layer_forward_infer_consistency()
    test_emformer_encoder_forward_infer_consistency()
    test_emformer_forward_infer_consistency()

import torch


def test_emformer_attention_forward():
    from emformer import EmformerAttention

    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    num_chunks = 3
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length
    attention = EmformerAttention(embed_dim=D, nhead=8)

    for use_memory in [True, False]:
        if use_memory:
            S = num_chunks
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
    U = 4
    R = 2
    L = 3
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


def test_convolution_module_forward():
    from emformer import ConvolutionModule

    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    num_chunks = 3
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length
    kernel_size = 31
    conv_module = ConvolutionModule(
        chunk_length,
        right_context_length,
        D,
        kernel_size,
    )

    utterance = torch.randn(U, B, D)
    right_context = torch.randn(R, B, D)
    cache = torch.randn(B, D, kernel_size - 1)

    utterance, right_context, new_cache = conv_module(
        utterance, right_context, cache
    )
    assert utterance.shape == (U, B, D)
    assert right_context.shape == (R, B, D)
    assert new_cache.shape == (B, D, kernel_size - 1)


def test_convolution_module_infer():
    from emformer import ConvolutionModule

    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    num_chunks = 1
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length
    kernel_size = 31
    conv_module = ConvolutionModule(
        chunk_length,
        right_context_length,
        D,
        kernel_size,
    )

    utterance = torch.randn(U, B, D)
    right_context = torch.randn(R, B, D)
    cache = torch.randn(B, D, kernel_size - 1)

    utterance, right_context, new_cache = conv_module.infer(
        utterance, right_context, cache
    )
    assert utterance.shape == (U, B, D)
    assert right_context.shape == (R, B, D)
    assert new_cache.shape == (B, D, kernel_size - 1)


def test_emformer_encoder_layer_forward():
    from emformer import EmformerEncoderLayer

    B, D = 2, 256
    chunk_length = 8
    right_context_length = 2
    left_context_length = 8
    kernel_size = 31
    num_chunks = 3
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length

    for use_memory in [True, False]:
        if use_memory:
            S = num_chunks
            M = S - 1
        else:
            S, M = 0, 0

        layer = EmformerEncoderLayer(
            d_model=D,
            nhead=8,
            dim_feedforward=1024,
            chunk_length=chunk_length,
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
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


def test_emformer_encoder_layer_infer():
    from emformer import EmformerEncoderLayer

    B, D = 2, 256
    chunk_length = 8
    right_context_length = 2
    left_context_length = 8
    kernel_size = 31
    num_chunks = 1
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length

    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0

        layer = EmformerEncoderLayer(
            d_model=D,
            nhead=8,
            dim_feedforward=1024,
            chunk_length=chunk_length,
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
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
            conv_cache,
        ) = layer.infer(
            utterance,
            lengths,
            right_context,
            memory,
            state,
            conv_cache,
        )
        assert output_utterance.shape == (U, B, D)
        assert output_right_context.shape == (R, B, D)
        if use_memory:
            assert output_memory.shape == (1, B, D)
        else:
            assert output_memory.shape == (0, B, D)
        assert len(output_state) == 4
        assert output_state[0].shape == (M, B, D)
        assert output_state[1].shape == (left_context_length, B, D)
        assert output_state[2].shape == (left_context_length, B, D)
        assert output_state[3].shape == (1, B)
        assert conv_cache.shape == (B, D, kernel_size - 1)


def test_emformer_encoder_forward():
    from emformer import EmformerEncoder

    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    left_context_length = 2
    num_chunks = 3
    U = num_chunks * chunk_length
    kernel_size = 31
    num_encoder_layers = 2

    for use_memory in [True, False]:
        if use_memory:
            S = num_chunks
            M = S - 1
        else:
            S, M = 0, 0

        encoder = EmformerEncoder(
            chunk_length=chunk_length,
            d_model=D,
            dim_feedforward=1024,
            num_encoder_layers=num_encoder_layers,
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
        )

        x = torch.randn(U + right_context_length, B, D)
        lengths = torch.randint(1, U + right_context_length + 1, (B,))
        lengths[0] = U + right_context_length

        output, output_lengths = encoder(x, lengths)
        assert output.shape == (U, B, D)
        assert torch.equal(
            output_lengths, torch.clamp(lengths - right_context_length, min=0)
        )


def test_emformer_encoder_infer():
    from emformer import EmformerEncoder

    B, D = 2, 256
    num_encoder_layers = 2
    chunk_length = 4
    right_context_length = 2
    left_context_length = 2
    num_chunks = 3
    kernel_size = 31

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
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
        )

        states = None
        conv_caches = None
        for chunk_idx in range(num_chunks):
            x = torch.randn(chunk_length + right_context_length, B, D)
            lengths = torch.randint(
                1, chunk_length + right_context_length + 1, (B,)
            )
            lengths[0] = chunk_length + right_context_length
            output, output_lengths, states, conv_caches = encoder.infer(
                x, lengths, states, conv_caches
            )
            assert output.shape == (chunk_length, B, D)
            assert torch.equal(
                output_lengths,
                torch.clamp(lengths - right_context_length, min=0),
            )
            assert len(states) == num_encoder_layers
            for state in states:
                assert len(state) == 4
                assert state[0].shape == (M, B, D)
                assert state[1].shape == (left_context_length, B, D)
                assert state[2].shape == (left_context_length, B, D)
                assert torch.equal(
                    state[3],
                    (chunk_idx + 1) * chunk_length * torch.ones_like(state[3]),
                )
            for conv_cache in conv_caches:
                assert conv_cache.shape == (B, D, kernel_size - 1)


def test_emformer_encoder_forward_infer_consistency():
    from emformer import EmformerEncoder

    chunk_length = 4
    num_chunks = 3
    U = chunk_length * num_chunks
    left_context_length, right_context_length = 1, 2
    D = 256
    num_encoder_layers = 3
    kernel_size = 31
    memory_sizes = [0, 3]

    for M in memory_sizes:
        encoder = EmformerEncoder(
            chunk_length=chunk_length,
            d_model=D,
            dim_feedforward=1024,
            num_encoder_layers=num_encoder_layers,
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
        )
        encoder.eval()

        x = torch.randn(U + right_context_length, 1, D)
        lengths = torch.tensor([U + right_context_length])

        # training mode with full utterance
        forward_output, forward_output_lengths = encoder(x, lengths)

        # streaming inference mode with individual chunks
        states = None
        conv_caches = None
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_length
            end_idx = start_idx + chunk_length
            chunk = x[start_idx : end_idx + right_context_length]  # noqa
            (
                infer_output_chunk,
                infer_output_lengths,
                states,
                conv_caches,
            ) = encoder.infer(
                chunk, torch.tensor([chunk_length]), states, conv_caches
            )
            forward_output_chunk = forward_output[start_idx:end_idx]
            assert torch.allclose(
                infer_output_chunk,
                forward_output_chunk,
                atol=1e-4,
                rtol=0.0,
            ), (
                infer_output_chunk - forward_output_chunk
            )


def test_emformer_forward():
    from emformer import Emformer

    num_features = 80
    chunk_length = 16
    right_context_length = 8
    left_context_length = 8
    num_chunks = 3
    U = num_chunks * chunk_length
    B, D = 2, 256
    kernel_size = 31

    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0
        model = Emformer(
            num_features=num_features,
            chunk_length=chunk_length,
            subsampling_factor=4,
            d_model=D,
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
        )
        x = torch.randn(B, U + right_context_length + 3, num_features)
        x_lens = torch.randint(1, U + right_context_length + 3 + 1, (B,))
        x_lens[0] = U + right_context_length + 3
        output, output_lengths = model(x, x_lens)
        assert output.shape == (B, U // 4, D)
        assert torch.equal(
            output_lengths,
            torch.clamp(
                ((x_lens - 1) // 2 - 1) // 2 - right_context_length // 4, min=0
            ),
        )


def test_emformer_infer():
    from emformer import Emformer

    num_features = 80
    chunk_length = 8
    U = chunk_length
    left_context_length, right_context_length = 128, 4
    B, D = 2, 256
    num_chunks = 3
    num_encoder_layers = 2
    kernel_size = 31

    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0
        model = Emformer(
            num_features=num_features,
            chunk_length=chunk_length,
            subsampling_factor=4,
            d_model=D,
            num_encoder_layers=num_encoder_layers,
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
        )
        states = None
        conv_caches = None
        for chunk_idx in range(num_chunks):
            x = torch.randn(B, U + right_context_length + 3, num_features)
            x_lens = torch.randint(1, U + right_context_length + 3 + 1, (B,))
            x_lens[0] = U + right_context_length + 3
            output, output_lengths, states, conv_caches = model.infer(
                x, x_lens, states, conv_caches
            )
            assert output.shape == (B, U // 4, D)
            assert torch.equal(
                output_lengths,
                torch.clamp(
                    ((x_lens - 1) // 2 - 1) // 2 - right_context_length // 4,
                    min=0,
                ),
            )
            assert len(states) == num_encoder_layers
            for state in states:
                assert len(state) == 4
                assert state[0].shape == (M, B, D)
                assert state[1].shape == (left_context_length // 4, B, D)
                assert state[2].shape == (left_context_length // 4, B, D)
                assert torch.equal(
                    state[3],
                    U // 4 * (chunk_idx + 1) * torch.ones_like(state[3]),
                )
            for conv_cache in conv_caches:
                assert conv_cache.shape == (B, D, kernel_size - 1)


def test_state_stack_unstack():
    from emformer import Emformer, stack_states, unstack_states

    num_features = 80
    chunk_length = 32
    encoder_dim = 512
    num_encoder_layers = 2
    kernel_size = 31
    left_context_length = 32
    right_context_length = 8
    memory_size = 32
    batch_size = 2

    model = Emformer(
        num_features=num_features,
        chunk_length=chunk_length,
        subsampling_factor=4,
        d_model=encoder_dim,
        num_encoder_layers=num_encoder_layers,
        cnn_module_kernel=kernel_size,
        left_context_length=left_context_length,
        right_context_length=right_context_length,
        memory_size=memory_size,
    )
    attn_caches = [
        [
            torch.zeros(memory_size, batch_size, encoder_dim),
            torch.zeros(left_context_length // 4, batch_size, encoder_dim),
            torch.zeros(
                left_context_length // 4,
                batch_size,
                encoder_dim,
            ),
        ]
        for _ in range(num_encoder_layers)
    ]
    conv_caches = [
        torch.zeros(batch_size, encoder_dim, kernel_size - 1)
        for _ in range(num_encoder_layers)
    ]
    states = [attn_caches, conv_caches]
    x = torch.randn(batch_size, 23, num_features)
    x_lens = torch.full((batch_size,), 23)
    num_processed_frames = torch.full((batch_size,), 0)
    y, y_lens, states = model.infer(
        x, x_lens, num_processed_frames=num_processed_frames, states=states
    )

    state_list = unstack_states(states)
    states2 = stack_states(state_list)

    for ss, ss2 in zip(states[0], states2[0]):
        for s, s2 in zip(ss, ss2):
            assert torch.allclose(s, s2), f"{s.sum()}, {s2.sum()}"

    for s, s2 in zip(states[1], states2[1]):
        assert torch.allclose(s, s2), f"{s.sum()}, {s2.sum()}"


def test_torchscript_consistency_infer():
    r"""Verify that scripting Emformer does not change the behavior of method `infer`."""
    from emformer import Emformer

    num_features = 80
    chunk_length = 32
    encoder_dim = 512
    num_encoder_layers = 2
    kernel_size = 31
    left_context_length = 32
    right_context_length = 8
    memory_size = 32
    batch_size = 2

    model = Emformer(
        num_features=num_features,
        chunk_length=chunk_length,
        subsampling_factor=4,
        d_model=encoder_dim,
        num_encoder_layers=num_encoder_layers,
        cnn_module_kernel=kernel_size,
        left_context_length=left_context_length,
        right_context_length=right_context_length,
        memory_size=memory_size,
    ).eval()
    attn_caches = [
        [
            torch.zeros(memory_size, batch_size, encoder_dim),
            torch.zeros(left_context_length // 4, batch_size, encoder_dim),
            torch.zeros(
                left_context_length // 4,
                batch_size,
                encoder_dim,
            ),
        ]
        for _ in range(num_encoder_layers)
    ]
    conv_caches = [
        torch.zeros(batch_size, encoder_dim, kernel_size - 1)
        for _ in range(num_encoder_layers)
    ]
    states = [attn_caches, conv_caches]
    x = torch.randn(batch_size, 23, num_features)
    x_lens = torch.full((batch_size,), 23)
    num_processed_frames = torch.full((batch_size,), 0)
    y, y_lens, out_states = model.infer(
        x, x_lens, num_processed_frames=num_processed_frames, states=states
    )

    sc_model = torch.jit.script(model).eval()
    sc_y, sc_y_lens, sc_out_states = sc_model.infer(
        x, x_lens, num_processed_frames=num_processed_frames, states=states
    )

    assert torch.allclose(y, sc_y)


if __name__ == "__main__":
    # test_emformer_attention_forward()
    # test_emformer_attention_infer()
    # test_convolution_module_forward()
    # test_convolution_module_infer()
    # test_emformer_encoder_layer_forward()
    # test_emformer_encoder_layer_infer()
    # test_emformer_encoder_forward()
    # test_emformer_encoder_infer()
    # test_emformer_encoder_forward_infer_consistency()
    # test_emformer_forward()
    # test_emformer_infer()
    # test_state_stack_unstack()
    test_torchscript_consistency_infer()

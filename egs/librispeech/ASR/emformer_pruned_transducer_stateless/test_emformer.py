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
            left_context_length=L,
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


def test_emformer_layer_infer():
    from emformer import EmformerLayer

    B, D = 2, 256
    R, L = 2, 5
    chunk_length = 2
    U = chunk_length

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
            left_context_length=L,
            max_memory_size=M,
        )

        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        memory = torch.randn(M, B, D)
        state = None
        (
            output_utterance,
            output_right_context,
            output_memory,
            output_state,
        ) = layer.infer(
            utterance,
            lengths,
            right_context,
            memory,
            state,
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
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
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
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
        )

        states = None
        for chunk_idx in range(num_chunks):
            x = torch.randn(U + R, B, D)
            lengths = torch.randint(1, U + R + 1, (B,))
            lengths[0] = U + R
            output, output_lengths, states = encoder.infer(x, lengths, states)
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
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            vgg_frontend=False,
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
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            vgg_frontend=False,
        )
        states = None
        for chunk_idx in range(num_chunks):
            x = torch.randn(B, U + R + 3, num_features)
            x_lens = torch.randint(1, U + R + 3 + 1, (B,))
            x_lens[0] = U + R + 3
            logits, output_lengths, states = model.infer(x, x_lens, states)
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


def test_emformer_attention_forward_infer_consistency():
    from emformer import EmformerEncoder

    chunk_length = 4
    num_chunks = 3
    U = chunk_length * num_chunks
    L, R = 1, 2
    D = 256
    num_encoder_layers = 1
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
            forward_output_right_context_utterance,
            forward_output_memory,
        ) = encoder_layer._apply_attention_forward(
            utterance,
            lengths,
            right_context,
            memory,
            attention_mask,
        )
        forward_output_utterance = forward_output_right_context_utterance[
            right_context.size(0) :  # noqa
        ]

        state = None
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
                infer_output_right_context_utterance,
                infer_output_memory,
                state,
            ) = encoder_layer._apply_attention_infer(
                chunk,
                chunk_length,
                chunk_right_context,
                chunk_memory,
                state,
            )
            infer_output_chunk = infer_output_right_context_utterance[
                chunk_right_context.size(0) :  # noqa
            ]
            forward_output_chunk = forward_output_utterance[start_idx:end_idx]
            assert torch.allclose(
                infer_output_chunk,
                forward_output_chunk,
                atol=1e-6,
                rtol=0.0,
            )


def test_emformer_layer_forward_infer_consistency():
    from emformer import EmformerEncoder

    chunk_length = 4
    num_chunks = 3
    U = chunk_length * num_chunks
    L, R = 1, 2
    D = 256
    num_encoder_layers = 1
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
            ) = encoder_layer.infer(
                chunk,
                chunk_length,
                chunk_right_context,
                chunk_memory,
                state,
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
        )
        encoder.eval()

        x = torch.randn(U + R, 1, D)
        lengths = torch.tensor([U + R])

        forward_output, forward_output_lengths = encoder(x, lengths)

        states = None
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_length
            end_idx = start_idx + chunk_length
            chunk = x[start_idx : end_idx + R]  # noqa
            chunk_right_context = x[end_idx : end_idx + R]  # noqa
            chunk_length = torch.tensor([chunk_length])
            infer_output_chunk, infer_output_lengths, states = encoder.infer(
                chunk,
                chunk_length,
                states,
            )
            forward_output_chunk = forward_output[start_idx:end_idx]
            assert torch.allclose(
                infer_output_chunk,
                forward_output_chunk,
                atol=1e-5,
                rtol=0.0,
            )


def test_emformer_infer_batch_single_consistency():
    """Test consistency of cached states and output logits between single
    utterance inference and batch inference."""
    from emformer import Emformer

    num_features = 80
    output_dim = 1000
    chunk_length = 8
    num_chunks = 3
    U = num_chunks * chunk_length
    L, R = 128, 4
    B, D = 2, 256
    num_encoder_layers = 2
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
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            vgg_frontend=False,
        )
        model.eval()

        def save_states(states):
            saved_states = []
            for layer_idx in range(len(states)):
                layer_state = []
                layer_state.append(states[layer_idx][0].clone())  # memory
                layer_state.append(
                    states[layer_idx][1].clone()
                )  # left_context_key
                layer_state.append(
                    states[layer_idx][2].clone()
                )  # left_context_val
                layer_state.append(states[layer_idx][3].clone())  # past_length
                saved_states.append(layer_state)
            return saved_states

        def assert_states_equal(saved_states, states, sample_idx):
            for layer_idx in range(len(saved_states)):
                # assert eqaul memory
                assert torch.allclose(
                    states[layer_idx][0],
                    saved_states[layer_idx][0][
                        :, sample_idx : sample_idx + 1  # noqa
                    ],
                    atol=1e-5,
                    rtol=0.0,
                )
                # assert equal left_context_key
                assert torch.allclose(
                    states[layer_idx][1],
                    saved_states[layer_idx][1][
                        :, sample_idx : sample_idx + 1  # noqa
                    ],
                    atol=1e-5,
                    rtol=0.0,
                )
                # assert equal left_context_val
                assert torch.allclose(
                    states[layer_idx][2],
                    saved_states[layer_idx][2][
                        :, sample_idx : sample_idx + 1  # noqa
                    ],
                    atol=1e-5,
                    rtol=0.0,
                )
                # assert eqaul past_length
                assert torch.equal(
                    states[layer_idx][3],
                    saved_states[layer_idx][3][
                        :, sample_idx : sample_idx + 1  # noqa
                    ],
                )

        x = torch.randn(B, U + R + 3, num_features)
        batch_logits = []
        batch_states = []
        states = None
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_length
            end_idx = start_idx + chunk_length
            chunk = x[:, start_idx : end_idx + R + 3]  # noqa
            lengths = torch.tensor([chunk_length + R + 3]).expand(B)
            logits, output_lengths, states = model.infer(chunk, lengths, states)
            batch_logits.append(logits)
            batch_states.append(save_states(states))
        batch_logits = torch.cat(batch_logits, dim=1)

        single_logits = []
        for sample_idx in range(B):
            sample = x[sample_idx : sample_idx + 1]  # noqa
            chunk_logits = []
            states = None
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_length
                end_idx = start_idx + chunk_length
                chunk = sample[:, start_idx : end_idx + R + 3]  # noqa
                lengths = torch.tensor([chunk_length + R + 3])
                logits, output_lengths, states = model.infer(
                    chunk, lengths, states
                )
                chunk_logits.append(logits)

                assert_states_equal(batch_states[chunk_idx], states, sample_idx)

            chunk_logits = torch.cat(chunk_logits, dim=1)
            single_logits.append(chunk_logits)
        single_logits = torch.cat(single_logits, dim=0)

        assert torch.allclose(batch_logits, single_logits, atol=1e-5, rtol=0.0)


if __name__ == "__main__":
    test_emformer_attention_forward()
    test_emformer_attention_infer()
    test_emformer_layer_forward()
    test_emformer_layer_infer()
    test_emformer_encoder_forward()
    test_emformer_encoder_infer()
    test_emformer_forward()
    test_emformer_infer()
    test_emformer_attention_forward_infer_consistency()
    test_emformer_layer_forward_infer_consistency()
    test_emformer_encoder_forward_infer_consistency()
    test_emformer_infer_batch_single_consistency()

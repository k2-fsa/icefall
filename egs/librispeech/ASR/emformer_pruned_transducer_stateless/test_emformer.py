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

        output_right_context_utterance, output_memory, next_key, next_val = \
            attention.infer(
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
        output_utterance, output_right_context, output_memory, output_state = \
            layer.infer(
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
        assert torch.equal(
            output_lengths, torch.clamp(lengths - R, min=0)
        )


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
            output, output_lengths, states = \
                encoder.infer(x, lengths, states)
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
    chunk_length = 16
    L, R = 32, 16
    B, D, U = 2, 256, 48
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
        x = torch.randn(B, U + R, num_features)
        x_lens = torch.randint(1, U + R + 1, (B,))
        x_lens[0] = U + R
        logits, output_lengths = model(x, x_lens)
        assert logits.shape == (B, U // 4, output_dim)
        assert torch.equal(
            output_lengths, torch.clamp(x_lens // 4 - R // 4, min=0)
        )


def test_emformer_infer():
    from emformer import Emformer
    num_features = 80
    output_dim = 1000
    chunk_length = 16
    U = chunk_length
    L, R = 32, 16
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
            x = torch.randn(B, U + R, num_features)
            x_lens = torch.randint(1, U + R + 1, (B,))
            x_lens[0] = U + R
            logits, output_lengths, states = \
                model.infer(x, x_lens, states)
            assert logits.shape == (B, U // 4, output_dim)
            assert torch.equal(
                output_lengths, torch.clamp(x_lens // 4 - R // 4, min=0)
            )
            assert len(states) == num_encoder_layers
            for state in states:
                assert len(state) == 4
                assert state[0].shape == (M, B, D)
                assert state[1].shape == (L // 4, B, D)
                assert state[2].shape == (L // 4, B, D)
                assert torch.equal(
                    state[3],
                    (chunk_idx + 1) * U // 4 * torch.ones_like(state[3])
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

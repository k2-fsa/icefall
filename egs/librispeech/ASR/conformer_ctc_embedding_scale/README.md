## Differences between `conformer_ctc` and `conformer_ctc_embedding_scale`

`conformer_ctc_embedding_scale` replaces `nn.Embedding` with modified
`Embedding`. Modified embedding contains two changes:

  - (1) The weight matrix is initialized to the range `(-std, std)` where
    `std = 1 / sqrt(embedding_dim)`

  - (2) The output of the embedding is scaled by `sqrt(embedding_dim)`

Also, `conformer_ctc_embedding_scale` modifies the `PositionalEncoding`
in `transformer.py`. It replaces

```python
self.xscale = math.sqrt(self.d_model)
x = x * self.xscale + self.pe[:, : x.size(1), :]
```
with

```python
self.pos_scale = 1. / math.sqrt(self.d_model)
x = x + self.pe[:, : x.size(1), :] * self.pos_scale
```

You can use

```bash
diff conformer_ctc/transformer.py conformer_ctc_embedding_scale/transformer.py
```

to find the exact differences.

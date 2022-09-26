import pytest
from transformer_decoder import TransformerDecoder
import torch

def test_left_context():
    with torch.no_grad():
        decoder = TransformerDecoder(10, 256, 0 , 0 ,15)# left_context=100)
        decoder.eval()
        out = decoder(torch.tensor([[1, 2, 5, 6, 8, 6, 7, 4]]))
        print(out)
        print(out.shape)
        out_context = decoder(torch.tensor([[1, 2, 5, 6, 8, 6, 7, 4]]))
        print(out[0,-1,:])
        print(out_context[0,-1,:])
        print(out[0,0,:])
        print(out_context[0,0,:])
        out_p = decoder(torch.tensor([[1, 2, 5, 6, 8, 6, 7, 4]]))

        print(out-out_p)
    assert False
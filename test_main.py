import unittest
import torch
from main import inputEmbedding, PositionalEncoding, Transformer

class MockEncoder(torch.nn.Module):
    def forward(self, x, mask):
        return x

class MockDecoder(torch.nn.Module):
    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        return tgt

class MockProjectionLayer(torch.nn.Module):
    def forward(self, x):
        # Project to vocab size
        batch, seq_len, d_model = x.shape
        vocab_size = 10
        return torch.zeros(batch, seq_len, vocab_size)

class TestTransformerModules(unittest.TestCase):
    def test_input_embedding(self):
        vocab_size = 10
        d_model = 16
        batch_size = 2
        seq_len = 5
        model = inputEmbedding(d_model, vocab_size)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        out = model(x)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_positional_encoding(self):
        d_model = 16
        seq_len = 5
        dropout = 0.1
        batch_size = 2
        model = PositionalEncoding(d_model, seq_len, dropout)
        x = torch.zeros(batch_size, seq_len, d_model)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_transformer(self):
        vocab_size = 10
        d_model = 16
        seq_len = 5
        batch_size = 2
        dropout = 0.1
        src = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
        src_mask = torch.ones(batch_size, seq_len)
        tgt_mask = torch.ones(batch_size, seq_len)
        encoder = MockEncoder()
        decoder = MockDecoder()
        src_embed = inputEmbedding(d_model, vocab_size)
        tgt_embed = inputEmbedding(d_model, vocab_size)
        src_pos = PositionalEncoding(d_model, seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, seq_len, dropout)
        projection_layer = MockProjectionLayer()
        transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
        enc_out = transformer.encode(src, src_mask)
        dec_out = transformer.decode(enc_out, src_mask, tgt, tgt_mask)
        proj_out = transformer.project(dec_out)
        self.assertEqual(proj_out.shape, (batch_size, seq_len, vocab_size))

if __name__ == '__main__':
    unittest.main()

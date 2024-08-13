import math
import copy
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        x  = self.lut(x)
        x *= math.sqrt(self.d_model)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute the positional encodings once in log space.
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        x = self.dropout(x)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout_rate=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def _attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 调整为度 d_model -> [batch, head, seq_length, d_k]
        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self._attention(
            query, key, value, mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        x = self.linears[-1](x)
        del query
        del key
        del value
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.w_1(x)
        x = x.relu()
        x = self.w_2(x)
        return x

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout_rate):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(self.norm(x)))
        return x

class EncoderSublayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate):
        super(EncoderSublayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(size, dropout_rate), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, size, self_attn, feed_forward, dropout_rate, N):
        super(Encoder, self).__init__()
        self.layers = clones(EncoderSublayer(size, self_attn, feed_forward, dropout_rate), N)
        self.norm = LayerNorm(size, eps=1e-6)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x
    
class DecoderSublayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_rate):
        super(DecoderSublayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(size, dropout_rate), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x,      x,      tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn( x, memory, memory, src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_rate, N):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderSublayer(size, self_attn, src_attn, feed_forward, dropout_rate), N)
        self.norm = LayerNorm(size, eps=1e-6)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return x

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, len_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, len_vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, len_src_vocab, len_tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout_rate=0.1):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        pe = PositionalEncoding(d_model, dropout_rate)
        attn = MultiHeadedAttention(h, d_model, dropout_rate)
        ffn  = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

        self.encoder = Encoder(d_model, c(attn), c(ffn), dropout_rate, N)
        self.decoder = Decoder(d_model, c(attn), c(attn), c(ffn), dropout_rate, N)
        self.src_embed = nn.Sequential(Embeddings(d_model, len_src_vocab), c(pe))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, len_tgt_vocab), c(pe))

        self.generator = Generator(d_model, len_tgt_vocab)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        out = self.encoder(src, src_mask)
        return out

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        out = self.decoder(tgt, memory, src_mask, tgt_mask)
        return out

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        out = self.encode(src, src_mask)
        out = self.decode(out, src_mask, tgt, tgt_mask)
        return out

if __name__ == "__main__":
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0

    test_model = Transformer(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    print(get_parameter_number(test_model))

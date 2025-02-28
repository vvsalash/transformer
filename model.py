import torch
from torch import nn
from math import sqrt, log
import torch.nn.functional as f
from torch.autograd import Variable
from copy import deepcopy

infinity: float = -1e9


class Embedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.emb(x) * sqrt(self.embed_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(f.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)
        return self.gamma * x_normalized + self.beta


class ResidualConnection(nn.Module):
    def __init__(self, normalized_shape: int, dropout: float = 0.1) -> None:
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_length: int = 5000) -> None:
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pos_features = torch.zeros(max_length, embed_dim)

        positions = torch.arange(0, max_length).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, embed_dim, 2) * (-log(10000.0) / embed_dim))
        arguments = positions * frequencies

        pos_features[:, 0::2] = torch.sin(arguments)
        pos_features[:, 1::2] = torch.cos(arguments)
        pos_features = pos_features.unsqueeze(0)
        self.register_buffer('pos_features', pos_features)

    def forward(self, x):
        x = x + Variable(self.pos_features[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def create_layer_stack(layer: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(layer) for _ in range(count)])


def self_attention(query_embeddings,
                   key_embeddings,
                   value_embeddings,
                   mask: torch.BoolTensor | None = None,
                   dropout=None):
    embed_dim = query_embeddings.size(-1)
    attention_scores = torch.matmul(
        query_embeddings, key_embeddings.transpose(-2, -1)
    ) / sqrt(embed_dim)

    if mask is not None:
        attention_scores = attention_scores.masked_fill_(mask == 0, infinity)
    attention_probs = f.softmax(attention_scores, dim=-1)

    if dropout is not None:
        attention_probs = dropout(attention_probs)

    return torch.matmul(attention_probs, value_embeddings)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_of_heads: int, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_of_heads == 0
        self.embed_dim = embed_dim // num_of_heads
        self.n_heads = num_of_heads
        self.linear_layers = create_layer_stack(nn.Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_embeddings, key_embeddings, value_embeddings, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query_embeddings.size(0)

        query, key, value = [
            layer(x).view(batch_size, -1, self.n_heads, self.embed_dim).transpose(1, 2)
            for layer, x in zip(
                self.linear_layers, (query_embeddings, key_embeddings, value_embeddings)
            )
        ]

        x = self_attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.embed_dim
        )
        return self.linear_layers[-1](x)


class EncoderLayer(nn.Module):
    def __init__(self,
                 normalized_shape: int,
                 attention,
                 feed_forward,
                 dropout: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.normalized_shape = normalized_shape
        self.self_attention = attention
        self.feed_forward = feed_forward
        self.residual_connections = create_layer_stack(
            ResidualConnection(normalized_shape, dropout), 2
        )

    def forward(self, x, mask):
        x = self.residual_connections[0](
            x, lambda embedding: self.self_attention(embedding, embedding, embedding, mask)
        )
        return self.residual_connections[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self,
                 normalized_shape: int,
                 attention,
                 source_attention,
                 feed_forward,
                 dropout: float = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.normalized_shape = normalized_shape
        self.self_attention = attention
        self.source_attention = source_attention
        self.feed_forward = feed_forward
        self.residual_connections = create_layer_stack(
            ResidualConnection(normalized_shape, dropout), 3
        )

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connections[0](
            x, lambda embedding: self.self_attention(embedding, embedding, embedding, target_mask)
        )
        x = self.residual_connections[1](x, lambda embedding: self.source_attention(
            embedding, encoder_output, encoder_output, source_mask
        ))
        return self.residual_connections[2](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, count: int) -> None:
        super(Encoder, self).__init__()
        self.layers = create_layer_stack(layer, count)
        self.norm = LayerNorm(layer.normalized_shape)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, count: int) -> None:
        super(Decoder, self).__init__()
        self.layers = create_layer_stack(layer, count)
        self.norm = LayerNorm(layer.normalized_shape)

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)


class OutputLayer(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.linear(x)


class Transformer(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 source_embeddings,
                 target_embeddings,
                 output_layer) -> None:
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        self.output_layer = output_layer

    def encode(self, source, source_mask):
        return self.encoder(self.source_embeddings(source), source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        return self.decoder(self.target_embeddings(target),
                            encoder_output,
                            source_mask,
                            target_mask)

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(
            self.encode(source, source_mask), source_mask, target, target_mask
        )


def build_model(source_vocab_size: int,
                target_vocab_size: int,
                count: int = 6,
                embed_dim: int = 512,
                ff_dim: int = 1024,
                n_heads: int = 8,
                dropout: float = 0.1):
    attention_layer = MultiHeadAttention(embed_dim, n_heads)
    feedforward_layer = FeedForward(embed_dim, ff_dim, dropout)
    positional_encoder = PositionalEncoder(embed_dim, dropout)
    model = Transformer(
        Encoder(EncoderLayer(embed_dim,
                             deepcopy(attention_layer),
                             deepcopy(feedforward_layer),
                             dropout), count),
        Decoder(DecoderLayer(embed_dim,
                             deepcopy(attention_layer),
                             deepcopy(attention_layer),
                             deepcopy(feedforward_layer),
                             dropout), count),
        nn.Sequential(Embedding(embed_dim, source_vocab_size), deepcopy(positional_encoder)),
        nn.Sequential(Embedding(embed_dim, target_vocab_size), deepcopy(positional_encoder)),
        OutputLayer(embed_dim, target_vocab_size)
    )

    for parameters in model.parameters():
        if parameters.dim() > 1:
            nn.init.xavier_uniform_(parameters)

    return model

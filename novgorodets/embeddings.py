import torch
import torch.nn as nn


class DualEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.word_char_emb_dim

        self.char_embeddings = nn.Embedding(
            config.vocab_char_size, d, padding_idx=config.pad_token_id
        )
        self.word_embeddings = nn.Embedding(
            config.vocab_word_size, d, padding_idx=0
        )
        self.projection = nn.Linear(2 * d, config.hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids, word_ids):
        bsz, seq_len = input_ids.shape
        pos_ids = self.position_ids[:, :seq_len]

        c = self.char_embeddings(input_ids)
        w = self.word_embeddings(word_ids)

        x = torch.cat([c, w], dim=-1)
        x = self.projection(x)
        x = x + self.position_embeddings(pos_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
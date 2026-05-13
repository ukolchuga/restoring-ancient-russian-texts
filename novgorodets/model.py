import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertEncoder

from config import DualBertConfig
from embeddings import DualEmbeddings


class DualBertForMaskedLM(BertPreTrainedModel):
    config_class = DualBertConfig

    def __init__(self, config: DualBertConfig):
        super().__init__(config)
        self.dual_embeddings = DualEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.mlm_dense = nn.Linear(config.hidden_size, config.word_char_emb_dim)
        self.mlm_act = nn.GELU()
        self.mlm_norm = nn.LayerNorm(config.word_char_emb_dim, eps=config.layer_norm_eps)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_char_size))

        self.post_init()

    def get_input_embeddings(self):
        return self.dual_embeddings.char_embeddings

    def set_input_embeddings(self, value):
        self.dual_embeddings.char_embeddings = value

    def forward(
        self,
        input_ids=None,
        word_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None or word_ids is None:
            raise ValueError("Both input_ids and word_ids are required.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        emb = self.dual_embeddings(input_ids=input_ids, word_ids=word_ids)
        ext_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape, input_ids.device)

        enc_out = self.encoder(
            emb,
            attention_mask=ext_mask,
            head_mask=[None] * self.config.num_hidden_layers,
            return_dict=True,
        )
        seq = enc_out.last_hidden_state

        x = self.mlm_dense(seq)
        x = self.mlm_act(x)
        x = self.mlm_norm(x)

        char_emb = self.dual_embeddings.char_embeddings.weight
        logits = x @ char_emb.T + self.mlm_bias

        logits = x @ char_emb.T + self.mlm_bias

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            emb_norm = self.dual_embeddings.char_embeddings.weight.norm()
            x_norm = x.norm()
            raise RuntimeError(
                f"NaN/Inf in logits! char_emb_norm={emb_norm:.2f}, x_norm={x_norm:.2f}"
            )

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.config.vocab_char_size), labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return MaskedLMOutput(loss=loss, logits=logits)
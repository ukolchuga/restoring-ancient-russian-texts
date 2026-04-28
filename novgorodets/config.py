from transformers import BertConfig


class DualBertConfig(BertConfig):
    model_type = "dual_bert"

    def __init__(
        self,
        vocab_char_size: int = 256,
        vocab_word_size: int = 50000,
        word_char_emb_dim: int = 192,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_char_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            **kwargs,
        )
        self.vocab_char_size = vocab_char_size
        self.vocab_word_size = vocab_word_size
        self.word_char_emb_dim = word_char_emb_dim
from transformers import RoFormerConfig


def get_roformer_config(vocab_size: int, pad_token_id: int):
    """
    Returns the RoFormer configuration with parameters optimized for Ancient Russian text.
    """
    return RoFormerConfig(
        vocab_size=vocab_size,
        embedding_size=512,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=514,
        pad_token_id=pad_token_id,
        rotary_value=False,
    )

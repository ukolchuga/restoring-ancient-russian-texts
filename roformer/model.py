from config import get_roformer_config
from transformers import RoFormerForMaskedLM


def get_model(vocab_size: int, pad_token_id: int):
    """
    Initializes and returns a RoFormer model for Masked Language Modeling.
    """
    config = get_roformer_config(vocab_size, pad_token_id)
    model = RoFormerForMaskedLM(config)
    return model

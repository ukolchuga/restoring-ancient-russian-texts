#!/usr/bin/env python3
"""
Script for training a custom Byte-Level BPE tokenizer on Ancient Russian text.
Produces a Hugging Face compatible tokenizer with context-specific tags and historical punctuation.
"""

import os

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# Configuration
CORPUS_FILE = "splits/train.txt"
VOCAB_SIZE = 50000
SAVE_DIR = "ancient_rus_tokenizer_BPE"


def main():
    """Trains the BPE tokenizer and saves it in Hugging Face format."""
    if not os.path.exists(CORPUS_FILE):
        print(f"Error: Training corpus {CORPUS_FILE} not found!")
        return

    print("Initializing Byte-Level BPE Tokenizer...")
    bpe_tokenizer = ByteLevelBPETokenizer()

    # Define standard and domain-specific special tokens
    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "[CTX_CHURCH]",
        "[CTX_DAILY]",
        "[CTX_LEGAL]",
        "[CTX_LIT]",
        "[CTX_EPIC]",
        "[CTX_SCIENCE]",
        "[GAP]",
        "·",
        ":",
    ]

    print(f"Training tokenizer on {CORPUS_FILE} with vocab size {VOCAB_SIZE}...")
    bpe_tokenizer.train(
        files=[CORPUS_FILE],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
    )

    # Save the base tokenizer state
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer_path = os.path.join(SAVE_DIR, "tokenizer.json")
    bpe_tokenizer.save(tokenizer_path)

    print("Configuring PreTrainedTokenizerFast wrapper...")
    # Wrap the BPE model into a Transformers-compatible fast tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        max_len=512,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    # Explicitly register additional special tokens to ensure correct handling
    fast_tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "[CTX_CHURCH]",
                "[CTX_DAILY]",
                "[CTX_LEGAL]",
                "[CTX_LIT]",
                "[CTX_EPIC]",
                "[CTX_SCIENCE]",
                "[GAP]",
                "·",
                ":",
            ]
        }
    )

    # Save final tokenizer artifacts for model training
    fast_tokenizer.save_pretrained(SAVE_DIR)
    print(f"BPE Tokenizer successfully trained and saved to: {SAVE_DIR}/")

    # Run quick validation tests
    print("\n--- Tokenization Validation ---")
    test_texts = [
        "[CTX_DAILY] поклонъ · ѿ · бориса · [GAP] · настасии",
        "[CTX_LEGAL] · ꙅ҃ · десѧ · коуно ·",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\nTest Case {i}: {text}")
        tokens = fast_tokenizer.tokenize(text)
        print(f"  Tokens:   {tokens}")
        ids = fast_tokenizer.encode(text)
        print(f"  Decoding: {fast_tokenizer.decode(ids)}")


if __name__ == "__main__":
    main()

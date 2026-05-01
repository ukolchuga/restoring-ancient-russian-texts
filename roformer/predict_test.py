#!/usr/bin/env python3
"""
Utility script for interactive model validation.
Runs a suite of historical text restoration test cases through the trained RoFormer model.
"""

import argparse

import torch
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(description="RoFormer Inference Test.")
    parser.add_argument(
        "--model_path",
        default="artifacts/roformer_training_output/final_model",
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0 if torch.cuda.is_available() else -1,
        help="Device ID (0 for GPU, -1 for CPU).",
    )
    args = parser.parse_args()

    print(f"Loading RoFormer pipeline from {args.model_path}...")
    try:
        roformer_pipe = pipeline(
            "fill-mask",
            model=args.model_path,
            tokenizer=args.model_path,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    # Define various historical contexts and masking scenarios
    test_cases = [
        {
            "desc": "📚 Chronicles (Context: Lit)",
            "text": "[CTX_LIT] И пошелъ князь игорь на <mask> землю со своею дружиною.",
            "expected": "рускую / свою",
        },
        {
            "desc": "⚖️ Russkaya Pravda (Context: Legal)",
            "text": "[CTX_LEGAL] Аже кто оубиеть <mask> , то платити виру 40 гривенъ.",
            "expected": "мужь",
        },
        {
            "desc": "🏡 Birch Bark Letters (Context: Daily)",
            "text": "[CTX_DAILY] поклоне ѿ ꙩндреꙗ · к ѥва · и к микифору про <mask> ѡкупи ꙩсподине",
            "expected": "серебро",
        },
        {
            "desc": "⛪️ Church Texts (Context: Church)",
            "text": "[CTX_CHURCH] И рече господь къ <mask> своимъ, глаголя...",
            "expected": "ученикомъ / людемъ",
        },
    ]

    print("\n" + "=" * 60)
    print(" RoFormer Model Validation - Ancient Russian")
    print("=" * 60)

    for idx, case in enumerate(test_cases, 1):
        print(f"\n[{idx}/{len(test_cases)}] {case['desc']}")
        print(f"Text: {case['text']}")
        print(f"Expected: {case['expected']}")

        try:
            results = roformer_pipe(case["text"], top_k=5)

            # Standardize output for single mask cases
            if not isinstance(results[0], dict):
                results = results[0]

            for i, res in enumerate(results):
                # Clean BPE artifacts for readability
                clean_word = res["token_str"].replace("Ġ", "").strip()
                score = res["score"] * 100
                print(f"  {i + 1}: '{clean_word}' ({score:.1f}%)")
        except Exception as e:
            print(f"  Inference failed: {e}")


if __name__ == "__main__":
    main()

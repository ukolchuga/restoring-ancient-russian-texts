import random
from typing import Any, Dict, List

import torch


class RoFormerPhysicalDegradationCollator:
    """
    Data collator that simulates physical damage to historical documents.
    Strategies:
    1. Edge Masking: Simulates broken or frayed document edges.
    2. Span Masking: Simulates holes or worn spots (Masked Language Modeling).
    3. Gap Augmentation: Randomly inserts [GAP] tokens to teach the model to handle unreadable sections.
    """

    def __init__(
        self,
        tokenizer: Any,
        mlm_prob: float = 0.15,
        max_span: int = 3,
        edge_prob: float = 0.1,
        add_random_gaps: bool = True,
        gap_prob: float = 0.05,
        gap_span_min: int = 1,
        gap_span_max: int = 5,
        max_gaps: int = 2,
    ):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.max_span = max_span
        self.edge_prob = edge_prob
        self.add_random_gaps = add_random_gaps
        self.gap_prob = gap_prob
        self.gap_span_min = gap_span_min
        self.gap_span_max = gap_span_max
        self.max_gaps = max_gaps

        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # Resolve GAP token ID if present in the vocabulary
        self.gap_token_id = None
        if "[GAP]" in tokenizer.get_vocab():
            self.gap_token_id = tokenizer.convert_tokens_to_ids("[GAP]")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )

        # Use pre-defined labels if available (typical for fixed evaluation sets like test_b)
        if "labels" in features[0]:
            labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        labels = input_ids.clone()
        batch_size, seq_len = input_ids.shape

        # Identify special tokens to protect them from being masked
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # Apply random [GAP] augmentation to simulate unreadable text segments
        if self.add_random_gaps and (self.gap_token_id is not None):
            for i in range(batch_size):
                if random.random() >= self.gap_prob:
                    continue
                n_gaps = random.randint(1, self.max_gaps)
                valid_indices = (
                    (~special_tokens_mask[i] & (attention_mask[i] == 1))
                    .nonzero(as_tuple=True)[0]
                    .tolist()
                )
                if not valid_indices:
                    continue

                for _ in range(n_gaps):
                    if not valid_indices:
                        break
                    start = random.choice(valid_indices)
                    span_len = random.randint(self.gap_span_min, self.gap_span_max)
                    end = min(start + span_len, seq_len)

                    sel = [p for p in range(start, end) if p in valid_indices]
                    if not sel:
                        continue

                    input_ids[i, sel] = self.gap_token_id
                    labels[i, sel] = -100
                    for p in sel:
                        special_tokens_mask[i, p] = True
                    valid_indices = [p for p in valid_indices if p not in sel]

        # Generate base MLM mask
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        final_mask = masked_indices.clone()

        for i in range(batch_size):
            # 1. Edge Masking: Select a start or end boundary and mask a small span
            if random.random() < self.edge_prob:
                valid = (~special_tokens_mask[i] & (attention_mask[i] == 1)).nonzero(
                    as_tuple=True
                )[0]
                if len(valid) > 5:
                    edge_len = random.randint(2, 5)
                    is_start = random.choice([True, False])
                    if is_start:
                        st = valid[0].item()
                        final_mask[i, st : st + edge_len] = True
                    else:
                        en = valid[-1].item()
                        final_mask[i, en - edge_len + 1 : en + 1] = True

            # 2. Span Masking: Expand base masked indices into small spans
            for j in range(seq_len):
                if masked_indices[i, j]:
                    span_len = random.randint(1, self.max_span)
                    end_idx = min(j + span_len, seq_len)
                    if not special_tokens_mask[i, j:end_idx].any():
                        final_mask[i, j:end_idx] = True

        # Finalize labels: set non-masked positions to -100 for CrossEntropy loss
        final_mask &= attention_mask == 1
        labels[~final_mask] = -100

        # Apply standard MLM substitution: 80% [MASK], 10% random token, 10% original
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & final_mask
        )
        input_ids[indices_replaced] = self.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & final_mask
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

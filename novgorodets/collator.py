import random
import torch


class DualPhysicalDegradationCollator:
    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        unk_word_id: int,
        vocab_char_size: int,
        special_token_ids: list[int],
        mlm_prob: float = 0.15,
        max_span: int = 3,
        edge_prob: float = 0.1,
        add_random_gaps: bool = False,
        gap_token_id: int | None = None,
        gap_prob: float = 0.02,
        gap_span_min: int = 1,
        gap_span_max: int = 5,
        max_gaps: int = 2,
    ):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.unk_word_id = unk_word_id
        self.vocab_char_size = vocab_char_size
        self.special_token_ids = set(special_token_ids)
        self.mlm_prob = mlm_prob
        self.max_span = max_span
        self.edge_prob = edge_prob
        self.add_random_gaps = add_random_gaps
        self.gap_token_id = gap_token_id
        self.gap_prob = gap_prob
        self.gap_span_min = gap_span_min
        self.gap_span_max = gap_span_max
        self.max_gaps = max_gaps

    def __call__(self, features):
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        word_ids = torch.tensor([f["word_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

        if "labels" in features[0]:
            labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "word_ids": word_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        labels = input_ids.clone()

        if "special_tokens_mask" in features[0]:
            special_tokens_mask = torch.tensor(
                [f["special_tokens_mask"] for f in features], dtype=torch.bool
            )
        else:
            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in self.special_token_ids:
                special_tokens_mask |= (input_ids == tid)

        # ---- RANDOM GAP AUGMENTATION ----
        # Вставляем [GAP] спаны в input_ids и помечаем их как спецтокены / без ground truth
        if self.add_random_gaps and (self.gap_token_id is not None):
            bsz, seq_len = input_ids.shape
            for i in range(bsz):
                if random.random() >= self.gap_prob:
                    continue
                n_gaps = random.randint(1, self.max_gaps)
                # доступные позиции (только ненулевой attention и не спецтокены)
                valid = (~special_tokens_mask[i] & (attention_mask[i] == 1)).nonzero(as_tuple=True)[0].tolist()
                if len(valid) == 0:
                    continue
                for _ in range(n_gaps):
                    if not valid:
                        break
                    start = random.choice(valid)
                    span_len = random.randint(self.gap_span_min, self.gap_span_max)
                    end = min(start + span_len, seq_len)
                    # выбираем реальные позиции в диапазоне, исключая спец/пады
                    sel = [p for p in range(start, end) if (p in valid)]
                    if not sel:
                        # попробуем с другим стартом
                        continue
                    # применяем GAP: заменяем input_ids, помечаем labels как -100 и специальный токен
                    input_ids[i, sel] = self.gap_token_id
                    # labels уже создан как клон input_ids ранее, поэтому пометим -100:
                    labels[i, sel] = -100
                    # пометим как спецтокены, чтобы MLM не выбирал их
                    for p in sel:
                        special_tokens_mask[i, p] = True
                    # исключаем использованные позиции из valid
                    valid = [p for p in valid if p not in sel]
        # ---- end RANDOM GAP AUGMENTATION ----

        prob = torch.full(labels.shape, self.mlm_prob, dtype=torch.float)
        prob.masked_fill_(special_tokens_mask, 0.0)
        base_mask = torch.bernoulli(prob).bool()
        final_mask = base_mask.clone()

        bsz, seq_len = input_ids.shape
        for i in range(bsz):
            if random.random() < self.edge_prob:
                valid = (~special_tokens_mask[i] & (attention_mask[i] == 1)).nonzero(as_tuple=True)[0]
                if len(valid) > 3:
                    edge_len = random.randint(2, 5)
                    if random.random() < 0.5:
                        st = valid[0].item()
                        final_mask[i, st : min(st + edge_len, seq_len)] = True
                    else:
                        en = valid[-1].item()
                        st = max(0, en - edge_len + 1)
                        final_mask[i, st : en + 1] = True

            for j in range(seq_len):
                if base_mask[i, j]:
                    span_len = random.randint(1, self.max_span)
                    end = min(j + span_len, seq_len)
                    if not special_tokens_mask[i, j:end].any():
                        final_mask[i, j:end] = True

        final_mask &= (attention_mask == 1)
        labels[~final_mask] = -100

        rep = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & final_mask
        input_ids[rep] = self.mask_token_id

        rnd = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & final_mask & ~rep
        random_chars = torch.randint(self.vocab_char_size, labels.shape, dtype=torch.long)
        input_ids[rnd] = random_chars[rnd]

        word_ids[final_mask] = self.unk_word_id

        return {
            "input_ids": input_ids,
            "word_ids": word_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
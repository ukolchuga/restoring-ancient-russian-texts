import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForMaskedLM, AutoTokenizer

app = FastAPI(title="Dobrynya Text Restoration")

# Static files and templates configuration
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cpu")

# Model configuration
MODEL_PATH_BPE = "AlexSychovUN/mini-roformer-ancient-rus-v2"
MODEL_PATH_CHAR = "./DualEmb-slav"

print("Loading BPE model...")
tokenizer_bpe = AutoTokenizer.from_pretrained(MODEL_PATH_BPE)
model_bpe = AutoModelForMaskedLM.from_pretrained(MODEL_PATH_BPE).to(device)
model_bpe.eval()

print("Loading character-level model...")
model_char = AutoModelForMaskedLM.from_pretrained(
    MODEL_PATH_CHAR, trust_remote_code=True
).to(device)
model_char.eval()

char_vocab = json.loads(
    Path("DualEmb-slav/char_vocab.json").read_text(encoding="utf-8")
)
word_vocab = json.loads(
    Path("DualEmb-slav/word_vocab.json").read_text(encoding="utf-8")
)
id_to_char = {v: k for k, v in char_vocab.items()}

SPECIAL_RE = re.compile(
    r"(\[CTX_[A-Z_]+\]|\[GAP\]|\[MASK\]|\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|[+:·])"
)


def split_special(text: str) -> list[str]:
    return [p for p in SPECIAL_RE.split(text) if p]


def align_char_to_word(text: str, char_v: dict, word_v: dict, max_len: int = 256):
    c_unk, c_pad, c_cls, c_sep = (
        char_v["[UNK]"],
        char_v["[PAD]"],
        char_v["[CLS]"],
        char_v["[SEP]"],
    )
    w_unk, w_pad = word_v.get("[UNK_WORD]", 0), word_v.get("[PAD_WORD]", 0)

    input_ids, word_ids = [c_cls], [word_v.get("[CLS]", w_unk)]

    for part in split_special(text.strip()):
        if SPECIAL_RE.fullmatch(part):
            input_ids.append(char_v.get(part, c_unk))
            word_ids.append(word_v.get(part, w_unk))
            continue
        chunks = re.split(r"(\s+)", part)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk.isspace():
                for ch in chunk:
                    input_ids.append(char_v.get(ch, c_unk))
                    word_ids.append(w_unk)
            else:
                wid = word_v.get(chunk, w_unk)
                for ch in chunk:
                    input_ids.append(char_v.get(ch, c_unk))
                    word_ids.append(wid)

    input_ids.append(c_sep)
    word_ids.append(word_v.get("[SEP]", w_unk))

    if len(input_ids) > max_len:
        input_ids, word_ids = input_ids[:max_len], word_ids[:max_len]
        input_ids[-1], word_ids[-1] = c_sep, word_v.get("[SEP]", w_unk)

    return {"input_ids": input_ids, "word_ids": word_ids}


class RestoreRequest(BaseModel):
    """
    Validation schema for the restoration API request.
    """

    text: str
    category: str
    mode: str = "char"
    top_k: int = 5
    temperature: float = 1.0


def generate_sequential(
    text: str,
    category: str,
    is_char: bool,
    top_k: int = 5,
    temperature: float = 1.0,
) -> List[List[Dict[str, Any]]]:
    """
    Performs 'Easy-First' Beam Search (sequential decoding).
    Dynamically fills the most confident masks first to provide better context for harder masks.
    Works for both character-level and BPE models.
    """
    if is_char:
        encoded = align_char_to_word(text, char_vocab, word_vocab)
        input_ids = torch.tensor(encoded["input_ids"]).to(device)
        word_ids = torch.tensor(encoded["word_ids"]).to(device)
        mask_token_id = char_vocab["[MASK]"]
        mask_str = "[MASK]"
        model = model_char
    else:
        inputs = tokenizer_bpe(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        word_ids = None
        mask_token_id = tokenizer_bpe.mask_token_id
        mask_str = tokenizer_bpe.mask_token
        model = model_bpe

    original_mask_indices = torch.where(input_ids == mask_token_id)[0].tolist()
    if not original_mask_indices:
        return []

    current_states = [
        {"input_ids": input_ids.clone(), "log_prob": 0.0, "inserted_tokens": {}}
    ]
    unfilled_masks = original_mask_indices.copy()

    with torch.no_grad():
        while unfilled_masks:
            # 1. FIND THE EASIEST MASK
            best_state_ids = current_states[0]["input_ids"].unsqueeze(0).to(device)

            if is_char:
                outputs = model(
                    input_ids=best_state_ids, word_ids=word_ids.unsqueeze(0)
                )
            else:
                outputs = model(input_ids=best_state_ids)
            logits = outputs.logits[0]

            best_mask_idx = None
            highest_prob = -1.0

            for m_idx in unfilled_masks:
                scaled_logits = logits[m_idx] / max(0.01, float(temperature))
                probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                max_prob = torch.max(probs).item()

                if max_prob > highest_prob:
                    highest_prob = max_prob
                    best_mask_idx = m_idx

            unfilled_masks.remove(best_mask_idx)

            # 2. EXPAND BEAMS FOR THE CHOSEN MASK (BATCHED)
            new_candidates = []
            batch_input_ids = torch.stack(
                [state["input_ids"] for state in current_states]
            ).to(device)

            if is_char:
                batch_word_ids = (
                    word_ids.unsqueeze(0).expand(len(current_states), -1).to(device)
                )
                outputs = model(input_ids=batch_input_ids, word_ids=batch_word_ids)
            else:
                outputs = model(input_ids=batch_input_ids)

            mask_logits = outputs.logits[:, best_mask_idx, :]
            scaled_mask_logits = mask_logits / max(0.01, float(temperature))
            mask_probs = torch.nn.functional.softmax(scaled_mask_logits, dim=-1)

            top_k_probs, top_k_indices = torch.topk(mask_probs, top_k, dim=-1)

            for state_idx, state in enumerate(current_states):
                for i in range(top_k):
                    token_id = top_k_indices[state_idx, i].item()
                    prob = top_k_probs[state_idx, i].item()

                    new_input_ids = state["input_ids"].clone()
                    new_input_ids[best_mask_idx] = token_id

                    # Track the inserted token at its specific index
                    new_inserted_tokens = dict(state["inserted_tokens"])
                    new_inserted_tokens[best_mask_idx] = token_id

                    new_log_prob = state["log_prob"] + math.log(max(prob, 1e-9))

                    new_candidates.append(
                        {
                            "input_ids": new_input_ids,
                            "log_prob": new_log_prob,
                            "inserted_tokens": new_inserted_tokens,
                        }
                    )

            # Retain top-K candidates
            current_states = sorted(
                new_candidates, key=lambda x: x["log_prob"], reverse=True
            )[:top_k]

    # HTML build, different for char-level vs BPE due to tokenization differences
    variants = []
    escaped_mask = html.escape(mask_str)
    escaped_cat = html.escape(f"[{category}]")

    for state in current_states:
        ordered_ids = [state["inserted_tokens"][idx] for idx in original_mask_indices]
        full_sentence = html.escape(text)
        inserted_phrase = ""

        if is_char:
            inserted_phrase = "".join(
                [id_to_char.get(tid, "") for tid in ordered_ids]
            ).strip()
            for token_id in ordered_ids:
                char_str = id_to_char.get(token_id, "")
                clean_token = "&nbsp;" if char_str == " " else html.escape(char_str)
                full_sentence = full_sentence.replace(
                    escaped_mask,
                    f'<span class="highlight-restored">{clean_token}</span>',
                    1,
                )
        else:
            inserted_phrase = tokenizer_bpe.decode(
                ordered_ids, clean_up_tokenization_spaces=True
            ).strip()
            for token_id in ordered_ids:
                token_str = tokenizer_bpe.decode([token_id])
                clean_token = html.escape(
                    token_str.replace("Ġ", "").replace("##", "").replace(" ", "")
                )
                full_sentence = full_sentence.replace(
                    escaped_mask,
                    f'<span class="highlight-restored">{clean_token}</span>',
                    1,
                )

        full_sentence = re.sub(
            r"\s+", " ", full_sentence.replace(escaped_cat, "").strip()
        )
        variants.append(
            {
                "word": inserted_phrase or "...",
                "score": round(math.exp(state["log_prob"]) * 100, 2),
                "full_sentence": full_sentence,
                "raw_log_prob": state["log_prob"],
            }
        )

    return variants


@app.get("/")
async def read_root(request: Request):
    """
    Renders the main frontend page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/restore")
async def restore_text(req: RestoreRequest) -> Dict[str, Any]:
    """
    Main API endpoint for text restoration.
    BPE/Character-level support, and Leiden special characters (-, #, *).
    """
    try:
        is_char = req.mode == "char"
        mask = "[MASK]" if is_char else tokenizer_bpe.mask_token
        text = req.text.replace("#", "[GAP]")

        if not is_char:
            # BPE, -, * -> one mask token each because bpe predicts partial words
            text = text.replace("-", mask).replace("*", mask)
            query = f"[{req.category}] {re.sub(r' +', ' ', text).strip()}"
            return {
                "status": "success",
                "results": [
                    generate_sequential(
                        query, req.category, False, req.top_k, req.temperature
                    )
                ],
            }

        else:
            # Character-level, - -> exactly on mask token

            base_text = text.replace("-", mask)
            if "*" in base_text:
                best_results, best_score = [], -float("inf")
                for length in range(1, 6):
                    test_text = base_text.replace("*", mask * length)
                    query = f"[{req.category}] {re.sub(r' +', ' ', test_text).strip()}"
                    variants = generate_sequential(
                        query, req.category, True, req.top_k, req.temperature
                    )

                    if isinstance(variants, list) and variants:
                        if (
                            "raw_log_prob" in variants[0]
                            and variants[0]["raw_log_prob"] > best_score
                        ):
                            best_score, best_results = (
                                variants[0]["raw_log_prob"],
                                variants,
                            )
                return {"status": "success", "results": [best_results]}
            else:
                query = f"[{req.category}] {re.sub(r' +', ' ', base_text).strip()}"
                return {
                    "status": "success",
                    "results": [
                        generate_sequential(
                            query, req.category, True, req.top_k, req.temperature
                        )
                    ],
                }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Start the development server
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # Settings for deploy hugging spaces
    # import os
    # port = int(os.environ.get("PORT", 7860))
    # uvicorn.run(app, host="0.0.0.0", port=port)

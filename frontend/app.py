import html
import math
import re
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
# TODO: Update MODEL_PATH_CHAR when character-level model is available
MODEL_PATH_BPE = "AlexSychovUN/mini-roformer-ancient-rus-v2"
MODEL_PATH_CHAR = "AlexSychovUN/mini-roformer-ancient-rus-v2"

print("Loading BPE model...")
tokenizer_bpe = AutoTokenizer.from_pretrained(MODEL_PATH_BPE)
model_bpe = AutoModelForMaskedLM.from_pretrained(MODEL_PATH_BPE).to(device)
model_bpe.eval()

print("Loading character-level model...")
tokenizer_char = AutoTokenizer.from_pretrained(MODEL_PATH_CHAR)
model_char = AutoModelForMaskedLM.from_pretrained(MODEL_PATH_CHAR).to(device)
model_char.eval()


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
    tokenizer,
    model,
    is_char_level: bool,
    top_k: int = 5,
    temperature: float = 1.0,
) -> List[List[Dict[str, Any]]]:
    """
    Performs 'Easy-First' Beam Search (sequential decoding).
    Dynamically fills the most confident masks first to provide better context for harder masks.
    Works for both character-level and BPE models.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]

    mask_token_id = tokenizer.mask_token_id
    original_mask_indices = torch.where(input_ids == mask_token_id)[0].tolist()

    if not original_mask_indices:
        return []

    # State tracking: inserted_tokens is now a dict {mask_index: token_id}
    current_states = [
        {"input_ids": input_ids.clone(), "log_prob": 0.0, "inserted_tokens": {}}
    ]
    unfilled_masks = original_mask_indices.copy()

    with torch.no_grad():
        while unfilled_masks:
            # 1. FIND THE EASIEST MASK
            best_state_ids = current_states[0]["input_ids"].unsqueeze(0).to(device)
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
    mask_token = tokenizer.mask_token
    escaped_mask = html.escape(mask_token)
    escaped_category = html.escape(f"[{category}]")

    for state in current_states:
        # Reconstruct the tokens in their ORIGINAL left-to-right order for the UI
        ordered_inserted_ids = [
            state["inserted_tokens"][idx] for idx in original_mask_indices
        ]

        inserted_phrase = tokenizer.decode(
            ordered_inserted_ids, clean_up_tokenization_spaces=True
        ).strip()

        full_sentence = html.escape(text)

        # Replace asterisks one by one in left-to-right order
        for token_id in ordered_inserted_ids:
            token_str = tokenizer.decode([token_id])

            if is_char_level:
                clean_token = "&nbsp;" if token_str == " " else html.escape(token_str)
            else:
                clean_token = (
                    token_str.replace("Ġ", "").replace("##", "").replace(" ", "")
                )
            replacement = f'<span class="highlight-restored">{clean_token}</span>'
            full_sentence = full_sentence.replace(escaped_mask, replacement, 1)

        full_sentence = full_sentence.replace(escaped_category, "").strip()
        full_sentence = re.sub(r"\s+", " ", full_sentence)

        final_prob = math.exp(state["log_prob"])

        variants.append(
            {
                "word": inserted_phrase if inserted_phrase else "...",
                "score": round(final_prob * 100, 2),
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
        tokenizer = tokenizer_char if is_char else tokenizer_bpe
        model = model_char if is_char else model_bpe
        mask = tokenizer.mask_token

        text = req.text.replace("#", "[GAP]")

        if not is_char:
            # BPE, -, * -> one mask token each because bpe predicts partial words
            text = text.replace("-", mask).replace("*", mask)
            text = re.sub(r"\s+", " ", text).strip()
            query = f"[{req.category}] {text}"

            results = generate_sequential(
                query,
                req.category,
                tokenizer,
                model,
                is_char,
                req.top_k,
                req.temperature,
            )
            return {"status": "success", "results": [results]}

        else:
            # Character-level, - -> exactly on mask token

            base_text = text.replace("-", mask)

            if "*" in base_text:
                best_results = []
                best_score = -float("inf")

                for length in range(1, 6):
                    test_text = base_text.replace("*", mask * length)
                    test_text = re.sub(r"\s+", " ", test_text).strip()
                    query = f"[{req.category}] {test_text}"

                    variants = generate_sequential(
                        query,
                        req.category,
                        tokenizer,
                        model,
                        is_char,
                        req.top_k,
                        req.temperature,
                    )

                    if variants:
                        top_variant_score = variants[0]["raw_log_prob"]
                        if top_variant_score > best_score:
                            best_score = top_variant_score
                            best_results = variants

                return {"status": "success", "results": [best_results]}

            else:
                base_text = re.sub(r"\s+", " ", base_text).strip()
                query = f"[{req.category}] {base_text}"
                results = generate_sequential(
                    query,
                    req.category,
                    tokenizer,
                    model,
                    is_char,
                    req.top_k,
                    req.temperature,
                )
                return {"status": "success", "results": [results]}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Start the development server
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # Settings for deploy hugging spaces
    # import os
    # port = int(os.environ.get("PORT", 7860))
    # uvicorn.run(app, host="0.0.0.0", port=port)

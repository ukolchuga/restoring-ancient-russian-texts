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

# Model configuration
MODEL_PATH = "AlexSychovUN/mini-roformer-ancient-rus-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer once on startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


class RestoreRequest(BaseModel):
    """
    Validation schema for the restoration API request.
    """

    text: str
    category: str
    top_k: int = 5
    temperature: float = 1.0


def generate_sequential(
    text: str, category: str, top_k: int = 5, temperature: float = 1.0
) -> List[List[Dict[str, Any]]]:
    """
    Performs 'Easy-First' sequential decoding (Ithaca-style Beam Search).
    Dynamically fills the most confident masks first to provide better context for harder masks.
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

    # 3. HTML RENDERING
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
            clean_token = token_str.replace("Ġ", "").replace("##", "").replace(" ", "")
            replacement = (
                f'<span class="highlight-restored">{html.escape(clean_token)}</span>'
            )
            full_sentence = full_sentence.replace(escaped_mask, replacement, 1)

        full_sentence = full_sentence.replace(escaped_category, "").strip()
        full_sentence = re.sub(r"\s+", " ", full_sentence)

        final_prob = math.exp(state["log_prob"])

        variants.append(
            {
                "word": inserted_phrase if inserted_phrase else "...",
                "score": round(final_prob * 100, 2),
                "full_sentence": full_sentence,
            }
        )

    return [variants]


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
    Normalizes input symbols (* and -) and processes through the sequential generator.
    """
    # Normalize input: * -> <mask>, - -> [GAP]
    formatted_text = req.text.replace("*", tokenizer.mask_token).replace("-", "[GAP]")
    formatted_text = re.sub(r"\s+", " ", formatted_text).strip()

    # Prepend category tag
    full_query = f"[{req.category}] {formatted_text}"

    try:
        cleaned_results = generate_sequential(
            full_query, req.category, top_k=req.top_k, temperature=req.temperature
        )
        return {"status": "success", "results": cleaned_results}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Start the development server
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # Settings for deploy hugging spaces
    # import os
    # port = int(os.environ.get("PORT", 7860))
    # uvicorn.run(app, host="0.0.0.0", port=port)

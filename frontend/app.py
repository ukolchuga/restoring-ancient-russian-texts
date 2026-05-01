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
    Performs sequential decoding (simplified beam search) for multiple mask tokens.
    Iteratively fills masks based on model confidence and contextual probability.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]

    mask_token_id = tokenizer.mask_token_id
    mask_indices = torch.where(input_ids == mask_token_id)[0].tolist()

    if not mask_indices:
        return []

    # State tracking: (tensor, accumulated log probability, list of inserted token IDs)
    current_states = [
        {"input_ids": input_ids.clone(), "log_prob": 0.0, "inserted_ids": []}
    ]

    with torch.no_grad():
        for mask_idx in mask_indices:
            new_candidates = []

            for state in current_states:
                model_inputs = {"input_ids": state["input_ids"].unsqueeze(0).to(device)}
                outputs = model(**model_inputs)

                logits = outputs.logits[0, mask_idx, :]
                scaled_logits = logits / max(0.01, float(temperature))
                probs = torch.nn.functional.softmax(scaled_logits, dim=-1)

                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

                for i in range(top_k):
                    token_id = top_k_indices[i].item()
                    prob = top_k_probs[i].item()

                    new_input_ids = state["input_ids"].clone()
                    new_input_ids[mask_idx] = token_id

                    new_inserted_ids = list(state["inserted_ids"])
                    new_inserted_ids.append(token_id)

                    new_log_prob = state["log_prob"] + math.log(prob)

                    new_candidates.append(
                        {
                            "input_ids": new_input_ids,
                            "log_prob": new_log_prob,
                            "inserted_ids": new_inserted_ids,
                        }
                    )

            # Retain top-K candidates for the next mask position
            current_states = sorted(
                new_candidates, key=lambda x: x["log_prob"], reverse=True
            )[:top_k]

    # Decode predictions back to text
    variants = []
    mask_token = tokenizer.mask_token

    escaped_mask = html.escape(mask_token)
    escaped_category = html.escape(f"[{category}]")

    for state in current_states:
        inserted_phrase = tokenizer.decode(
            state["inserted_ids"], clean_up_tokenization_spaces=True
        ).strip()

        full_sentence = html.escape(text)

        for token_id in state["inserted_ids"]:
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

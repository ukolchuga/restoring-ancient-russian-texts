import html
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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


MODEL_PATH = "AlexSychovUN/mini-roformer-ancient-rus-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


class RestoreRequest(BaseModel):
    """Data Model (scheme) of the user input"""

    text: str
    category: str
    top_k: int = 5
    temperature: float = 1.0


def get_mask_predictions(
    text: str, top_k: int = 5, temperature: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Inference function for prediction ONLY for the first mask

    Args:
        text (str): input text with tokens <mask>.
        top_k (int): Number of output examples.
        temperature (float): Softmax probability smoothing coefficient.

    Returns:
        List[Dict[str, Any]]: List of dicts [{"token_str": str, "score": float}, ...]
    """
    # Text tokenization
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Find indexes for all mask tokens
    mask_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    if len(mask_indices) == 0:
        return []

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits only for the first mask
    first_mask_index = mask_indices[0]
    logits = outputs.logits[0, first_mask_index, :]

    temp = max(0.01, float(temperature))
    scaled_logits = logits / temp

    probs = torch.nn.functional.softmax(
        scaled_logits, dim=-1
    )  # Sum of probabilities = 1.0
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

    results = []
    for i in range(top_k):
        token_id = top_k_indices[i].item()
        prob = top_k_probs[i].item()
        token_str = tokenizer.decode([token_id])
        results.append({"token_str": token_str, "score": prob})

    return results


def generate_sequential(
    text: str, category: str, top_k: int = 5, temperature: float = 1.0
):
    mask_token = tokenizer.mask_token  # <mask>
    masks_count = text.count(mask_token)

    if masks_count == 0:
        return []

    # Start state: one node with the possibility 1.0 (100%)
    current_states = [{"text": text, "prob": 1.0, "inserted": []}]

    # Iteration through all masks
    for step in range(masks_count):
        new_candidates = []

        # Unfold hypotheses for each current state (beam)
        for state in current_states:
            step_results = get_mask_predictions(
                state["text"], top_k=top_k, temperature=temperature
            )

            for res in step_results:
                # Cleaning tokens from special tokens BPE (Ġ - space before word)
                clean_token = res["token_str"].replace("Ġ", "")
                if not clean_token.strip():
                    continue

                # Prediction for the first mask
                new_text = state["text"].replace(mask_token, clean_token, 1)

                # Chain rule
                new_prob = state["prob"] * res["score"]

                new_inserted = list(state["inserted"])
                new_inserted.append(clean_token)

                new_candidates.append(
                    {"text": new_text, "prob": new_prob, "inserted": new_inserted}
                )
        # Sorting the branches and saving only the best ones (Beam Width)
        current_states = sorted(new_candidates, key=lambda x: x["prob"], reverse=True)[
            : top_k * 2
        ]
    # Save only top_k variants
    current_states = current_states[:top_k]

    variants = []
    for state in current_states:
        inserted_phrase = "".join(state["inserted"]).strip()
        full_sentence = (
            state["text"]
            .replace(f"[{category}]", "")
            .replace(tokenizer.cls_token, "")  # <s>
            .replace(tokenizer.sep_token, "")  # </s>
            .strip()
        )
        full_sentence = re.sub(r"\s+", " ", full_sentence)
        full_sentence = html.escape(full_sentence)

        variants.append(
            {
                "word": inserted_phrase if inserted_phrase else "...",
                "score": round(state["prob"] * 100, 2),
                "full_sentence": full_sentence,
            }
        )

    return [variants]


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/restore")
async def restore_text(req: RestoreRequest) -> Dict[str, Any]:
    # Input normalization: * -> <mask >, - -> [GAP]
    formatted_text = req.text.replace("*", f" {tokenizer.mask_token} ").replace(
        "-", " [GAP] "
    )
    formatted_text = re.sub(r"\s+", " ", formatted_text).strip()

    # TAG + TEXT
    full_query = f"[{req.category}] {formatted_text}"

    try:
        cleaned_results = generate_sequential(
            full_query, req.category, top_k=req.top_k, temperature=req.temperature
        )
        return {"status": "success", "results": cleaned_results}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # Settings for deploy hugging spaces
    # import os
    # port = int(os.environ.get("PORT", 7860))
    # uvicorn.run(app, host="0.0.0.0", port=port)

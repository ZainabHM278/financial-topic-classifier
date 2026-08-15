"""
Financial Topic Classifier — Inference Example

Loads the LoRA-fine-tuned Qwen2.5-0.5B model and classifies a piece of raw
financial text into one of 20 topic categories.

Usage:
    python inference_example.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# Path or Hugging Face Hub repo ID where the LoRA adapter was saved.
# e.g. "./lora-financial-topic-classifier" (local) or
#      "your-hf-username/financial-topic-classifier-lora" (Hub)
ADAPTER_PATH = "./lora-financial-topic-classifier"

LABEL_NAMES = [
    "Analyst Update",
    "Fed | Central Banks",
    "Company | Product News",
    "Treasuries | Corporate Debt",
    "Dividend",
    "Earnings",
    "Energy | Oil",
    "Financials",
    "Currencies",
    "General News | Opinion",
    "Gold | Metals | Materials",
    "IPO",
    "Legal | Regulation",
    "M&A | Investments",
    "Macro",
    "Markets",
    "Politics",
    "Personnel Change",
    "Stock Commentary",
    "Stock Movement",
]

MAX_LENGTH = 96

# ---------------------------------------------------------------------------
# Load tokenizer, base model, and LoRA adapter
# ---------------------------------------------------------------------------


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_NAMES),
        id2label=id2label,
        label2id=label2id,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Attach the fine-tuned LoRA adapter weights on top of the base model
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------


def predict_topic(text, tokenizer, model):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    predicted_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_id].item()

    return {
        "predicted_topic": LABEL_NAMES[predicted_id],
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tokenizer, model = load_model()

    examples = [
        "Apple reports record quarterly earnings, beating analyst expectations",
        "The Federal Reserve signaled it may pause rate hikes amid slowing inflation",
        "MAIA Biotech files for a $15 million IPO on Nasdaq",
        "Global markets react as trade tensions escalate between US and China",
    ]

    for text in examples:
        result = predict_topic(text, tokenizer, model)
        print(f"Text: {text}")
        print(f"  -> {result}\n")

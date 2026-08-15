# Financial Topic Classifier — Qwen2.5-0.5B + LoRA

## Overview

This project fine-tunes **Qwen2.5-0.5B**, a small open-source language model, to classify short financial texts (tweets) into one of **20 topic categories** — including Earnings, M&A, IPO, Legal/Regulation, Dividend, and Fed/Central Banks. Rather than full fine-tuning, the model is adapted using **LoRA (Low-Rank Adaptation)** via Hugging Face's `peft` library — a parameter-efficient technique that trains only a small fraction of the model's weights while keeping the rest frozen.

The project was built to demonstrate practical LLM fine-tuning skills for AI/ML engineering roles: dataset handling, tokenization decisions grounded in data, parameter-efficient fine-tuning, rigorous evaluation methodology, and honest interpretation of results — including class imbalance and a real data-integrity bug caught and fixed mid-project.

**Headline result:** fine-tuning just **0.22% of the model's ~495M parameters** achieved **90.5% test accuracy** and **89.2% macro F1** on a held-out, never-touched test set.

---

## Dataset

- **Source:** [`zeroshot/twitter-financial-news-topic`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) on Hugging Face
- **Size:** 16,990 training examples, 4,117 validation examples (21,107 total)
- **Task:** single-label classification across 20 financial topic categories
- **Label format:** labels shipped as plain integers (`Value('int64')`), not Hugging Face's `ClassLabel` type — the human-readable category names had to be recovered from the dataset card rather than the schema itself.

### Class balance

The dataset is significantly imbalanced — roughly an **80x gap** between the largest and smallest class:

| Category | Train count |
|---|---|
| Company \| Product News (largest) | 3,545 |
| Stock Commentary | 2,118 |
| Macro | 1,822 |
| ... | ... |
| Currencies | 166 |
| Gold \| Metals \| Materials | 69 |
| IPO (smallest) | 44 |

This imbalance directly informed the choice to track **macro F1** (unweighted average across all 20 classes) alongside accuracy, since accuracy alone can mask poor performance on rare classes.

---

## Model & Technique

**Base model:** [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) — a ~494M parameter, dense, decoder-only transformer (Apache 2.0 licensed). The *base* (non-instruction-tuned) variant was used deliberately, since the task uses a sequence-classification head rather than conversational prompting.

**Trained adapter:** [ZainabHM278/financial-topic-classifier-lora](https://huggingface.co/ZainabHM278/financial-topic-classifier-lora) — the fine-tuned LoRA weights, hosted on the Hugging Face Hub.

**Fine-tuning technique: LoRA** (via `peft`)

Instead of updating all ~494M parameters, LoRA freezes the base model's pretrained weights and injects small trainable low-rank matrices (`A`, `B`) alongside targeted layers. The effective weight becomes `W_frozen + (B × A)`, and only `A`/`B` are trained.

**LoRA configuration:**
| Parameter | Value | Rationale |
|---|---|---|
| `r` (rank) | 8 | Standard, conservative default |
| `lora_alpha` | 16 | Conventional 2× rank scaling |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention projection layers, the standard/baseline LoRA target |
| `lora_dropout` | 0.05 | Regularization |

**Result:** 1,099,264 trainable parameters out of 495,149,952 total — **0.22%** trainable. This figure was independently verified by manually computing the adapter matrix sizes across all 24 transformer layers plus the classification head, confirming it matched `peft`'s reported count exactly.

---

## Methodology

### Tokenization

Rather than guessing a `max_length`, the actual token-length distribution of the training set was measured first:

| Statistic | Value |
|---|---|
| Min | 2 |
| Median | 39 |
| Mean | 40.7 |
| 95th percentile | 67 |
| 99th percentile | 89 |
| Max | 168 |

**`max_length=96`** was chosen to comfortably cover the 99th percentile with minimal truncation, while avoiding the padding overhead of a larger, unjustified round number.

### Train / Dev / Test Split

The dataset ships with only `train` and `validation` splits — no separate test set. To avoid indirectly tuning against the same data used for final reporting, a **90/10 stratified split** was carved out of the original `train` set:

- **Train** (15,291 examples) — used for gradient updates
- **Dev** (1,699 examples) — used during training for monitoring and best-checkpoint selection
- **Test** (4,117 examples, the original `validation` split) — held out entirely, used exactly once for final reported metrics

Stratification (by label) was used to ensure rare classes like IPO (44 total examples) were proportionally represented in both splits rather than risking under-representation by chance.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 16 (per device, T4 x2) |
| Learning rate | 2e-4 |
| Eval/save strategy | Per epoch |
| Best checkpoint selection | Macro F1 on dev set |

Training completed in ~40 minutes on Kaggle's T4 x2 GPUs.

---

## Results

### Final test set metrics (held out, evaluated once)

| Metric | Score |
|---|---|
| Accuracy | **90.5%** |
| Macro F1 | **89.2%** |

### Per-class highlights

- **Strongest classes:** Dividend (F1 0.98), Earnings (F1 0.97), Personnel Change (F1 0.96) — linguistically distinctive categories with ample training data.
- **Rare classes performed better than expected:** IPO (F1 0.90, only 40 training examples) and Currencies (F1 0.84, 166 examples) held up well, suggesting these categories have distinctive enough language patterns to partially offset low support.
- **Weakest class:** Gold | Metals | Materials (F1 0.79, only 13 test examples) — the smallest test support of any class, making this score the least statistically reliable in the report.

### Confusion patterns

The confusion matrix revealed a genuine three-way overlap between **Macro**, **Fed | Central Banks**, and **General News | Opinion** — categories that manual inspection of raw examples (before training) had already flagged as conceptually fuzzy (e.g., a tweet about the Fed's rate decisions plausibly fits either "Macro" or "Fed | Central Banks" depending on framing). This was confirmed both qualitatively (reading examples) and quantitatively (the trained model's actual misclassification pattern), rather than being an unexplained model weakness.

A smaller, similarly explainable confusion appeared between **Stock Commentary** and **Stock Movement** — plausibly reflecting a genuinely subtle real-world labeling distinction rather than a model shortcoming.

---

## Inference Example

```python
predict_topic("Apple reports record quarterly earnings, beating analyst expectations")
# {'predicted_topic': 'Earnings', 'confidence': 0.9883}

predict_topic("MAIA Biotech files for a $15 million IPO on Nasdaq")
# {'predicted_topic': 'IPO', 'confidence': 0.9922}
```

**Notable finding:** even deliberately ambiguous inputs (e.g., text straddling Macro/Fed/General News boundaries) consistently produced high-confidence (~100%) predictions. This suggests the model's softmax confidence is **not a reliable signal of genuine topic ambiguity** — it's calibrated to the clear-cut structure of the training distribution, not to edge cases. This is a meaningful caveat for any downstream use of confidence scores.

---

## Challenges & Learnings

- **Label mapping recovery:** the dataset's label column was a plain integer type, not Hugging Face's `ClassLabel` — the human-readable mapping had to be sourced from the dataset card rather than the schema, establishing a general checklist for handling this in future projects.
- **A real data-integrity bug, caught before it mattered:** `datasets.class_encode_column()` was initially used to enable stratified splitting, but it silently re-encodes label integers based on **alphabetically sorted string order** (`'0','1','10','11',...`) rather than numeric order — corrupting the label-to-index mapping. This was caught via a sanity check (verifying the rare "IPO" class's example count matched expectations before and after the split) rather than by any error message, since the bug never raised an exception. The fix was to build a `ClassLabel` directly from the already-verified `label_names` list and cast to it explicitly, rather than relying on automatic encoding.
- **Model/tokenizer config mismatch:** training initially failed with `"Cannot handle batch sizes > 1 if no padding token is defined"` — despite the tokenizer already having a correctly configured pad token. The base model's separate `config.pad_token_id` was `None` by default (since Qwen2.5-0.5B was never pretrained with a classification/padding convention), and had to be explicitly set to match the tokenizer before training.

---

## Limitations & Future Work

- **Thin rare-class evaluation:** classes like Gold | Metals | Materials had only 13 test examples — individual scores for the smallest classes should be treated as noisy estimates, not precise measurements.
- **Confidence miscalibration:** as noted above, the model does not reliably express uncertainty on ambiguous inputs. A calibration technique (e.g., temperature scaling) could be explored.
- **Possible next steps:** class weighting or oversampling for rare categories, testing higher LoRA rank (`r=16`) for additional capacity, training for additional epochs (validation loss was still improving at epoch 3), or comparing against a full fine-tuning baseline to quantify LoRA's performance trade-off directly.

---

## How to Run

**To retrain from scratch** (see `notebook.ipynb` for the full walkthrough):
1. Load the dataset: `datasets.load_dataset("zeroshot/twitter-financial-news-topic")`
2. Tokenize with `Qwen/Qwen2.5-0.5B`'s tokenizer, `max_length=96`
3. Load `Qwen/Qwen2.5-0.5B` via `AutoModelForSequenceClassification` (`num_labels=20`)
4. Apply LoRA via `peft.LoraConfig` / `get_peft_model` (see configuration table above)
5. Train with Hugging Face `Trainer` using the hyperparameters above
6. Evaluate on the held-out test set

**To run inference with the already-trained model** (no retraining needed):

```bash
pip install -r requirements.txt
python inference_example.py
```

This loads `Qwen/Qwen2.5-0.5B` as the base model and attaches the fine-tuned LoRA adapter directly from the Hugging Face Hub ([ZainabHM278/financial-topic-classifier-lora](https://huggingface.co/ZainabHM278/financial-topic-classifier-lora)) — no local model files required.

**Environment:** Kaggle Notebooks, GPU T4 x2, Python, `transformers`, `datasets`, `peft`, `scikit-learn`.

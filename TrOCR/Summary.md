# A Descriptive Study on Teacher-Student Knowledge Distillation for OCR

## Table of Contents

- [Abstract](#abstract)
- [1. The Distillation Pattern](#1-the-distillation-pattern)
- [2. Teacher & Student Profiles](#2-teacher--student-profiles)
- [3. Cost Analysis](#3-cost-analysis)
- [4. What Gets Distilled — And What Doesn't](#4-what-gets-distilled--and-what-doesnt)
- [5. Relation to Knowledge Distillation Literature](#5-relation-to-knowledge-distillation-literature)
- [6. Key Takeaways](#6-key-takeaways)

---

## Abstract

This study describes how a 235-billion-parameter vision-language model (Qwen3-VL-235B) was used as a **teacher** to generate training labels for a 558-million-parameter OCR model (TrOCR-Large) — the **student**. The student achieved a Character Error Rate (CER) of 0.000765 on domain-specific table cell images, effectively inheriting the teacher's comprehension at **421× fewer parameters** and near-zero marginal inference cost. This document frames the approach within the broader knowledge distillation literature and analyses the cost-accuracy trade-off. For training specifics (hyperparameters, infrastructure, checkpointing), refer to the companion `MODEL_TRAINING_REPORT.md`.

---

## 1. The Distillation Pattern

Knowledge distillation traditionally involves training a smaller "student" network to replicate a larger "teacher" network's output distributions (soft targets). This project uses a variant known as **data-augmentation-based distillation**: the teacher generates hard labels (exact OCR transcriptions) on unlabeled images, and the student is fine-tuned on those labels using standard cross-entropy loss. The student never accesses the teacher's logits or internal representations.

```
  ┌──────────────────────────┐
  │       TEACHER             │
  │  Qwen3-VL-235B-A22B      │     One-time labeling
  │  235B params (22B active) │ ──────────────────────►  129,041 labeled images
  │  AWS Bedrock API          │
  └──────────────────────────┘
                                          │
                                          ▼
  ┌──────────────────────────┐    ┌──────────────────────┐
  │       STUDENT             │    │     DEPLOYED MODEL    │
  │  TrOCR-Large (Stage 1)   │ ──►│  558M params · 2.3 GB │
  │  558M params              │    │  Single GPU · ~ms/img  │
  │  Fine-tuned 10 epochs     │    │  CER: 0.000765        │
  └──────────────────────────┘    └──────────────────────┘
```

This approach works because the teacher's intelligence is **captured in the data** rather than transferred through gradient flow. The student sees the same images the teacher labelled, learns the same input-output mapping, and generalises within the domain.

---

## 2. Teacher & Student Profiles

| Dimension              | Teacher (Qwen3-VL-235B)            | Student (TrOCR-Large)           |
|------------------------|------------------------------------|---------------------------------|
| **Architecture**       | Mixture-of-Experts VLM             | ViT encoder + TrOCR decoder     |
| **Total parameters**   | 235,000M                           | 558M                            |
| **Active parameters**  | 22,000M                            | 558M                            |
| **Parameter ratio**    | —                                  | **421× smaller** (total) / **39× smaller** (active) |
| **Weights on disk**    | Not self-hosted (API)              | 2.3 GB (safetensors)            |
| **Inference mode**     | AWS Bedrock managed API            | Self-hosted, single A10G GPU    |
| **Input**              | Image + system prompt              | Image only (384×384)            |
| **Output**             | Free-form text                     | Constrained OCR tokens          |
| **Strengths**          | General-purpose vision-language    | Domain-specialised, fast        |

The teacher is a general-purpose model capable of understanding arbitrary images and following complex instructions. The student is a purpose-built OCR model that does one thing — read table cell text — but does it extremely well and extremely cheaply.

---

## 3. Cost Analysis

### Labeling (Teacher) — One-Time Cost

The teacher model on AWS Bedrock is priced at **$0.53 per 1M input tokens** and **$2.66 per 1M output tokens**. Each image requires a system prompt (~200 tokens), an image (variable token count depending on resolution), and produces a short transcription (~5–20 output tokens). For 129,041 images, the total labeling cost is a one-time fixed expense.

### Inference (Student) — Recurring Cost

The student runs on a single `ml.g5.xlarge` instance (1× A10G, ~$1.01/hr on-demand for SageMaker inference). At batch-32 throughput, it processes thousands of images per hour.

Crucially, **inference runs only 5–10 minutes per day** in practice. The system caches results, so the model is only invoked for new or changed images — not on every request. This collapses the effective daily compute time to a small window.

| Time Horizon | Billed Compute          | Estimated Cost         |
|--------------|-------------------------|------------------------|
| Per day      | 5–10 min × $1.01/hr     | **~$0.08–$0.17/day**   |
| Per month    | ~2.5–5 hr active        | **~$2.50–$5.00/month** |
| Per year     | ~30–60 hr active        | **~$30–$60/year**      |

The marginal cost per image, when amortised over cached repeat lookups, is effectively zero.

### The Trade-Off

| Scenario                                  | Approach                     | Ongoing Cost Profile        |
|-------------------------------------------|------------------------------|-----------------------------|
| Run the teacher on every new image        | Bedrock API per call         | Scales linearly with volume |
| Label once with teacher, deploy student   | Fixed label cost + cheap GPU | Near-flat after labeling    |

For any workload beyond the initial labeling volume, the student approach dominates. The caching layer amplifies this advantage further: the model handles the **incremental** workload each day, not the full corpus. Annual inference cost at this usage pattern is comparable to a single day's worth of teacher API calls during labeling.

---

## 4. What Gets Distilled — And What Doesn't

### Successfully transferred

- **Character-level OCR accuracy**: The student achieves 0.000765 CER (~1 error per 1,300 characters), matching the teacher's labeling quality on the target domain.
- **Domain vocabulary**: Dates, currencies, numeric formats, header text — all well represented in the 129K training samples.
- **Robustness to font/style variation**: The teacher labeled images across multiple visual styles; the student generalised across them.

### Not transferred

- **General vision-language understanding**: The student cannot answer questions about images, follow open-ended instructions, or process anything outside its trained domain.
- **Out-of-domain generalisation**: Novel table formats, languages, or image degradation patterns not present in the training set may produce errors.
- **Reasoning about content**: The teacher can interpret what a cell value *means*; the student only transcribes what it *sees*.

This is the fundamental trade-off of distillation: the student inherits task-specific performance but not the teacher's breadth.

---

## 5. Relation to Knowledge Distillation Literature

| Distillation Type           | How It Works                                    | This Project              |
|-----------------------------|-------------------------------------------------|---------------------------|
| **Soft-target (classic)**   | Student mimics teacher's probability distributions (logits) at temperature T | Not used — no access to teacher logits |
| **Feature-based**           | Student aligns intermediate representations with teacher's hidden layers | Not used — different architectures |
| **Data augmentation-based** | Teacher generates labeled data; student trains on it with standard loss | **This is what we do** |
| **Self-distillation**       | Model distills knowledge from its own deeper layers to shallower ones | Not applicable |

The data-augmentation approach is increasingly common in the LLM era, where large API-based models (GPT-4, Claude, Qwen) generate training data for smaller task-specific models. The key advantage is simplicity: no custom distillation loss functions, no architecture alignment requirements — just high-quality labeled data and standard supervised training.

---

## 6. Key Takeaways

1. **A 235B-parameter model's OCR capability can be effectively compressed into a 558M-parameter model** through label-based distillation, achieving near-perfect domain accuracy.

2. **The cost structure inverts after distillation**: the teacher's per-query API cost is replaced by a fixed training cost plus near-zero inference cost.

3. **No logit access is required**: hard-label distillation is sufficient for well-defined tasks like OCR where the output space is constrained.

4. **The student's limitation is scope, not quality**: within its domain, the student matches the teacher; outside it, the student has no capability.

5. **This pattern is reusable**: any task where a large VLM can generate reliable labels on unlabeled data is a candidate for the same pipeline — document classification, table structure recognition, form field extraction, and beyond.

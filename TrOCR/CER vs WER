# CER vs WER: Understanding OCR Evaluation Metrics
 
## 1. Definitions
 
### Character Error Rate (CER)
 
CER measures accuracy at the **individual character** level using the Levenshtein (edit) distance between the predicted and reference strings.
 
```
CER = (S + D + I) / N
```
 
| Symbol | Meaning                              |
|--------|--------------------------------------|
| S      | Character substitutions              |
| D      | Character deletions                  |
| I      | Character insertions                 |
| N      | Total characters in the reference    |
 
**Example** — Reference: `$1,234` → Prediction: `$1.234`
 
- 1 substitution (`,` → `.`), 0 deletions, 0 insertions
- CER = 1 / 6 = **0.1667** (16.7%)
 
### Word Error Rate (WER)
 
WER applies the same edit-distance formula but operates on **whole words** (whitespace-delimited tokens).
 
```
WER = (S_w + D_w + I_w) / N_w
```
 
| Symbol | Meaning                          |
|--------|----------------------------------|
| S_w    | Word substitutions               |
| D_w    | Word deletions                   |
| I_w    | Word insertions                  |
| N_w    | Total words in the reference     |
 
**Example** — Reference: `net revenue increased` → Prediction: `net revnue increased`
 
- 1 word substitution (`revenue` → `revnue`), 0 deletions, 0 insertions
- WER = 1 / 3 = **0.3333** (33.3%)
- CER = 1 / 20 = **0.05** (5%) — only one character was wrong
 
---
 
## 2. Head-to-Head Comparison
 
| Dimension               | CER                                       | WER                                        |
|--------------------------|--------------------------------------------|--------------------------------------------|
| **Granularity**          | Character-level                            | Word-level                                 |
| **Sensitivity**          | Fine-grained; a single typo is one error   | Coarse; a single typo penalizes a full word|
| **Typical range**        | Usually lower than WER for the same output | Usually higher than CER                    |
| **Best for**             | Short strings, codes, numbers, table cells | Sentences, paragraphs, natural language    |
| **Tokenization needed?** | No (operates on raw characters)            | Yes (whitespace or language-specific split)|
| **Language agnostic?**   | Yes — works on any script                  | Depends on tokenizer quality               |
| **Interpretability**     | "X errors per 1000 characters"             | "X errors per 100 words"                   |
| **Can exceed 1.0?**      | Yes (more insertions than reference chars) | Yes (same reason at the word level)        |
 
### When to Use Which
 
| Scenario                                      | Recommended Metric |
|-----------------------------------------------|---------------------|
| Table-cell OCR (short, numeric, mixed)        | **CER**             |
| Full-page document OCR                        | WER (+ CER)         |
| Handwriting recognition                       | CER                 |
| Speech-to-text (ASR)                          | **WER**             |
| License plate / serial number recognition     | CER                 |
| Machine translation quality (post-edit)       | WER or TER          |
| Multi-lingual OCR (CJK, Arabic, etc.)        | **CER**             |
# SST-2 Sentiment Analysis: CE Baseline vs Supervised Contrastive Learning

## Project Overview

Build a sentiment analysis system on the SST-2 dataset comparing two approaches:
1. **Baseline:** RoBERTa-base fine-tuned with standard Cross-Entropy (CE) loss
2. **Novel:** RoBERTa-base fine-tuned with CE + Supervised Contrastive Learning (SCL) loss

Based on the paper: Gunel et al., "Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning" (ICLR 2021) — https://arxiv.org/abs/2011.01403

---

## Environment & Dependencies

```
torch
transformers
datasets
scikit-learn
matplotlib
numpy
pandas
```

GPU expected: NVIDIA RTX (8GB VRAM). Use `fp16` mixed precision if needed. Use batch size 32 (or 16 with gradient accumulation of 2).

---

## Part 1: Data Loading

- Load SST-2 from HuggingFace: `load_dataset("stanfordnlp/sst2")`
- Use the `train` split (67K examples) for training
- Use the `validation` split (872 examples) as the test set (the official test labels are hidden)
- Tokenize with `AutoTokenizer.from_pretrained("roberta-base")`, max_length=128, padding=True, truncation=True
- **No additional preprocessing** — the dataset is pre-cleaned. The Penn Treebank tokenization artifacts (e.g., "do n't", "it 's") should be left as-is since the RoBERTa tokenizer handles them natively.

---

## Part 2: Baseline Model (CE only)

### Architecture
- `AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)`
- Standard classification head on top of the `[CLS]` token

### Training
- Optimizer: AdamW, lr=2e-5, weight_decay=0.01
- Scheduler: linear warmup (6% of total steps) then linear decay
- Epochs: 3
- Batch size: 32 (or 16 with gradient_accumulation_steps=2)
- Loss: standard CrossEntropyLoss (built into the model)

### Evaluation
- Report: accuracy, F1 (macro), precision, recall on the validation set
- Save predictions for later comparison

---

## Part 3: SCL Model (CE + Supervised Contrastive Loss)

### Architecture
- Same RoBERTa-base backbone
- Add a **projection head**: a 2-layer MLP (hidden_dim → 128 → 128) with ReLU activation, applied on the `[CLS]` embedding. This projects embeddings into a lower-dimensional space where contrastive learning operates.
- The classification head remains the same as the baseline (applied on the raw `[CLS]` embedding, NOT on the projected embedding)

### SCL Loss Implementation

For a batch of N examples with embeddings h_1...h_N and labels y_1...y_N:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: L2-normalized projected embeddings, shape (batch_size, proj_dim)
            labels: class labels, shape (batch_size,)
        Returns:
            scalar loss
        """
        # Cosine similarity matrix (since embeddings are L2-normalized, dot product = cosine sim)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Mask: 1 where labels match (positive pairs), 0 otherwise
        labels = labels.unsqueeze(0)  # (1, B)
        mask = torch.eq(labels, labels.T).float()  # (B, B)

        # Remove self-similarities from both the mask and the logits
        batch_size = embeddings.shape[0]
        self_mask = torch.eye(batch_size, device=embeddings.device).bool()
        mask = mask.masked_fill(self_mask, 0)  # no self-positive
        sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))  # no self in denominator

        # For numerical stability
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # Log-softmax over the similarity matrix (denominator = all non-self examples)
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(self_mask, 0)  # zero out self
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-probability over positive pairs
        positive_count = mask.sum(dim=1)  # number of positives per example
        mean_log_prob = (mask * log_prob).sum(dim=1) / (positive_count + 1e-8)

        # Only include examples that have at least one positive pair
        valid = positive_count > 0
        loss = -mean_log_prob[valid].mean()

        return loss
```

### Combined Training Loop

```python
# Pseudocode for each training step:

# 1. Forward pass through RoBERTa
outputs = model.roberta(input_ids, attention_mask=attention_mask)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

# 2. CE loss: pass [CLS] through classification head
logits = classification_head(cls_embedding)
loss_ce = F.cross_entropy(logits, labels)

# 3. SCL loss: pass [CLS] through projection head, L2-normalize, compute contrastive loss
projected = projection_head(cls_embedding)
projected = F.normalize(projected, p=2, dim=1)
loss_scl = scl_criterion(projected, labels)

# 4. Combined loss
loss = loss_ce + lambda_weight * loss_scl
loss.backward()
```

### Hyperparameters
- Same optimizer/scheduler as baseline (AdamW, lr=2e-5)
- Epochs: 3-4
- **temperature τ**: default 0.3 (also run ablations with 0.1 and 0.5)
- **lambda_weight λ**: default 0.5 (also run ablations with 0.1 and 1.0)
- projection_dim: 128

---

## Part 4: Experiments to Run

### Experiment 1: Full-data comparison
- Train CE-only and CE+SCL on the full training set
- Report accuracy and F1 on validation
- Expected: SCL gives a small improvement (~0.5-1%)

### Experiment 2: Few-shot learning (THIS IS THE KEY EXPERIMENT)
- Subsample the training set to **1%, 5%, 10%, 50%** of the data
- For each subsample size, train BOTH CE-only and CE+SCL
- Use stratified sampling to maintain class balance
- Run each configuration 3 times with different random seeds and report mean ± std
- Expected: SCL's advantage grows as data shrinks. At 1% (~670 examples), the gap should be significant.

### Experiment 3: Noise robustness
- Take the full training set
- Randomly flip **5%, 10%, 15%** of the labels (flip 0→1 and 1→0)
- Train both CE-only and CE+SCL on each noisy version
- Report accuracy on the (clean) validation set
- Expected: SCL degrades more gracefully than CE under label noise

### Experiment 4: t-SNE visualization
- After training, extract [CLS] embeddings for all validation examples from both the CE-only model and the CE+SCL model
- Apply t-SNE (sklearn, perplexity=30) to reduce to 2D
- Plot side-by-side, colored by true label (red=negative, blue=positive)
- Expected: CE+SCL shows tighter, more separated clusters

---

## Part 5: Outputs & Visualizations

Generate the following plots and tables:

### Tables
1. **Main results table**: Accuracy and F1 for CE vs CE+SCL on full data
2. **Few-shot results table**: Accuracy (mean±std over 3 seeds) for each data fraction (1%, 5%, 10%, 50%, 100%) × (CE, CE+SCL)
3. **Noise robustness table**: Accuracy for each noise level (0%, 5%, 10%, 15%) × (CE, CE+SCL)
4. **Hyperparameter ablation table**: Accuracy for temperature τ ∈ {0.1, 0.3, 0.5} and λ ∈ {0.1, 0.5, 1.0}

### Plots
1. **Few-shot line chart**: x-axis = % of training data, y-axis = accuracy, two lines (CE and CE+SCL) with error bars
2. **Noise robustness line chart**: x-axis = noise level %, y-axis = accuracy, two lines
3. **t-SNE scatter plots**: side-by-side, CE-only vs CE+SCL, colored by label
4. **Training loss curves**: CE loss and SCL loss over training steps
5. **Confusion matrices**: for both models on full-data validation

Save all plots as PNG files. Save all result tables as CSV files.

---

## File Structure

```
project/
├── data/                   # cached datasets (auto by HuggingFace)
├── models/                 # saved model checkpoints
│   ├── ce_baseline/
│   └── scl_model/
├── results/
│   ├── tables/             # CSV result tables
│   └── plots/              # PNG visualizations
├── src/
│   ├── dataset.py          # data loading, tokenization, subsampling
│   ├── model.py            # model class with projection head + SCL loss
│   ├── train_baseline.py   # CE-only training script
│   ├── train_scl.py        # CE+SCL training script
│   ├── evaluate.py         # evaluation metrics, confusion matrix
│   ├── visualize.py        # t-SNE plots, training curves, result charts
│   └── experiments.py      # orchestrates few-shot and noise experiments
├── run_all.py              # master script to run everything
├── requirements.txt
└── README.md
```

---

## Important Implementation Notes

1. **Do NOT use HuggingFace Trainer for the SCL model** — you need a custom training loop to compute both losses. You CAN use Trainer for the CE baseline.
2. **Batch size matters for SCL** — you need enough same-class examples per batch for meaningful contrastive pairs. Batch size 32 with SST-2's ~50/50 balance gives ~16 positives per class, which is sufficient.
3. **L2-normalize embeddings before SCL** — this is critical. Without normalization the loss is unstable.
4. **The projection head is only used during training** — at inference/evaluation time, classification uses the raw [CLS] embedding through the classification head. The projection head is discarded.
5. **Freeze nothing** — fine-tune the entire RoBERTa backbone. No frozen layers.
6. **Use the validation split as test** — the official SST-2 test labels are hidden (-1). Everyone uses the validation split for reporting results.
7. **Set seeds** for reproducibility: `torch.manual_seed(seed)`, `random.seed(seed)`, `np.random.seed(seed)`.
8. **Mixed precision (fp16)** is recommended if VRAM is tight on the GPU.

# CNN Age Inference

This repository contains code for training and evaluating convolutional neural networks (CNNs) for age inference from hand dorsal img data.
---


### 1. Create a virtual environment

```bash
python3.10 -m venv .venv
.venv\Scripts\activate
```

---

### 2. Install dependencies
Install all base dependencies:

```bash
pip install -r requirements.txt
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PointNet (Windows-compatible, no CUDA ops)

Clone the simple PointNet repo:

```bash
cd ..
git clone https://github.com/fxia22/pointnet.pytorch.git
cd pointnet.pytorch
pip install -r requirements.txt
```

Install the library:

```bash
pip install -e .
```

---

### 3. Model architecture

The default model is `EfficientNetAgeRegressor` (`models/efficientnet_age.py`), a thin wrapper over torchvision EfficientNet-B{0-7}.

- Backbone: pick the variant with `--model {b0..b7}`; the corresponding canonical image size is enforced via `--img-size`.
- Initialization: loads the best available torchvision weights (DEFAULT enum when present, otherwise ImageNet-pretrained).
- Head: replaces the EfficientNet classifier with a 2-unit linear layer that jointly regresses the Gaussian mean and log-variance of age.
- Loss: `train_age.py` optimizes the negative log-likelihood under that Gaussian (`gaussian_nll_loss`) so the network learns both central tendency and epistemic spread.
- Outputs: the forward pass returns `(mu, log_var)`, which downstream utilities convert into adult probabilities via Gaussian tail integration.

---

### 4. Binary age-gate evaluation

After training all models, treat the age-inference head as a binary age gate with an application-defined threshold (default 18 years). The network performs probabilistic age regression and outputs the parameters of a Gaussian age posterior (mu, sigma^2); integrate the Gaussian tail above the threshold to obtain the probability of the user being an adult, noted as p_adult( tau ).

#### Case 1 - Keep children away from adult content
Use this configuration when minors must not access adult-only experiences.

Decision rule (threshold at 18):
- If p_adult >= tau -> True -> user admitted
- If p_adult < tau -> False -> user rejected

Report the following metrics:
- Minor Incorrectly Admitted (FPR): percentage of minors incorrectly admitted. <-- undesired risk.
- Adult Incorrectly Rejected (FNR): percentage of adults wrongly denied access.
- Adult Access Rate (TPR): percentage of adults correctly admitted. <-- desired usability.
- Minor Rejection Rate (TNR): percentage of minors correctly denied access.

#### Case 2 - Keep adults away from children
Use this configuration on child-specific platforms where only minors should be admitted.

Decision rule (threshold at 18):
- If p_adult < tau -> True -> user admitted
- If p_adult >= tau -> False -> user rejected

Report the following metrics:
- Adult Incorrectly Admitted (FPR): percentage of adults incorrectly admitted. <-- undesired risk.
- Minor Incorrectly Rejected (FNR): percentage of minors wrongly denied access.
- Minor Access Rate (TPR): percentage of minors correctly admitted. <-- desired usability.
- Adult Rejection Rate (TNR): percentage of adults correctly denied access.

#### ROC analysis
For both cases, produce a ROC curve with:
- x-axis: FPR (undesired risk).
- y-axis: TPR (desired usability).
- Each point reflects one threshold value tau (i.e., the confidence cutoff).
- Always report the Area Under Curve (AUC) next to the plot to summarize overall trade-offs.


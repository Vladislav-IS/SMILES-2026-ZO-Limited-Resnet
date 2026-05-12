# Zero-Order Fine-Tuning of ResNet18 on CIFAR100

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run evaluation

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 256 \
    --output results.json
```

---

## Experiments Description

### 1. `head_init.py`

I replaced `nn.kaiming_uniform_` with `nn.normal_` ($mean=0$, $std=10^-3$) for `layer.weight`. `layer.bias` remained initialised with `nn.init.zeros_`.

### 2. `train_data.py`

I created a class-balanced dataset as a subset of CIFAR-100. Class balancing was implemented by manually selecting samples class-by-class. The total number of samples is 8192. Dataset is constructed with its own `SEED`.

### 3. `augmentation.py`
 
The file remained untouched.

### 4. `zo_optimizer.py`

I replaced the simple 2-point estimator with SPSA using `gaussian` noise sampling and 300 independent directions to reduce variance of gradient estimation. I used Adam optimizer with fixed $lr=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$. Optimizing layers are `fc.weight`, `fc.bias`. `batch_size = 32`, and `n_batches` = 256.

### 5. Final results

You can also find final results in `results.json` file.

| # | Checkpoint | Accuracy |
|---|-----------|-----------------|
| 1 | **Baseline (ImageNet head)** | 0.37% |
| 2 | **Initialized head (no fine-tuning)** | 0.89% |
| 3 | **Fine-tuned (ZO)** | 13.08% |

### 6. Failed experiments

During experiments, I tried:

- using additional augmentations. Augmentations worsened the final result. Apparently, the optimizer converged less well on augmented data;

- using repeated samples in train dataset. The result worsened. This effect can be explained by the reduced intra‑class diversity caused by using only a small set of unique images repeatedly;

- removing the manual shuffling (`np.random.shuffle` in `train_data.py`) of indices during the creation of class-balanced dataset and using only built-in DataLoader shuffling. It led to a slight deterioration in quality (from 13.08% to 12.44%). This fact can be explained by the high sensitivity of the optimizer to data structure.

- using other initialization strategies with a larger spread of values (for example, `nn.xavier_uniform_`, `nn.orthogonal` and `nn.normal_` with larger $std$). It seems that a larger spread worsens the convergence of the optimizer;

- using gradient clipping. Clipping neither worsened nor improved the final result;

- changing optimizer learning rate. Increasing the learning rate in a narrow range caused the optimizer to diverge, while decreasing it caused it to stop converging;

- changing `batch_size` and `n_batches` values. The current parameters provided the best result;

- changing the list of fine‑tuned layers. Increasing the list (and therefore increasing the trainable parameters) led to worse convergence.

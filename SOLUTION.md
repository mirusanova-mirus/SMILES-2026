# Solution

## Result

```
Baseline (ImageNet head)       0.37%
Initialized head (no FT)      49.31%
Fine-tuned (ZO)               49.49%
```

Budget: 64 steps * batch 128 = 8192 samples.

## How to reproduce

```bash
pip install -r requirements.txt
python validate.py --data_dir ./data --batch_size 128 --n_batches 64 --output results.json
```

Seed 42 (from `validate.py`), everything is deterministic. Feature extraction for NCM
is done on CPU on purpose — on MPS the ResNet's BatchNorm gives inf/NaN features for
me, which makes the head NaN. CPU forward is around 30 seconds for 2000 images, not a
big deal.

CIFAR100 and ResNet18 weights are downloaded automatically on first run.

## What I ended up doing

I spent quite a bit of time trying to figure out what can actually be done with ZO at
8192 samples — short answer, not much. Most of the result comes from the head init,
and ZO adds a little on top because in 51200-dim head space any SPSA gradient estimate
is mostly noise.

### head_init.py

Cosine classifier via class prototypes (NCM):
1. Forward the pretrained ResNet18 (with fc removed) on 20 samples per class.
2. Compute per-class mean features μ_c.
3. Set W_c = T * μ_c / ||μ_c||, b = 0.

T=2 is the temperature. argmax does not depend on T (so init accuracy is the same for
any T), but for larger T the softmax saturates and CE gradient becomes basically zero,
which kills ZO. T=2 keeps softmax unsaturated.

If anything fails (no data, NaN features) — fallback to orthogonal init.

### zo_optimizer.py

Only **fc.bias** is tuned (100 parameters). This took me a while to settle on, see
the failed attempts section. Short version: on full fc.weight (51200 dims) SPSA does
not work because of noise, and bias is exactly the one parameter NCM explicitly sets
to zero and leaves to optimize.

Inside `.step()`:
- Simultaneous SPSA with Rademacher directions (±1, no normalisation).
- q=32 queries per gradient estimate, central difference.
- K=2 update attempts per batch, each with a fresh estimate.
- Adam (lr=5e-3, β=(0.9, 0.999)) — but **state is reset between batches**.
- Trust-region accept/reject: after each Adam step we re-evaluate loss on the same
  batch, if it got worse we revert weights and Adam state.

The Adam reset is needed because otherwise momentum carries noise from previous
batches and the weights drift in a random direction. The accept/reject is needed
because otherwise we accept "improvements" that only help this specific batch.

All forwards inside step() are free in terms of budget — only the sample count is
counted (.step() is called exactly n_batches times).

### augmentation.py

Just Resize + RandomHorizontalFlip + Normalize. No AutoAugment / ColorJitter / Erasing.
With 64 batches, strong augmentation only adds loss variance between batches, and ZO
only sees more noise from it.

### train_data.py

`drop_last=True`, otherwise the last batch can be smaller and break step consistency.

## What contributed

- NCM init: ~22% (default Kaiming) → 49.31%. This is **the main thing**.
- ZO with bias-only + accept/reject + Adam reset: +0.18% on top.

I know the ratio is weird, but it honestly reflects the reality of the task. When you
already have a strong head, on 51k-dim space with 8192 samples and SPSA — there's
almost nothing to improve. ZO under these constraints is at best a small per-class
threshold calibration.

## What I tried that didn't work

### Initialization

**Logistic regression on extracted features.** Extracted features through backbone,
fit multinomial LR via gradient descent (analytical softmax-CE gradient, no backward
through the model). With n_per_class=100, n_iter=400, lr=0.5, WD=1e-3 got init=35%.
Severe overfit on 10k samples. Could probably be fixed with regularisation /
cross-validation / LBFGS, but NCM on the same data gives 49% without any of that, so
I went with NCM.

**NCM with T=10.** Same 49.31% accuracy, but softmax is saturated → CE gradient near
zero → ZO can't do anything on top. So T=2.

### ZO optimizer

**Vanilla central difference with normalised direction (skeleton-style).** With
||u||=1 the estimator is biased by 1/D (E[<∇f,u>u] = ∇f/D on the unit sphere). So
the steps are essentially zero. Replaced with unnormalised Rademacher — unbiased.

**Full fc.weight + Adam, lr=3e-3, q=16.** Loss within a step decreases, but loss
across steps jumps randomly because of noise. Final accuracy 49.2-49.3%, doesn't
move. Per-coord SNR ~0.02 just doesn't work.

**fc.weight + fc.bias + BN affines from layer4.** Got worse. Perturbing BN
simultaneously shifts the feature distribution at the last block, and the head
doesn't know about it — the gradient estimate gets badly biased.

**K=4 inner Adam updates without accept/reject.** Heavy overfit on specific batches,
49.31% → 47.84%. Too much "optimization" per batch.

**K=4 with accept/reject but no Adam reset.** 47.93%. The accept/reject check alone
does not save you, because Adam momentum carries the noisy drift from previous
batches.

**K=4 + reset + accept/reject + bias-only.** 49.29%, still slightly worse than init.
On bias-only too you can overfit if you do too many steps per batch.

**K=2 + reset + accept/reject + bias-only.** Final config, 49.49%. This is the sweet
spot — enough attempts to regularly catch useful directions, not enough to accumulate
overfit.

### Augmentation

**AutoAugment(CIFAR10) + ColorJitter + RandomErasing.** With 64 batches, loss between
batches scattered between 1.5 and 2.5. Gradient signal drowned in noise.

**RandomCrop(padding=16) + flip.** A bit better but still noticeable noise.

In the end kept only flip — the safest augmentation (symmetric, doesn't shift the
feature distribution).

### Budget split

Tried different `n_batches × batch_size`:
- 32 × 32 = 1024 (half the budget unused) — 49.3%
- 64 × 64 = 4096 — 49.3%
- 64 × 128 = 8192 (full) — 49.49%
- 128 × 64 = 8192 — 49.3-49.4%

Larger batch gives less noisy loss per step, which matters for accept/reject — more
candidates pass the check. So 64 × 128.

## What I left out

With a bigger budget (or the ability to do multiple passes over 8192) I would:
1. Bump NCM to n_per_class=100-500 for more stable prototypes (init would go up by
   maybe ~0.5%).
2. After bias-only ZO try adding fc.weight with very small lr and accept/reject as a
   safety net. With 200+ batches Adam would have time to accumulate useful signal
   even on 51k-dim.

With the available 8192 samples — 49.49% is the ceiling I found.

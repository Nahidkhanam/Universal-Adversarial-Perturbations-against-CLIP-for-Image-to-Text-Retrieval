# Universal Adversarial Perturbations against CLIP for Image-to-Text Retrieval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m8fWfc18vOIRxmn26gmstP4hKJAxRdxN)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

A study of how a **single, shared adversarial perturbation** degrades image-to-text retrieval in a **frozen, pretrained CLIP** model. The perturbation is learned from 5,000 training images, bounded by **8/255** in the L∞ norm, and evaluated on **unseen images**, on the **whole Flickr30K dataset**, on **other CLIP models**, and under **resizing**.

> This is a simplified, educational implementation inspired by prior work on universal adversarial perturbations for vision-language models (Zhang et al., SIGIR 2024). It is **not** an exact reproduction, and its numbers are **not directly comparable** with published results.

---

## Highlights

- One perturbation δ shared across all images, with the CLIP parameters frozen.
- Held-out evaluation on the Karpathy test split (1,000 images, 5,000 captions), plus a full-gallery evaluation (31,783 images, about 158,900 captions).
- Three attacks compared under identical data, budget and evaluation: **baseline**, **strong**, **robust**.
- Controls: random-noise perturbation, half-budget perturbation, resize test, and an ablation study.
- Transfer tests from ViT-B/32 to ViT-B/16 and RN50.
- Memory-efficient chunked Recall@K, so the full image-caption similarity matrix is never stored.

## Results (single run, seed 0)

All values are percentages. The perturbation was learned on CLIP ViT-B/32.

### White-box: ViT-B/32 on 1,000 held-out test images

| Setting | R@1 | R@5 | R@10 |
|---|---|---|---|
| Clean images | 78.8 | 94.9 | 98.2 |
| Random ±ε sign noise (control) | 78.0 | 94.3 | 97.9 |
| Baseline UAP | 34.0 | 57.1 | 65.9 |
| **Strong UAP** | **27.1** | **47.8** | **56.0** |
| Robust UAP | 64.6 | 86.6 | 91.7 |

### Whole dataset: 31,783 query images vs about 158,900 captions (ViT-B/32)

| Setting | R@1 | R@5 | R@10 |
|---|---|---|---|
| Clean images | 40.63 | 64.77 | 73.73 |
| Baseline UAP | 7.06 | 16.57 | 22.72 |
| Strong UAP | 7.36 | 15.95 | 21.11 |
| Robust UAP | 24.87 | 45.32 | 54.62 |

Results for held-out queries only (excluding the 5,000 training images) match these closely.

### Transfer (R@1, perturbation learned on ViT-B/32)

| Model | Clean | Random noise | Baseline | Strong | Robust |
|---|---|---|---|---|---|
| ViT-B/16 | 82.2 | 83.1 | 80.5 | 79.2 | 77.8 |
| RN50 | 80.0 | 76.1 | 66.7 | 66.6 | 67.7 |

### Budget and resize robustness (ViT-B/32, R@1)

| Setting | Baseline | Strong | Robust |
|---|---|---|---|
| Full budget (ε = 8/255) | 34.0 | 27.1 | 64.6 |
| Half budget (0.5 × δ) | 62.7 | 54.2 | 75.6 |
| After resize (224 → 112 → 224) | 67.6 | 62.7 | 59.3 |

### Ablation (ViT-B/32, small budget: 2,000 training images, 6 epochs)

| Configuration | R@1 | R@5 | R@10 |
|---|---|---|---|
| Clean | 78.8 | 94.9 | 98.2 |
| A. Similarity loss (baseline) | 52.8 | 77.8 | 84.3 |
| B. Contrastive CE | 56.3 | 82.2 | 87.5 |
| C. B + rank margin | 45.0 | 70.8 | 79.5 |
| D. C + clean-embedding displacement (**Strong**) | 37.8 | 64.8 | 75.9 |
| E. D + low-frequency δ | 70.1 | 90.9 | 95.0 |
| F. D + augmentation | 71.1 | 88.9 | 93.9 |
| G. D + both (**Robust**) | 73.4 | 91.4 | 95.3 |

### Takeaways

1. The learned perturbation generalises to unseen images and is far stronger than random noise of the same size.
2. The strong attack beats the baseline on the 1,000-image test split and at half budget, but the two are **comparable on the whole dataset**.
3. The low-frequency, augmented (robust) attack is weaker on clean inputs but is the **strongest after resizing**.
4. **Transfer is limited.** ViT-B/16 drops by at most about 4 points, and about 4 of the 13-point RN50 drop comes from random noise alone.

## Method

**Perturbation.** A single δ ∈ ℝ³ˣ²²⁴ˣ²²⁴ is parameterised as `δ = ε · tanh(p)` with ε = 8/255, so ‖δ‖∞ ≤ ε always holds. It is added in [0, 1] pixel space, clipped, and **then** normalised with CLIP's mean and standard deviation.

**Attack variants** (every loss is *minimised*, so correct captions become less similar to the perturbed image):

| Variant | δ | Loss | Augmentation |
|---|---|---|---|
| `baseline` | full resolution | mean cosine similarity to the 5 correct captions | none |
| `strong` | full resolution | contrastive CE + 0.5 · rank margin + 5 · clean-embedding displacement | none |
| `robust` | low-frequency (56×56, bilinearly upsampled) | same as `strong` | random flip and resized crop |

- **Contrastive CE:** log-probability of the correct captions among all captions in the batch (temperature 100).
- **Rank margin:** pushes the best wrong caption above the best correct caption (margin 2 in logit units).
- **Displacement:** cosine similarity between the perturbed and clean image embeddings.

**Optimisation settings**

| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate 0.1 on p, cosine annealing |
| Batch size / epochs | 64 / 15 |
| Training images | 5,000 sampled from the Karpathy train split (seed 0) |
| Perturbation budget | ε = 8/255 (L∞) |
| Precision | fp32 |

**Evaluation.** Image-to-text Recall@1/5/10, where a query counts as a hit if any of its five captions is in the top K.

## Repository structure

```
uap-clip-image-retrieval/
├── notebook/
│   ├── UAP_CLIP_Flickr30K_v3.ipynb   # main notebook (this README)
│   └── UAP_CLIP_Retrieval.ipynb      # earlier 200-image prototype (see notes below)
├── results/
│   ├── results.json
│   ├── uap_summary.png
│   └── original_vs_adv.png
├── README.md
└── LICENSE
```

## Quick start (Google Colab)

1. Click the **Open in Colab** badge above.
2. Set the runtime to **GPU** (a free T4 is enough).
3. Run all cells. The notebook will:
   - download Flickr30K through `kagglehub` and the Karpathy split file,
   - load CLIP ViT-B/32 (training target) and ViT-B/16 and RN50 (transfer targets),
   - train the three attacks,
   - evaluate and save `results.json`, `uap_deltas.pt` and the figures.

**Runtime:** roughly 1 to 1.5 hours on a T4 with the ablation and full-gallery evaluation enabled. Set `RUN_ABLATION = False` or `EVAL_FULL = False` in the configuration cell to shorten it.

### Configuration

All settings are in one cell near the top of the notebook:

```python
EPS = 8 / 255
TRAIN_MODELS = ["ViT-B/32"]            # add "RN50" to train on an ensemble
TRANSFER_MODELS = ["ViT-B/16", "RN50"]
TRAIN_N = 5000
EPOCHS = 15
BATCH = 64
RUN_ABLATION = False
EVAL_FULL = True
```

### Running locally

```bash
pip install torch torchvision ftfy regex tqdm kagglehub pandas matplotlib
pip install git+https://github.com/openai/CLIP.git
```

A CUDA GPU is required. The Kaggle dataset download may ask for credentials; set `KAGGLE_USERNAME` and `KAGGLE_KEY` or place a `kaggle.json` in `~/.kaggle/`.

## Data

- **Images:** [Flickr30K](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) (31,783 images, 5 captions each).
- **Split and captions:** the Karpathy split (29,000 train, 1,014 val, 1,000 test) from `dataset_flickr30k.json`.

The datasets are not redistributed here. Please follow their original licences and terms of use.

## Limitations

- **Single run, seed 0.** A repeat of the baseline gave R@1 of 31.7% instead of 34.0%, so differences below about 3 percentage points should not be over-interpreted.
- One dataset (Flickr30K), one task (image-to-text retrieval) and one budget (8/255).
- Weak transfer, especially to ViT-B/16.
- No defence is evaluated.
- Not a reproduction of Zhang et al. (2024); the models, datasets and metrics differ, so the numbers are not comparable with the paper.

## Notes on the earlier prototype

`UAP_CLIP_Retrieval.ipynb` was an early 200-image experiment. It had three problems that this version fixes:

1. The loss was **maximised** instead of minimised, so the optimisation increased the similarity to the correct captions.
2. The perturbation was added **after** CLIP normalisation and then clamped to [0, 1], which distorted the images.
3. The perturbation was optimised and evaluated on the **same images**, so generalisation was not tested.

Its results should not be compared with the numbers above.

## Future work

- Repeat over multiple seeds and report mean and standard deviation.
- Train on an ensemble of models to improve transfer, and test additional models and datasets.
- Evaluate other budgets and attack types, such as text-side and multimodal attacks.
- Evaluate defences such as adversarial training and input purification.


## References

1. A. Radford et al., "Learning transferable visual models from natural language supervision," ICML, 2021.
2. S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, "Universal adversarial perturbations," CVPR, 2017.
3. P.-F. Zhang, Z. Huang, and G. Bai, "Universal adversarial perturbations for vision-language pre-trained models," SIGIR, 2024.
4. J. Zhang et al., "AnyAttack: Towards large-scale self-supervised adversarial attacks on vision-language models," CVPR, 2025.
5. P. Young, A. Lai, M. Hodosh, and J. Hockenmaier, "From image descriptions to visual denotations," TACL, 2014.
6. A. Karpathy and L. Fei-Fei, "Deep visual-semantic alignments for generating image descriptions," CVPR, 2015.

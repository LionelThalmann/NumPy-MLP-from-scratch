
# NumPy MLP: From-Scratch Implementation

<a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>

---

## 📚 Project Summary
This notebook implements and trains a **multi-layer perceptron (MLP)** in **pure Python + NumPy** - no PyTorch, no TensorFlow.

| Dataset | Task | Instances | Features |
|---------|------|-----------|----------|
| **Credit-Card Default of Clients** | Binary classification | 30 000 | 23 |
| **Amazon Reviews** <sub>(Kaggle competition `184-702-tu-ml-2025-s-reviews`)</sub> | 50-class text classification (bag-of-words) | 750 | 10 000 |

The aim was to learn back-propagation by building every component from scratch while exploring depth, width, activations and learning-rate settings.

---

## 📂 Repo Structure
```

├── MLP\_from-scratch.ipynb    # data → model → training → evaluation (all in one)
├── datasets/                 # auto-created; Kaggle zip + extracted CSV land here
└── README.md

````

---

## 🛠 Implementation Highlights

| Component       | Details                                                   |
|-----------------|-----------------------------------------------------------|
| **Weight init** | He initialization `np.random.randn(*shape) * √(2 / fan_in)` |
| **Activations** | `relu`, `tanh`, `sigmoid`                                 |
| **Output**      | `softmax` + cross-entropy                                 |
| **Optimizer**   | Vanilla SGD (`W -= α·dW`, `b -= α·db`)                    |
| **Grid search** | Hidden {10, 64, 500}, layers {1–3}, α {0.1, 0.05, 0.01}, iters {50, 100, 500} |

---

## 🏋️ How to Reproduce
```bash
git clone <YOUR-REMOTE-URL>/numpy-mlp-from-scratch.git
cd numpy-mlp-from-scratch

python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn tqdm ucimlrepo kaggle jupyterlab

# Configure Kaggle credentials once
export KAGGLE_USERNAME=<your_kaggle_username>
export KAGGLE_KEY=<your_kaggle_key>

jupyter lab MLP_from-scratch.ipynb
````


## 📊 Results

### Credit-Card (top configurations)

| Hidden Neurons | Layers | Activation | Parameters | RAM Bytes |   Accuracy |
| -------------: | :----: | :--------: | ---------: | --------: | ---------: |
|             64 |    2   |    ReLU    |      5 761 |    46 088 | **82.1 %** |
|            500 |    1   |    ReLU    |     12 501 |   100 008 |     81.9 % |
|             10 |    2   |    ReLU    |        361 |     2 888 |     81.8 % |
|             64 |    2   |   Sigmoid  |      5 761 |    46 088 |     77.8 % |
|             10 |    1   |   Sigmoid  |        251 |     2 008 |     77.8 % |

### Amazon Reviews (top configurations)

| Hidden Neurons | Layers | Activation | Parameters | RAM Bytes |   Accuracy |
| -------------: | :----: | :--------: | ---------: | --------: | ---------: |
|             64 |    1   |   Sigmoid  |    643 314 | 5 146 512 | **60.6 %** |
|             64 |    1   |   Sigmoid  |    643 314 | 5 146 512 |     57.3 % |
|             64 |    1   |    Tanh    |    643 314 | 5 146 512 |     52.6 % |
|             32 |    1   |   Sigmoid  |    321 682 | 2 573 456 |     48.6 % |
|             16 |    1   |    ReLU    |    160 866 | 1 286 928 |     22.0 % |

*(See notebook for the complete grid and RAM calculations.)*

---

## 💡 Key Take-aways

* **Hand-coded back-prop** clarifies gradient flow and shows how vectorised NumPy trumps Python loops.
* On Credit-Card, a **deep/wide tanh-like variant** (ReLU/tanh) reached \~82 % accuracy; gains level off past 500 neurons.
* For sparse Amazon BoW, a **shallow sigmoid** model performed best - depth was less important than activation & capacity.
* Even with vanilla SGD and \~260 k parameters the model hits > 80 % on Credit-Card - solid for a zero-framework baseline.

---

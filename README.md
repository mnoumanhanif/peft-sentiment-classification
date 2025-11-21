# 🧠 **Parameter-Efficient Fine-Tuning for Sentiment Classification**

*A Comprehensive Ablation Study Across LoRA, QLoRA, IA3, and Full Fine-Tuning*

---

## 📌 **Overview**

This repository contains the official implementation for the study:

***“Parameter-Efficient Fine-Tuning for Sentiment Classification: A Comprehensive Ablation Study Across Three Architectures”***

The work evaluates **Full Fine-Tuning** vs. **three state-of-the-art PEFT methods**:

* **LoRA**
* **QLoRA (4-bit Quantized LoRA)**
* **IA3 (Infused Adapter by Inhibiting Activations)**

across **three transformer architectures**:

* **DistilBERT (67M)**
* **RoBERTa-Base (125M)**
* **DeBERTa-v3 (184M)**

for the **IMDb Sentiment Classification** task.

The notebook performs a **42-run experimental matrix** (3 architectures × 7 configurations × 2 seeds), reproducing the full analysis and tables presented in the report.

📄 *The full PDF report is available in this repository:*
➡️ **`Gen_AI Assignment 03 (24K-8001) - IEEE.pdf`** 

---

## 🏗️ **Repository Contents**

| File                                                                   | Description                                                                                                                |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Parameter-Efficient Fine-Tuning for Sentiment Classification.ipynb** | Main Jupyter notebook containing dataset processing, PEFT configuration, training, metrics collection, and visualizations. |
| **Gen_AI Assignment 03 (24K-8001) - IEEE.pdf**                         | Full research report with methodology, results, tables, and analysis.                                                      |
| **requirements.txt**                                                   | Python dependencies needed to run the notebook.                                                                            |

This repository intentionally remains lightweight and notebook-centric for ease of use and reproducibility.

---

## 🎯 **Research Motivation**

Fine-tuning large transformers is expensive:

* High VRAM
* High compute
* Slow training

**Parameter-Efficient Fine-Tuning (PEFT)** reduces these costs by updating only a tiny subset of parameters while freezing the base model.

This study evaluates how PEFT methods behave across model sizes and architectures, quantifies the trade-offs, and determines **optimal configurations for resource-constrained environments**.

---

## 🔬 **Methods Evaluated**

### **1️⃣ Full Fine-Tuning (Baseline)**

Updates **all** model parameters.
Highest accuracy, highest VRAM, slowest training.

### **2️⃣ LoRA**

Injects rank-decomposed matrices (B,A) into linear layers.

* Very efficient (≤ 1–3M trainable params)
* Sensitive to placement (Attention vs FFN layers)

### **3️⃣ QLoRA**

LoRA with **4-bit NF4 quantization** of base weights.

* Reduces 7.2GB → 1.5GB VRAM (DeBERTa)
* Slightly slower due to dequantization overhead
* Best Pareto frontier performance

### **4️⃣ IA3**

Scales activations via learned vectors.

* Smallest parameter footprint (0.03%)
* Struggled with DeBERTa-v3 convergence

---

## 📚 **Dataset**

IMDb Sentiment Classification

* 25,000 training samples
* 25,000 test samples
* Tokenization length = **256**
* Binary classification: *positive / negative*

Loaded using `datasets.load_dataset("imdb")`.

---

## 🧪 **Experimental Design (42 Runs)**

Each experiment was executed for:

* **2 epochs**
* **Effective batch size = 16**
* **Learning rates = 1e-5 and 2e-5**
* **Seeds = 42, 43** (metrics averaged as μ)

Hardware used:

* **NVIDIA RTX A4000 (16GB VRAM)**
* **Intel i7-13700**
* **32 GB RAM**

Metrics reported:

* Accuracy
* VRAM peak usage
* Training time
* Trainable parameters

All metrics aggregated inside the notebook.

---

## 📊 **Key Results (from PDF report)**

### ✔ **QLoRA is the best overall PEFT method**

* Achieved **97–99%** of Full FT accuracy
* Reduced VRAM by **84–90%**
* Enabled training of **DeBERTa-v3 at just 1.52 GB VRAM**

### ✔ **FFN layers are critical**

PEFT adapters placed in:

* **Attention + FFN** → large accuracy gains
* FFN placement outperformed simply increasing LoRA rank

### ✔ **Learning Rate Sensitivity**

Increasing LR to **2×10⁻⁵** improved all PEFT methods.
QLoRA performance improved from **93.15% → 93.72%** on DeBERTa-v3.

### ✔ **IA3 was unstable for larger models**

Only **55% accuracy** on DeBERTa-v3 at LR=1e-5.

### ✔ **Full FT remains the performance ceiling**

But with the highest compute and VRAM cost.

---

## 🧩 **Notebook Features**

The notebook contains:

### 🔹 Dataset preprocessing

### 🔹 Model loading for 3 architectures

### 🔹 LoRA, QLoRA, IA3 PEFT configuration

### 🔹 Training loop with metrics extraction

### 🔹 VRAM monitoring

### 🔹 Aggregation of 42 experimental runs

### 🔹 Plot generation:

* Pareto Frontier (Accuracy vs VRAM)
* LoRA Rank Sensitivity
* Model Performance Comparison

All results match the tables and figures from the PDF .

---

## 🚀 **How to Run the Notebook**

### 1. Clone the repository

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open Jupyter

```bash
jupyter notebook
```

### 4. Run the notebook

Open:

**`Parameter-Efficient Fine-Tuning for Sentiment Classification.ipynb`**

Run all cells.

---

## 📦 **Requirements**

Suggested `requirements.txt`:

```
transformers
datasets
peft
accelerate
bitsandbytes
scikit-learn
pandas
numpy
matplotlib
seaborn
torch
```

---

## 📝 **Citation**

If you use this work, please cite the student research project:

```
M. N. Hanif. Parameter-Efficient Fine-Tuning for Sentiment Classification:
A Comprehensive Ablation Study Across Three Architectures. 2025.
```

---

## 📄 **License**

Released under the **MIT License** — free to use, modify, and distribute.

---

## 🙌 **Acknowledgements**

This work uses:

* HuggingFace Transformers
* PEFT Library
* IMDb Dataset
* NVIDIA CUDA Tooling

Special thanks to the FAST-NUCES Data Science Department.

Just say: **“Generate the README.md file as Markdown”**.

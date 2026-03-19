# MALTO Hackathon

This repository contains my solution and inference results for the **MALTO Recruitment Hackathon** (Politecnico di Torino). The approach involves fine-tuning the **Qwen 2.5** large language model using **QLoRA** (Quantized Low-Rank Adaptation) for the multi-class text classification task.

##  Project Overview

The goal of this project was to leverage a modern Large Language Model (Qwen 2.5) to accurately classify text into distinct categories. By utilizing QLoRA, the model was efficiently fine-tuned on an A100 GPU to achieve high accuracy without requiring full-parameter tuning.

**Key Highlights:**
* **Competition:** [MALTO Recruitment Hackathon](https://www.kaggle.com/competitions/malto-recruitment-hackathon)
* **Model:** Qwen 2.5
* **Technique:** QLoRA (Parameter-Efficient Fine-Tuning)
* **Task:** Multi-class classification
* **Hardware:** NVIDIA A100 GPU

### Performance Metrics
Based on the evaluation during training, the model achieved the following results over 2 epochs:

| Epoch | Training Loss | Validation Loss | Macro F1 |
|-------|---------------|-----------------|----------|
| 1     | 0.6923        | 0.1514          | 0.8927   |
| 2     | 0.3863        | 0.0232          | **0.9787** |

---

##  Repository Structure

```text
├── data/                     
├── notebooks/                 
│   └── Qwen_2_5.ipynb         
├── results/                   
│   ├── qwen_raw_probs.npy     
│   └── submission_qwen_qlora.csv 
├── .gitignore                 
├── requirements.txt           
└── README.md                  
```

---

##  Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. It is highly recommended to use a virtual environment or Conda environment. You will also need a CUDA-compatible GPU for training/inference.

### Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Hesam-AHT/MALTO-Hackathon
cd MALTO-Hackathon
pip install -r requirements.txt
```

*Note: Required packages include `torch`, `transformers`, `peft`, `bitsandbytes`, `numpy`, `pandas`, and `scikit-learn`.*

### Data Preparation
Place the Kaggle training and testing data inside the `data/` directory. Update the file paths in the `notebooks/Qwen_2_5.ipynb` notebook to point to your local dataset.


##  Usage

**1. Training the Model**
Open `notebooks/Qwen_2_5.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab. Run the cells sequentially to load the dataset, initialize the Qwen 2.5 model with QLoRA configurations, and start the training loop.

**2. Inference and Submission**
The final cells of the notebook handle predictions on the test set. Upon completion, the script outputs two files into the `results/` directory:
* `qwen_raw_probs.npy`: A NumPy array containing the raw logits/probabilities across all classes.
* `submission_qwen_qlora.csv`: A two-column CSV (`ID`, `LABEL`) ready for final scoring/submission on Kaggle.


## 👤 Author
* **Name:** Amirhesam Torkashvand
* **Matricola:** S336838
* **Kaggle Username:** amirhesamaht

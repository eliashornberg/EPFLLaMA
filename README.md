# EPFLLaMA: A Lightweight LLM Finetuned on EPFL Curriculum
<p align="center">
  <img width="753" alt="EPFLLama" src="https://github.com/user-attachments/assets/4d681a49-d658-46d5-afdc-f8b2504f807c">
</p>



## Project Overview

EPFLLaMA is a project that enhances the TinyLlama model through Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), utilizing datasets from student annotations and Stack Exchange. The project aims to create an AI tutor specialized in EPFL course content, with a particular focus on Multiple-Choice Question Answering (MCQA) tasks.

### Key Features

- Specialized for EPFL curriculum content
- Enhanced performance on MCQA tasks
- Utilizes advanced techniques: SFT, DPO, and Chain-of-Thought prompting
- Incorporates quantization for reduced memory footprint

## Project Structure
```
EPFLLAMA
│
├── README.md
│
├── data
│   ├── all_datasets
│   │   ├── MCQA_DPO.jsonl
│   │   ├── MCQA_unique_data.jsonl
│   │   ├── merged_DPO_test.jsonl
│   │   ├── merged_DPO_train.jsonl
│   │   └── sft_2000.jsonl
│   │
│   ├── annotated
│   │   ├── 383057.json
│   │   └── 384928.json
│   │
│   └── annotation_scripts
│       ├── DPO_Annotation.py
│       └── MCQA_Annotation.py
│
├── model
│   ├── dataset_example
│   │   ├── dpo_preference_example.jsonl
│   │   ├── mcqa_example.jsonl
│   │   └── MCQA_sft_test.jsonl
│   │
│   ├── models
│   │   ├── model_base.py
│   │   └── model_dpo.py
│   │
│   ├── Create_Loss_Plots.ipynb
│   ├── data_processing.ipynb
│   ├── Evaluate.ipynb
│   ├── evaluator.py
│   ├── main_config.yaml
│   ├── requirements.txt
│   ├── Training.ipynb
│   └── utils.py
│
├── pdfs
│   ├── litterature_reviews
│   │   ├── 383057.pdf
│   │   └── 384928.pdf
│   │
│   ├── progress_report
│   │   ├── ab-eh-me.pdf
│   │
│   ├── project_proposal
│   │   ├── ab_eh_me.pdf
│   │
│   └── project_report
│       ├── ab-eh-me.pdf

```
## Data Collection and Preparation

The project utilizes various data sources:

1. Student-annotated data from EPFL curricula
2. Stack Exchange datasets (Data Science, Computer Science, Physics, Mathematics)
3. GPT-generated preference pairs

Data collection scripts can be found in `model/models/data_processing.ipynb` and annotation scripts in `data/annotation_scripts`.

## Model Architecture

EPFLLaMA is based on the TinyLlama architecture, a compact and efficient language model with 1.1 billion parameters. It incorporates:

- 22 layers with 32 attention heads each
- Grouped-query attention mechanism
- RoPE (Rotary Positional Embedding)
- SwiGLU activation function

## Training Process

The training process involves two main phases:

1. **Supervised Fine-Tuning (SFT)**: Using the SFTTrainer from the trl library.
2. **Direct Preference Optimization (DPO)**: Implementing the DPO loss function to align the model with human preferences.

Additionally, the project explores:

- Parameter-Efficient Fine-Tuning (PEFT)
- Low-Rank Adaptation (LoRA)
- Quantization techniques

## Model Improvements

The project implements one main improvement:

**Quantization**: Reduces the model size while maintaining performance, using techniques like LLM.int8().

## Results and Evaluation

The EPFLLaMA model demonstrates:

- Improved performance on MCQA tasks compared to baselines
- Robust performance across various technical subjects
- Effective adaptation for educational purposes

Detailed results and analysis can be found in the project report (`pdfs/project_report/ab-eh-me.pdf`).

## Getting Started

To use or contribute to this project:

1. Clone the repository
2. Install dependencies: `pip install -r model/models/requirements.txt`
3. Explore the Jupyter notebooks in `model/models/` for training and evaluation

To use our pre-trained models:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model
model = AutoModelForCausalLM.from_pretrained("Ali7538/EPFLLaMA")
tokenizer = AutoTokenizer.from_pretrained("Ali7538/EPFLLaMA")

# For MCQA tasks
mcqa_model = AutoModelForCausalLM.from_pretrained("Ali7538/EPFLLaMA_MCQA")
mcqa_tokenizer = AutoTokenizer.from_pretrained("Ali7538/EPFLLaMA_MCQA")

# For faster inference with the quantized model
quantized_model = AutoModelForCausalLM.from_pretrained("Ali7538/EPFLLaMA_MCQA_Quantized")
quantized_tokenizer = AutoTokenizer.from_pretrained("Ali7538/EPFLLaMA_MCQA_Quantized")
```

## Contributors

- Ali Bakly
- Elias Ulf Hörnberg
- Othmane Sqalli Houssaini

## Acknowledgments

This project was developed as part of the CS-552 course at EPFL. Special thanks to the course staff and the NLP lab for providing resources and guidance.

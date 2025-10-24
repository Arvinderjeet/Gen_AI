md
# Gen_AI

This project encompasses code related to generative artificial intelligence, covering model creation, deployment, from-scratch implementations, API utilization, interactive applications, and terminal-based tools.

## Key Features & Benefits

- **Generative AI Model Creation:** Code for building generative AI models from scratch.
- **API Integration:** Examples of using generative AI APIs.
- **Deployment Strategies:**  Guidance and code snippets for deploying generative AI models.
- **Interactive Applications:**  Implementation of interactive applications powered by generative AI.
- **Terminal-Based Tools:** Command-line tools leveraging generative AI capabilities.
- **Fine-Tuning & Pre-Training:** Scripts for fine-tuning and pre-training large language models.

## Prerequisites & Dependencies

Before running the code, ensure you have the following installed:

- **Python:**  Version 3.7 or higher is recommended.
- **pip:** Python package installer (comes with Python).
- **Hugging Face Transformers:** Library for working with pre-trained models.
- **Datasets:**  Library for accessing and preparing datasets.
- **PyTorch:** Deep learning framework.
- **Other dependencies:**  Listed in `requirements.txt`.

## Installation & Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arvinderjeet/Gen_AI.git
   cd Gen_AI
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples & API Documentation

### Fine-Tuning GPT-2

```python
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, pipeline
)
from datasets import load_dataset

model_name = 'gpt2'
dataset = load_dataset("text", data_files="shoolini_finetune.txt")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    tokeni...
```

_File: `2_CODE/Fine-Tuning/Fine-tuning using GPT2 as FM.py`_

This script demonstrates fine-tuning a GPT-2 model using the Hugging Face Transformers library.  Modify the `shoolini_finetune.txt` dataset to suit your fine-tuning needs.

### Pre-Training Llama 2

```python
from transformers import (
    LlamaTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import load_dataset
import os
import sys

# Set model name and load tokenizer FIRST (before any function definitions)
model_name = "Llama-2-7b-hf"
tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examp...
```

_File: `2_CODE/Pre-Training/Pre-training using LLAMA2 as FM.py`_

This script showcases pre-training a Llama 2 model using the Hugging Face Transformers library. Ensure you have the Llama 2 model properly configured with Hugging Face. Modify the `data_set.txt` file accordingly.

### Model using Hugging Face API (Google Colab)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

HF_TOKEN="<hf_token_here>"

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain recursion in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs =...
```

_File: `2_CODE/google_collab/Model_using_HF.py`_

This script demonstrates how to load and use a pre-trained model from the Hugging Face Hub. **Important**: You must replace `<hf_token_here>` with your actual Hugging Face API token.

## Configuration Options

- **Hugging Face API Token:**  Required for accessing certain models and resources from the Hugging Face Hub. Set the `HF_TOKEN` variable in relevant scripts.
- **Model Names:**  The `model_name` variable in the scripts can be modified to use different pre-trained models.
- **Dataset Paths:** Adjust the `data_files` parameter in the `load_dataset` function to point to your desired datasets.
- **Training Arguments:**  The `TrainingArguments` class in the fine-tuning and pre-training scripts allows you to configure various training parameters such as learning rate, batch size, and number of epochs.

## Contributing Guidelines

We welcome contributions to this project! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Submit a pull request to the main branch.

## License Information

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

## Acknowledgments

- Hugging Face Transformers Library
- Datasets Library
- PyTorch
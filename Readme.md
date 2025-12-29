# Bengali Punctuation Restoration

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Project Structure](#project-structure)

## Problem Statement
Automatic punctuation restoration for Bengali text. The model takes unpunctuated Bengali text as input and predicts the appropriate punctuation marks for each token position. This is particularly useful for:
- ASR (Automatic Speech Recognition) outputs
- Text normalization
- Document processing

**Example:**
```
Input:  ধাঁধাটি ছিলো এরকম প্রথমে একজন ছদ্মবেশে থাকা মানুষের কথা চিন্তা করো তারপর ভাবো সে কী করতে পারে
Output: ধাঁধাটি ছিলো এরকম- প্রথমে একজন ছদ্মবেশে থাকা মানুষের কথা চিন্তা করো, তারপর ভাবো সে কী করতে পারে?
```

## Solution Approach
This task is formulated as a **token classification problem** where each token in the input text is classified into one of 8 punctuation classes:
- `O` (No punctuation)
- `COMMA` (,)
- `DARI` (।)
- `QUESTION` (?)
- `EXCLAMATION` (!)
- `HYPHEN` (-)
- `SEMICOLON` (;)
- `COLON` (:)

The solution uses a transformer-based approach with XLM-RoBERTa-Large as the backbone, augmented with a BiLSTM layer for sequence modeling.

**N.B:** Coding is done in Python and PyTorch is used as the ML framework.

## Dataset

### Data Sources
The model is trained on Bengali text data from multiple sources:
- ASR transcription datasets
- Bengali text corpora
- Paraphrase datasets

### Dataset Statistics
- **Training data:** ~145,000 sentences
- **Validation data:** ~36,000 sentences

### Class Distribution
The dataset contains varied distribution of punctuation types:

| Class | Train Count | Valid Count |
|:------|:------------|:------------|
| O (No punctuation) | 145,422 | 36,357 |
| COMMA (,) | 42,584 | 10,786 |
| DARI (।) | 66,853 | 16,882 |
| QUESTION (?) | 56,906 | 14,358 |
| EXCLAMATION (!) | 23,089 | 5,828 |
| HYPHEN (-) | 12,308 | 3,109 |
| SEMICOLON (;) | 5,926 | 1,499 |
| COLON (:) | 22,739 | 5,741 |

## Preprocessing

### Text Normalization
Bengali text requires normalization due to multiple Unicode representations of the same character. The `bnunicodenormalizer` library is used to convert all text to a consistent Unicode representation.

### Punctuation Processing
1. **Unnecessary punctuation removal:** Remove punctuation marks not in our target set
2. **Duplicate punctuation handling:** Remove consecutive duplicate punctuation
3. **Space normalization:** 
   - Remove spaces before punctuation
   - Add spaces after punctuation
4. **Label alignment:** Align punctuation labels with tokenized tokens (handles subword tokenization)

### Data Processing Pipeline
```
Raw Text → Normalize → Extract Punctuation → Remove Unnecessary Punctuation
→ Normalize Spaces → Tokenize → Align Labels → Training Data
```

## Model Architecture

### Base Model
- **Model:** XLM-RoBERTa-Large
- **Model Name:** `xlm-roberta-large`
- **Hidden Size:** 1024
- **Attention Heads:** 16
- **Hidden Layers:** 24

### Custom Architecture
```python
Input Text
    ↓
XLM-RoBERTa-Large (Transformer Encoder)
    ↓
BiLSTM (lstm_size=128, bidirectional)
    ↓
Dropout (p=0.2)
    ↓
Linear Layer (256 → 8 classes)
    ↓
Output (Punctuation predictions)
```

### Key Features
- **Transformer backbone:** XLM-RoBERTa-Large for multilingual representation
- **BiLSTM layer:** Captures sequential dependencies for better context modeling
- **Token-level classification:** Predicts punctuation for each token independently

## Training Process

### Training Configuration
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** AdamW
  - Learning Rate: 2e-6
  - Weight Decay: 1e-2
- **Scheduler:** CosineAnnealingLR
- **Batch Size:** 8 (training), 16 (validation)
- **Gradient Accumulation:** 4 steps
- **Max Sequence Length:** 128 tokens
- **Epochs:** 25
- **Early Stopping:** Patience of 3 epochs

### Metrics
- **Primary Metric:** Weighted F1-Score
- **Secondary Metrics:** Accuracy, Loss

### Hardware
- **GPU:** Tesla P100
- **Training Time:** ~9 hours for 25 epochs

## Results

### Model Performance

| Epoch | Train Loss | Valid Loss | Train F1 | Train Acc | Valid F1 | Valid Acc |
|:------|:-----------|:-----------|:---------|:----------|:---------|:----------|
| 22    | 0.0041     | 0.0474     | 0.9929   | 0.9930    | 0.9632   | **0.9604** |
| 23    | 0.0038     | 0.0499     | 0.9934   | 0.9935    | 0.9609   | 0.9591    |
| 24    | 0.0035     | 0.0501     | 0.9936   | 0.9938    | **0.9645** | 0.9600  |
| 25    | 0.0035     | 0.0496     | 0.9938   | 0.9940    | 0.9614   | 0.9579    |

### Best Model Performance
- **Validation F1-Score:** 96.45%
- **Validation Accuracy:** 96.00%
- **Training F1-Score:** 99.36%
- **Training Accuracy:** 99.38%

The model achieved these results at **Epoch 24**, demonstrating excellent generalization with minimal overfitting.

### Sample Predictions
```
Input:  অমুসলিমদের যেহেতু ইসলামের নিয়ম কানুন মানতে হয় না তাই এই ধরণের অভিযানে কর্তৃপক্ষ অমুসলিমদের লক্ষ্যবস্তু করে না বলেও জানান মি ইয়াহইয়া
Output: অমুসলিমদের যেহেতু ইসলামের নিয়ম- কানুন মানতে হয় না, তাই এই ধরণের অভিযানে কর্তৃপক্ষ অমুসলিমদের লক্ষ্যবস্তু করে না বলেও জানান মি ইয়াহইয়া।

Input:  বিশেষ করে ডেনিমের নীল জিন্সগুলো ছিল হিপ্পি সম্প্রদায়ের সাধারণ পোশাক
Output: বিশেষ করে ডেনিমের নীল জিন্সগুলো ছিল হিপ্পি সম্প্রদায়ের সাধারণ পোশাক!

Input:  ধাঁধাটি ছিলো এরকম প্রথমে একজন ছদ্মবেশে থাকা মানুষের কথা চিন্তা করো
Output: ধাঁধাটি ছিলো এরকম- প্রথমে একজন ছদ্মবেশে থাকা মানুষের কথা চিন্তা করো।
```

## Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bengali-punctuation-restoration.git
cd bengali-punctuation-restoration

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Bengali normalizer
pip install git+https://github.com/csebuetnlp/normalizer
```

### Required Packages
```
torch>=1.10.0
transformers>=4.30.0
pandas>=1.3.0
numpy>=1.21.0
tqdm>=4.62.0
scikit-learn>=1.0.0
bnunicodenormalizer
```

## Usage

### Project Structure
```
bengali-punctuation-restoration/
├── configuration.py          # Configuration and hyperparameters
├── dataset.py               # Custom Dataset class
├── Model.py                 # Model architecture
├── prepare_data.py          # Data preprocessing script
├── train.py                 # Training script
├── utils.py                 # Training utilities
├── punctuation_prediction.py # Inference script
├── best_model.bin          # Trained model weights
├── requirements.txt         # Dependencies
└── bengali-punctuation-restoration.ipynb  # Jupyter notebook
```

### Data Preparation

Use the `prepare_data.py` script to preprocess your data.

This script will:
- Normalize Bengali text
- Extract and process punctuation marks
- Generate token-level labels
- Create a processed `modified_data.parquet` file ready for training

Configuration parameters can be modified in `configuration.py` or passed via command-line arguments.

### Training

```bash
python train.py
```

The training configuration can be modified in `configuration.py`:
```python
class CONFIG:
    model_name = "xlm-roberta-large"
    max_length = 128
    train_batch_size = 8
    valid_batch_size = 16
    num_epochs = 25
    learning_rate = 2e-6
    # ... other parameters
```

### Inference

```python
from helper import pun_inference_fn
from Model import TokenClassificationModel
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = TokenClassificationModel()
model.load_state_dict(torch.load("best_model.bin", map_location="cpu"))
model.eval()

# Single sentence prediction
text = ""
result = pun_inference_fn(text, model, tokenizer)
print(result)
```

### Interactive Training and Inference

For interactive exploration, training, and inference, use the [Jupyter notebook](bengali-punctuation-restoration.ipynb). The notebook contains the complete workflow with visualization and step-by-step explanations.

## Future Improvements
- [ ] Add support for more punctuation types
- [ ] Experiment with lighter models (DistilBERT)
- [ ] Add data augmentation techniques
- [ ] Add support for mixed language text

## Citation
If you use this code in your research, please cite:

```bibtex
@misc{bengali-punctuation-restoration,
  author = {Sabbir Hossain Ujjal},
  title = {Bengali Punctuation Restoration using XLM-RoBERTa and BiLSTM},
  year = {2023},
  publisher = {GitHub},
  url = {[https://github.com/yourusername/bengali-punctuation-restoration](https://github.com/sabbirhossainujjal/Bengali_Punctuation_Restoration)}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- XLM-RoBERTa model from Hugging Face Transformers
- [Bengali Unicode Normalizer](https://csebuetnlp.github.io/)
- Dataset sources and contributors

---

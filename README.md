# NEISS Pedestrian Classification

This project aims to classify pedestrian-related incidents within NEISS (National Electronic Injury Surveillance System) data using fine-tuned transformer models.

The NEISS dataset includes a wide range of variables, such as:

`CPSC_Case_Number`, `Treatment_Date`, `Age`, `Sex`, `Race`, `Hispanic`, `Body_Part`, `Diagnosis`, `Disposition`, `Location`, `Fire_Involvement`, `Product_1`, `Alcohol`, `Drug`, `Narrative_1`, `Stratum`, `PSU`, `Weight`

This project primarily focuses on the `Narrative_1` field — a free-text description of why the patient presented to the emergency department. The goal is to fine-tune a pre-trained Large Language Model (LLM) to determine whether the narrative describes a pedestrian being struck by a motor vehicle.


## Project Overview

- **`old_models/`**  
  Contains earlier modeling experiments using DistilBERT with PEFT (Parameter-Efficient Fine-Tuning) and LoRA. These models explored lightweight fine-tuning techniques but were eventually outperformed by full fine-tuning.

- **`neiss_colab.ipynb`**  
  The latest model training notebook, run in Google Colab. It fine-tunes a full BERT model (without PEFT/LoRA) using GPU resources. The input was enhanced by concatenating structured variables like `Body Part`, `Diagnosis`, `Disposition`, and `Product 1` to the narrative text for improved context and performance.

- **`run_saved_model.ipynb`**  
  A notebook for running inference using the best-performing model, which was uploaded to the Hugging Face Model Hub. It demonstrates loading the model, running predictions, and displaying evaluation metrics on the labeled test set.

## Current Best Results

The current best model — a fully fine-tuned BERT model — achieves the following performance on a held-out test set:

- **Accuracy**: 93.9%  
- **Precision**: 93.2%

---

## Repository Structure

```python
NEISS/
│
├── data/                           # Contains NEISS data (2014–2023) downloaded from the official site
│   ├── labeled_data/               # Hand-labeled samples used for training and evaluation
│   
│
├── old models/                     # Earlier model experiments using DistilBERT and LoRA
│   └── results/                    # Checkpoints and results from old model training runs             
│
│
├── neiss_colab.ipynb               # Google Colab notebook: fine-tunes BERT (no LoRA) using GPU
├── run_saved_model.ipynb           # Loads the best Hugging Face model and runs evaluation on test datata
└── README.md                       # Project overview and instructions
```

## Model Deployment

The best-performing model was fully fine-tuned and uploaded to the Hugging Face Model Hub under the repository [`DevJGraham/neiss_clf_bert_uncased`](https://huggingface.co/DevJGraham/neiss_clf_bert_uncased).

You can easily load the model and tokenizer using the Hugging Face `transformers` library:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("DevJGraham/neiss_clf_bert_uncased")
tokenizer = AutoTokenizer.from_pretrained("DevJGraham/neiss_clf_bert_uncased")
```




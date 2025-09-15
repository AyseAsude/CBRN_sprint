# Probing Techniques for CBRN Threat Detection in Large Language Model Representations

## Overview

We developed a probe-based approach for detecting harmful CBRN content in large language model (LLM) representations. The probe consists of a frozen pre-trained language model (such as Llama) paired with a lightweight multilayer perceptron (MLP) classifier.

## Methodology

### Architecture

Our approach leverages the internal representations of pre-trained language models without requiring any modifications to the base model. The process works as follows:

1. **Input Processing**: Instruction-answer pairs are fed into the frozen language model, which generates internal representations (hidden states).

2. **Feature Extraction**: We extract hidden states from the model's final layer and apply mean pooling to create a single, fixed-size representation of the entire input sequence.

3. **Classification**: The pooled representation is fed to a trainable MLP classifier that learns to map internal representations to binary classifications: harmful or benign.

### Theoretical Foundation

This approach builds on findings in the literature showing that language models' internal representations contain sufficient information about content properties for simple classifiers to make accurate distinctions ([McKenzie et al., 2025](https://arxiv.org/abs/2506.10805); [Patel and Wang, 2024](https://openreview.net/pdf?id=qbvtwhQcH5)). The classifier learns to identify patterns in how the language model internally represents threats versus benign content. This targeted approach enables the detection of dual-use information and potential CBRN hazards in language model outputs without requiring extensive computational resources, maintaining computational efficiency through the lightweight classification head.

## Models

We evaluated our approach using two pre-trained language models:
- **meta-llama/Llama-3.1-8B-Instruct**
- **mistralai/Mistral-7B-Instruct-v0.2**

## Datasets

### Harmful Content
We used [Sorry Bench](https://github.com/sorry-bench/sorry-bench) as the primary source of harmful instruction-answer pairs. Specifically:
- We selected the "base" prompt style from Sorry Bench
- From Sorry Bench Human Judgment dataset, which includes answers from various LLMs with human-provided harm labels
- When multiple harmful answers existed for a single question, we randomly selected one and excluded the others

### Benign Content
For benign training samples, we utilized:
- **A subset of**: [CBRN-Finetuning dataset](https://huggingface.co/datasets/WangWeiQi/CBRN-Finetuning)
- **A subset of refusal samples**: [Sorry Bench](https://github.com/sorry-bench/sorry-bench) where models decline to answer harmful requests

### Test set
- **A subset of**: [CBRN-Finetuning dataset](https://huggingface.co/datasets/WangWeiQi/CBRN-Finetuning)
- **A subset of WMDP**: Manually selected "most explicitly" harmful-appearing samples from the [WMDP dataset](https://huggingface.co/datasets/cais/wmdp)

## Evaluation

Due to time constraints during this sprint, we limited the size of our test set to enable timely evaluation.

We used [Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B) to generate ground truth safety labels for our evaluation. The process worked as follows:

- Answer Generation: Each model (Mistral and Llama) generated responses to the same set of test questions from WMDP and CBRN datasets
- Safety Classification: Llama Guard evaluated each unique question-answer pair, classifying them as either "safe" or "unsafe"
- Ground Truth Assignment: Llama Guard's classifications were treated as ground truth labels for probe evaluation

### Overall Test Set Composition

| Model | Total Examples | Safe Examples | Unsafe Examples | Safe % | Unsafe % |
|-------|----------------|---------------|-----------------|--------|----------|
| Mistral | 167 | 148 | 19 | 88.6% | 11.4% |
| Llama | 167 | 163 | 4 | 97.6% | 2.4% |

### Test Set Breakdown by Dataset

| Model | Dataset | Examples | Safe | Unsafe | Safe % | Unsafe % |
|-------|---------|----------|------|--------|--------|----------|
| Mistral | WMDP | 117 | 98 | 19 | 83.8% | 16.2% |
| Mistral | CBRN | 50 | 50 | 0 | 100.0% | 0.0% |
| Llama | WMDP | 117 | 113 | 4 | 96.6% | 3.4% |
| Llama | CBRN | 50 | 50 | 0 | 100.0% | 0.0% |


## Results

### Performance Metrics

![Accuracy by Dataset](plots/accuracy_by_dataset.png)
*Figure 1: Classification accuracy across different datasets*

![F1 Score by Dataset](plots/f1_score_by_dataset.png)
*Figure 2: F1 scores across different datasets*

![Confusion Matrices](plots/2_confusion_matrices.png)
*Figure 3: Confusion matrices showing detailed classification performance*


### Limitations

A critical limitation of our evaluation is the severe class imbalance in the test set. Llama Guard classified only 2.4% (4/167) of Llama-generated responses and 11.4% (19/167) of Mistral-generated responses as unsafe. This imbalance makes it difficult to assess true positive rate reliability, as conclusions about harmful content detection are based on very few examples.

## Future Work

Repeating this work with larger datasets would provide more stable metrics and stronger conclusions. Due to time constraints, we had to limit both training and testing dataset sizes. While the probe shows promise, evaluation remains limited by dataset imbalance. Nevertheless, training probes is significantly faster and cheaper than training LLMs, providing a low-barrier entry point for safety applications.

## Usage

### Training a probe:
```bash
python src/train_probe.py --config configs/train_probe_config.yaml
```

### Generating answers for testing:
```bash
python src/test_probe_dynamic.py --config configs/test_probe_dynamic_config.yaml
```

### To evaluate using Llama Guard:

```bash
python -m src.llamaguard_eval.main \
    --dataset_path ./generated_answers.json \
    --dataset_type json \
    --models meta-llama/Llama-Guard-3-8B
```

### Testing generated question-answer pairs with probe predictions:

```bash
python src/test_probe.py --config configs/test_probe_config.yaml
```

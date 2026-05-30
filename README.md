# Automatic Elasticsearch Query Evaluation System

This project develops an automatic NLP-based system for evaluating Elasticsearch queries written in JSON format. The system compares a student query against a correct reference query and assigns a rubric-based score from 0 to 4 with an explanatory rationale.

## Problem Statement

Writing Elasticsearch queries can be challenging because small mistakes in JSON structure, Boolean logic, filtering, sorting, or operators may lead to incorrect search results. This project aims to support students and developers by automatically evaluating query correctness and identifying errors more efficiently.

## Dataset

The project uses the Reuters-21578 news dataset, organized into JSON files. The data was processed and mapped into Elasticsearch-style fields:

- `title` and `content`: text fields
- `topics` and `countryKeys`: keyword fields
- `date`: date field
- `georeferences`: nested field

## Generated Query Types

The dataset includes multiple Elasticsearch query categories:

- Match queries
- Boolean queries
- Range queries
- Sorting queries
- Nested queries
- Multi-field search queries

## Sample Structure

Each generated sample contains:

- Task description
- Correct reference query
- Student submission query
- Rubric-based score from 0 to 4
- Evaluation rationale

## Dataset Split

The dataset was split into:

- 70% training
- 15% validation
- 15% testing

## Methodology

1. Preprocess the Reuters dataset.
2. Generate Elasticsearch query evaluation samples.
3. Convert JSON data into JSONL instruction-tuning format.
4. Evaluate the base model before fine-tuning.
5. Fine-tune Mistral-7B using Unsloth and LoRA.
6. Evaluate the fine-tuned model and compare results with the baseline.

## Model

The project uses:

- Base model: Mistral-7B Instruct
- Fine-tuning method: LoRA
- Optimization framework: Unsloth
- Task format: instruction tuning

## Results

| Metric | Before Fine-Tuning | After Fine-Tuning |
|---|---:|---:|
| Accuracy | 54.84% | 80.65% |
| MAE | 0.7419 | 0.1935 |
| QWK | 0.6895 | 0.9565 |

The fine-tuned model achieved a clear improvement in prediction accuracy and produced scores closer to the ground truth.

## Key Findings

- Fine-tuning improved accuracy by 25.81%.
- Predictions became more aligned with the rubric-based ground truth.
- Score levels 0 and 4 achieved strong performance.
- Score 1 remained the most challenging category.

## Reflection

The dataset contained 201 samples, which was suitable for the project scope. However, increasing the dataset size could improve generalization. Training was performed using an NVIDIA A100 GPU, but shared GPU usage affected training speed and stability. Dropout and early stopping were used to reduce overfitting and improve training reliability.

## Future Work

- Expand the dataset size.
- Support more Elasticsearch query types.
- Handle more complex query structures.
- Improve performance on difficult score categories.

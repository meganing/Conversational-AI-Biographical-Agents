# Personality-Consistent Biographical Conversational Agent

## Overview

This project explores how a conversational agent can respond with a stable fictional personality instead of producing generic chatbot answers. The prototype uses Geralt of Rivia as a test character and combines dialogue fine-tuning, retrieved narrative context, and automated personality evaluation.

The repository contains the implementation notebooks and recorded experiment outputs used to compare two training-data workflows. The comparison is not a direct Gemini-versus-Qwen chat-model benchmark: Qwen is the fine-tuned dialogue model in both branches, while Gemini is used for one data-generation path and for automated personality scoring.

## Problem

Fluent language generation does not automatically produce a convincing character. A useful character agent must balance three concerns:

- Preserve recognizable speech habits and behavioral traits across different prompts.
- Use relevant story context without inventing details about the character.
- Measure personality consistency in a way that is more informative than surface-level text similarity.

This project investigates a practical pipeline for addressing those concerns with open-weight language-model fine-tuning and an external personality-scoring loop. It also compares whether the source of the training examples changes the resulting persona alignment.

## What we built

The system is organized as two related experimental workflows:

1. Prepare and annotate character dialogue with Big Five personality dimensions.
2. Create training examples through two branches:
   - Gemini-assisted: Gemini 2.5 Flash generates synthetic training examples.
   - Qwen-assisted: Qwen generates the training examples locally.
3. Fine-tune `Qwen2.5-7B-Instruct` with parameter-efficient LoRA/QLoRA methods in each branch.
4. Generate preference examples and apply a DPO stage to each fine-tuned model.
5. Assemble responses from persona instructions, retrieved narrative context, and conversation history.
6. Use Gemini 2.5 Flash as the personality evaluator for both branches.
7. Compare the SFT and DPO results using distance from the reference Big Five profile.

The context-retrieval component is intended to reduce lore-related hallucination, while the personality loop provides a measurable signal for character drift.

## Architecture

```text
                         Training examples
                              /          \
             Gemini-assisted path     Qwen-assisted path
                              \          /
                         Qwen2.5-7B-Instruct
                         SFT -> DPO fine-tuning
                                    |
User prompt -> persona rules + retrieved context + dialogue history
                                    |
                         Candidate response
                                    |
                         Gemini personality evaluator
                                    |
                         Big Five distance comparison
```

## Data and evaluation

The source study used 5,986 extracted character dialogue lines. The lines were cleaned, deduplicated, and annotated with scores for openness, conscientiousness, extraversion, agreeableness, and neuroticism. The canonical character profile was used as the reference point for evaluation.

The primary metric is Euclidean distance between the evaluator's Big Five vector and the reference vector; lower values indicate closer personality alignment. The included comparison files contain ten prompt evaluations for each model stage, so the results should be interpreted as an experiment snapshot rather than a general benchmark.

Recorded comparison:

| Training-data workflow | SFT distance | DPO distance | Reported change |
| --- | ---: | ---: | ---: |
| Gemini-generated examples | 0.6950 | 0.6485 | 6.68% lower |
| Qwen-generated examples | 0.8248 | 0.7139 | 13.45% lower |

## Repository structure

```text
gemini/
├── data_generation.ipynb              # Gemini-assisted data generation
└── gemini_assisted_training.ipynb     # Gemini workflow and training/evaluation steps
qwen/
└── qwen_assisted_training.ipynb       # Qwen-assisted training/evaluation workflow
results/
├── gemini_thesis_results_comparison.csv
├── gemini_results_chart.png
├── qwen_thesis_results_comparison.csv
└── qwen_results_chart.png
```

## Contributions

I worked with one other engineer on the technical implementation. My work covered the model-training and evaluation workflows, Qwen fine-tuning, Gemini-assisted data/evaluation steps, experiment organization, and comparison of the recorded results. Another team member acquired the source dataset.

## Running the notebooks

The notebooks were developed in a GPU notebook environment. They use packages including:

- Python
- PyTorch and Hugging Face Transformers
- Unsloth and PEFT/LoRA tooling
- Hugging Face Datasets
- NumPy, pandas, and SciPy
- Google's generative-AI client

The notebooks expect local dataset and model paths that are not included here. Before running them, adapt those paths and provide your own API credentials through the notebook runtime or environment variables. Never commit credentials to the repository.

## What is intentionally excluded

This public repository does not include raw dialogue datasets, generated training files, model checkpoints, adapter weights, cached files, or other large/private artifacts. These materials may contain team-owned or third-party content and are not necessary to review the implementation approach.

## Limitations and next steps

- The evaluation set is small and should not be treated as a statistically conclusive benchmark.
- Automated personality scoring can be sensitive to prompt wording and evaluator behavior.
- The repository documents the dialogue and evaluation pipeline; an end-to-end voice deployment is outside the current public artifact.
- A stronger follow-up would use a larger held-out evaluation set, human ratings, ablation studies, and explicit provenance/consent controls for any real-person data.

# Biographical Agents — Conversational AI

This repository contains the reproducible notebooks and evaluation artifacts for the URD-3 Conversational-AI project, focused on building a personality-consistent biographical agent.

## What this repository demonstrates

- Dataset-generation and evaluation workflow using Gemini.
- Qwen2.5-7B-Instruct supervised fine-tuning and preference-optimization workflow.
- Comparison artifacts from the Gemini- and Qwen-based workflows.

## My contribution

I worked with one other engineer to implement the technical system. My contribution included the training/evaluation implementation, Qwen fine-tuning workflow, Gemini-assisted data/evaluation workflow, experiment organization, and result comparison. Dataset acquisition was performed by another team member.

## Repository structure

```text
gemini/
├── URD3_dataGen.ipynb
└── URD3_Train_gemini.ipynb
qwen/
└── URD3_Train_qwen.ipynb
results/
├── gemini_thesis_results_comparison.csv
├── gemini_results_chart.png
├── qwen_thesis_results_comparison.csv
└── qwen_results_chart.png
```

## Important scope note

This is a cleaned source-and-results repository. Model checkpoints, adapter weights, cached training files, raw datasets, and generated training data are intentionally excluded because they are large, may contain team-owned or third-party material, and are not required to understand the implementation.

The notebooks were sanitized to remove an accidentally embedded API key. Before running them, configure your own provider credentials through environment variables or the notebook runtime; never commit keys to GitHub.

## Reproducibility

The notebooks were originally developed in a GPU notebook environment and use packages including `unsloth`, `transformers`, `datasets`, `pandas`, `scipy`, and Google's generative-AI client. Paths and package versions may need to be adapted before rerunning.

The committed CSV files and charts are retained as recorded experiment artifacts. Verify the dataset split, sample counts, and evaluation protocol before treating the reported values as a formal benchmark.

## Status

The repository is being prepared for portfolio use. It documents the implementation and experiment workflow; it is not intended to provide downloadable model weights or a one-click production deployment.

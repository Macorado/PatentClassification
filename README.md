# PatentClassification — Predicting USPTO Patent Allowance Outcomes

## Overview

This project builds a machine learning pipeline to predict whether a patent application results in allowed claims using USPTO prosecution history signals (office actions, rejection patterns, and citation activity).

## Data

Input data is stored locally as pickle files:

- `data/office_actions.pkl`
- `data/rejections.pkl`
- `data/citations.pkl`

> Note: Data files are not committed to GitHub.

## How to Run

1. Create venv + install dependencies:

   - `python -m venv .venv`
   - activate venv
   - `pip install -r requirements.txt`

2. Run pipeline:
   - `python run.py`

## Outputs

Model metrics and figures are written to:

- `reports/metrics/`
- `reports/figures/`

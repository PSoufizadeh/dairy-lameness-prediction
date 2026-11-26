# Dairy Cattle Lameness Prediction — Exploratory Pilot

**Research portfolio | Veterinary data science | Precision livestock farming**

This exploratory pilot investigates whether routinely collected dairy farm records can support estimation of locomotion (lameness) scores in dairy cattle. Candidate inputs include production, body condition, reproductive and historical clinical observations. The project is ongoing and is not a validated diagnostic or operational decision-support system.

## Scope

The public edition contains selected, reorganized Python utilities adapted from the original exploratory notebooks: spreadsheet concatenation, missingness analysis, grouped descriptive summaries, score transitions, preprocessing, and classification experiments using K-nearest neighbors, decision trees, random forests and neural networks (a scikit-learn MLP baseline). The original TensorFlow experiments and collaborative notebooks are not reproduced verbatim.

The public refactor uses animal-grouped holdout and cross-validation and fits preprocessing and optional oversampling inside the training pipeline. These are methodological improvements to the public example, not claims about the original experiments. Temporal availability of features and independent farm-level validation require further investigation. The score is treated as a classification target; quadratic weighted kappa is included to reflect its ordered nature, assuming ordered numeric score labels.

## Repository structure

- `src/experiment.py`: data loading, preprocessing, model selection and evaluation.
- `src/eda.py`: exploratory summaries.
- `requirements.txt`: Python dependencies.
- `.gitignore`: local-data and artifact exclusions.

## Usage

Install dependencies using `python -m pip install -r requirements.txt`. Import `run_experiment` from `src.experiment` and supply an authorized local dataframe, an explicit feature list, a numeric ordered score target and an animal identifier. Do not include identifiers, outcome-derived variables or future observations among predictors. Outputs are returned in memory and should remain private. No source dataset is distributed, and the repository is not an exact reproduction of the original study.

## Confidentiality and status

Farm records, animal identifiers, study-specific results, fitted models, figures and manuscript drafts are excluded. This is an early-stage research pilot; no clinical validity or deployment readiness is claimed. For academic enquiries or collaboration, contact Parham Soufizadeh via [LinkedIn](https://www.linkedin.com/in/parham-soufizadeh).

The public code is a curated refactor of selected methods; it does not imply sole authorship of the original collaborative work. No reuse license is granted here.

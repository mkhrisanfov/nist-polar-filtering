# Identification of potentially erroneous retention indices for polar stationary phases in NIST 23 database via the “yellow cards” multi-model voting approach
*by Mikhail Khrisanfov, Dmitriy Matyushin*

This repository contains code used for optimization of the architecute and hyperparameters of the models (`notebooks/*-optuna-*.py`), the 5-fold training and prediction procedure `notebooks/5Fold-*.py` The `src` directory contains definitions of the models and their respective Dataset classes along with some utility functions.

## Abstract

Predicting Kovats retention indices for polar stationary phases remains a challenge due to the limited chemical space available compared to the non-polar entries in the NIST RI database. In this work, a two-stage transfer learning approach was employed: (i) pretraining on non-polar indices and (ii) fine-tuning on polar indices, to enhance prediction accuracy across five model architectures. Model hyperparameters were optimized via Optuna, achieving competitive prediction accuracies. The ensemble-voting [“yellow cards” framework](https://github.com/mkhrisanfov/yellow-cards-workflow), employing five models across five differently shuffled datasets, was adapted to identify 436 potentially erroneous entries (0.5% of the dataset). Investigation into these errors uncovered potential systematic shifts and mislabeling in published literature, demonstrating that this ensemble filtering approach is an effective tool for the automated refinement of polar retention indices.


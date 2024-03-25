# -*- coding: utf-8 -*-

from scipy import stats
import numpy as np

# model names and weighted f1-score from the 3 runs
models = {"AGENDA-T5": [0.41, 0.32, 0.43],
          "AGENDA-MT5": [0.32, 0.31, 0.38],
          "AGENDA-BERT": [0.66, 0.64, 0.67],
          "AGENDA-MBERT": [0.70, 0.69, 0.64],
    
          "AGENDA-RTE-EN-T5": [0.67, 0.64, 0.66],
          "AGENDA-RTE-BI-MT5": [0.68, 0.72, 0.74],
          "AGENDA-RTE-EN-BERT": [0.67, 0.66, 0.65],
          "AGENDA-RTE-BI-MBERT": [0.69, 0.68, 0.65],
          
          "AGENDA-TFIDF-SVM": [0.61, 0.37, 0.33],
          "AGENDA-MLC-T5": [0.32, 0.32, 0.40],
          "AGENDA-MLC-MT5": [0.33, 0.34, 0.40],
          "AGENDA-MLC-BERT": [0.43, 0.40, 0.46],
          "AGENDA-MLC-MBERT": [0.47, 0.50, 0.53],
          
          "RTE-EN-T5": [0.48, 0.47, 0.49],
          "RTE-BI-MT5": [0.41, 0.40, 0.44],
          "RTE-EN-BERT": [0.41, 0.39, 0.45],
          "RTE-BI-MBERT": [0.34, 0.38, 0.42],
          "BART-MNLI": [0.37, 0.37, 0.31],
          
          "SBERT-HYPOTHESES": [0.38, 0.36, 0.38],
          "MSBERT-HYPOTHESES": [0.33, 0.31, 0.32],
          "SBERT-LABELS": [0.14, 0.16, 0.12],
          "MSBERT-LABELS": [0.21, 0.18, 0.15],
         }

# just the names of the models
all_models = list(models.keys())
n_models = len(all_models)
alpha = 0.05

# Perform pairwise t-tests
for i in range(n_models):
    model_a = all_models[i]
    values_model_a = np.array(models[model_a])
    mean_model_a = np.mean(values_model_a)
    
    for j in range(i + 1, n_models):
        model_b = all_models[j]
        values_model_b = np.array(models[model_b])
        t_stat, p_value = stats.ttest_rel(values_model_a, values_model_b)
        
        mean_model_b = np.mean(values_model_b)
        
        
        if p_value < alpha and mean_model_a > mean_model_b:
            print(f"t-test between model {model_a} and model {model_b}: p-value = {p_value}")
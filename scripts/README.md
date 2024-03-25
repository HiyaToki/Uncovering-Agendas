# Techniques for Agenda Detection

Code used for experiments described in the paper "Uncovering Agendas: A Novel French & English Dataset for Agenda Detection on Social Media".

## Instructions

Please doanlowd and prepare all required data, following instructions found in: 
```
/data/agenda/
```
First, fine-tune BERT and T5 for the RTE task by running `./scripts/rte_bert/train_bert_rte.py` and `/scripts/rte_t5/train_t5_rte.py`. This will fine-tune the models on the combined (and binarized) RTE/MNLI/SNLI datasets. <br> 

Second, fine-tune the resulting models for the agenda task by running `./scripts/rte_bert/train_bert_agenda.py` and `/scripts/rte_t5/train_t5_agenda.py`. This will fine-tune the RTE models on the the agenda dataset. Since the experiments are done three-fold, there are going to be three models per run. Models will be organized under R1, R2, and R3 directories. R**x** indicates the corresponding train/dev/test number. <br>

Third, run `./scripts/rte_bert/predict_bert_agenda.py` and `/scripts/rte_t5/predict_t5_agenda.py` to generate prediction outputs using the RTE and Agenda fine-tuned models. The results will go under `../evaluations/`. 

Fourth, fine-tune BERT and T5 directly on the agenda dataset, without "pre-training" on RTE. These are the "MLC" models. For that run `./scripts/bert/train_bert_agenda.py` and `/scripts/t5/train_t5_agenda.py`. Since the experiments are done three-fold, there are going to be three models per run. <br>

Fifth, run `./scripts/bert/predict_bert_agenda.py` and `/scripts/t5/predict_t5_agenda.py` to generate prediction outputs using the "MLC" Agenda fine-tuned models. The results will go under `../evaluations/`. 

Sixth, run `./scripts/evaluate_file_agenda.py` to obtain evaluation results for all models. The evaluation reports will go under `../evaluations/Rx/threshold/`. Please parse these files and prepare the `./scripts/stats_ttest_evaluation.py` script by copy/pasting the results to conduct the statistical significance analysis. For the paper, we used the weighted F1-score. 

Seventh, after transfering the evaluation results in the `./scripts/stats_ttest_evaluation.py` script, run it to get the statistical significance analysis results. 

This shoudl be all :)

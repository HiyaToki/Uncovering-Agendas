# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from sklearn.metrics import classification_report, f1_score

import logging
logging.disable(logging.WARNING)

# set path and import map file
sys.path.append('../scripts/')
import agenda_label_maps

type_to_label = agenda_label_maps.type_to_label
label_to_index = agenda_label_maps.label_to_index
ordered_agenda_labels = agenda_label_maps.ordered_agenda_labels

# method to load a json dataset to memory
def load_json(filename):
    print("Loading file: ", filename)
    with open(filename, 'r', encoding = 'utf-8') as json_file:
        dataset = json.load(json_file)

    return dataset

# method to encode ground truth labels as 1-hot vectors
def endoce_gt_one_hot(labels):
    one_hot_labels = []
    for agenda_labels in labels:
        one_hot_label = [0] * len(label_to_index)

        for label in agenda_labels:
            one_hot_label[label_to_index[label]] = 1

        one_hot_labels.append(one_hot_label)
    
    return one_hot_labels

# method to get predictions and ground truth labels
def get_gt_labels_plus(dataset):
    confidences = []
    is_french = []
    labels = []

    for key in dataset.keys():
        for language in dataset[key]["prediction"].keys():
            confidences.append(dataset[key]["prediction"][language])
            labels.append(dataset[key]["labels"])

            if language == "fr":
                is_french.append(True)

            else:
                is_french.append(False)

    return labels, confidences, is_french

### SCRIPT ###

# model names
models = ["AGENDA-RTE-EN-BERT",
          "AGENDA-RTE-BI-MBERT",
          "AGENDA-MBERT",
          "AGENDA-BERT",
          "RTE-EN-BERT",
          "RTE-BI-MBERT",
          "AGENDA-RTE-EN-T5",
          "AGENDA-RTE-BI-MT5",
          "AGENDA-MT5",
          "AGENDA-T5",
          "RTE-EN-T5",
          "RTE-BI-MT5",
          "AGENDA-MLC-BERT",
          "AGENDA-MLC-MBERT",
          "AGENDA-MLC-T5",
          "AGENDA-MLC-MT5",
          "SS-SBERT-HYPOTHESES",
          "SS-MSBERT-HYPOTHESES",
          "SS-SBERT-LABELS",
          "SS-MSBERT-LABELS",
         ]

# model pre-predicted test files
test_files = ["agenda_rte_en_bert_test.json",
              "agenda_rte_bi_mbert_test.json",
              "agenda_mbert_test.json",
              "agenda_bert_test.json",
              "rte_en_bert_test.json",
              "rte_bi_mbert_test.json",
              "agenda_rte_en_t5_test.json",
              "agenda_rte_bi_mt5_test.json",
              "agenda_mt5_test.json",
              "agenda_t5_test.json",
              "rte_en_t5_test.json",
              "rte_bi_mt5_test.json",
              "agenda_mlc_bert_test.json",
              "agenda_mlc_mbert_test.json",
              "agenda_mlc_t5_test.json",
              "agenda_mlc_mt5_test.json",
              "ss_sbert_hypotheses_test.json",
              "ss_msbert_hypotheses_test.json",
              "ss_sbert_labels_test.json",
              "ss_msbert_labels_test.json"
             ]

# model pre-predicted dev files
dev_files = ["agenda_rte_en_bert_dev.json",
             "agenda_rte_bi_mbert_dev.json",
             "agenda_mbert_dev.json",
             "agenda_bert_dev.json",
             "rte_en_bert_dev.json",
             "rte_bi_mbert_dev.json",
             "agenda_rte_en_t5_dev.json",
             "agenda_rte_bi_mt5_dev.json",
             "agenda_mt5_dev.json",
             "agenda_t5_dev.json",
             "rte_en_t5_dev.json",
             "rte_bi_mt5_dev.json",
             "agenda_mlc_bert_dev.json",
             "agenda_mlc_mbert_dev.json",
             "agenda_mlc_t5_dev.json",
             "agenda_mlc_mt5_dev.json",
             "ss_sbert_hypotheses_dev.json",
             "ss_msbert_hypotheses_dev.json",
             "ss_sbert_labels_dev.json",
             "ss_msbert_labels_dev.json"
             ]

for run in ["R1", "R2", "R3"]:
    base_directory = "../evaluation/" + run + "/"
    
    for eval_mode in ["threshold", "no-threshold"]:
    
        # produce all reports based on pre-predicted output files
        for i in range(len(test_files)):
            test_file_path = base_directory + test_files[i]
            dev_file_path = base_directory + dev_files[i]
            model_choice = models[i]
        
            # load data
            dev_data = load_json(dev_file_path)
            test_data = load_json(test_file_path)
            
            dev_labels, dev_confidences, _ = get_gt_labels_plus(dev_data)    
            test_labels, test_confidences, test_is_french = get_gt_labels_plus(test_data)
        
            print("Evaluating model:", model_choice)
        
            # encode ground truth as 1-hot vectors
            dev_one_hot_labels = endoce_gt_one_hot(dev_labels)
            test_one_hot_labels = endoce_gt_one_hot(test_labels)
        
            # activate specified mode for eval
            if eval_mode == "threshold":
                scores = []
                thresholds = np.arange(0.3, 1.0, 0.01)
                
                # scan optimal threshold for multi-label evaluation on the dev set
                for threshold in thresholds:
                    dev_one_hot_predictions = []
                    for confidence in dev_confidences:
                        probabilities = confidence["confidences"]
                        one_hot_prediction = [0] * len(label_to_index)
        
                        for i in range(len(probabilities)):
                            if probabilities[i] >= threshold:
                                one_hot_prediction[i] = 1
        
                        dev_one_hot_predictions.append(one_hot_prediction)
        
                    # based on macro f1 score
                    score = f1_score(dev_one_hot_labels, dev_one_hot_predictions, average = 'weighted', zero_division = 0)
                    scores.append(score)
        
                # get the optimal threshold
                optimal_threshold_index = np.argmax(scores)
                optimal_threshold = thresholds[optimal_threshold_index]
        
                print("\tOptimal Threshold found at: ", optimal_threshold, "\n")
        
                # use optimal threshold for final evaluation report using the fixed test set
                one_hot_predictions = []
                for confidence in test_confidences:
                    probabilities = confidence["confidences"]
                    is_predictions = confidence["is_predictions"]
                    one_hot_prediction = [0] * len(label_to_index)
        
                    for i in range(len(probabilities)):
                        if probabilities[i] >= optimal_threshold:
                            one_hot_prediction[i] = 1
        
                    one_hot_predictions.append(one_hot_prediction)
        
            else:
                # no use of threshold, just look at default output
                one_hot_predictions = []
                for confidence in test_confidences:
                    probabilities = confidence["confidences"]
                    is_predictions = confidence["is_predictions"]
                    one_hot_prediction = [0] * len(label_to_index)
        
                    for i in range(len(probabilities)):
                        if is_predictions[i]:
                            one_hot_prediction[i] = 1
        
                    one_hot_predictions.append(one_hot_prediction)
        
                print("")
        
            # classification report for Agenda multi-label eval using optimal threshold
            report = classification_report(test_one_hot_labels, one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)
        
            # create output report file
            output_file = base_directory + eval_mode + "/" + model_choice + "-REPORT.txt"
            os.makedirs(os.path.dirname(output_file), exist_ok = True)
        
            # write overal report
            with open(output_file, "w") as fp:
                if eval_mode == "threshold":
                    fp.write("\n\nAGENDA EVALUATION RESULTS: " + model_choice + " OPTIMAL THRESHOLD: " + str(round(optimal_threshold, 2)) + "\n")
                else:
                    fp.write("\n\nAGENDA EVALUATION RESULTS: " + model_choice + "\n")
        
                fp.write("\t" + report + "\n")
        
            # get english subset report
            if len(set(test_is_french)) == 2:
                en_one_hot_labels = [test_one_hot_labels[i] for i in range(len(test_one_hot_labels)) if not test_is_french[i]]
                en_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if not test_is_french[i]]
        
                # english subset classification reports
                report = classification_report(en_one_hot_labels, en_one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)
        
                # write english report
                with open(output_file, "a") as fp:
                    fp.write("\n\nENGLISH SUBSET AGENDA EVALUATION RESULTS: \n")
                    fp.write("\t" + report + "\n")
        
            # get french subset report
            if len(set(test_is_french)) == 2:
                fr_one_hot_labels = [test_one_hot_labels[i] for i in range(len(test_one_hot_labels)) if test_is_french[i]]
                fr_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if test_is_french[i]]
        
                # french subset classification reports
                report = classification_report(fr_one_hot_labels, fr_one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)
        
                # write french report
                with open(output_file, "a") as fp:
                    fp.write("\n\nFRENCH SUBSET AGENDA EVALUATION RESULTS: \n")
                    fp.write("\t" + report + "\n")

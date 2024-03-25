# -*- coding: utf-8 -*-

import os
import sys
import csv
import ast
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizer, BertForSequenceClassification

import logging
logging.disable(logging.WARNING)

# set path and import map file
sys.path.append('../../scripts')
import agenda_label_maps

type_to_label = agenda_label_maps.type_to_label
label_to_index = agenda_label_maps.label_to_index
index_to_label = agenda_label_maps.index_to_label
ordered_en_hypotheses = agenda_label_maps.en_hypotheses
ordered_fr_hypotheses = agenda_label_maps.fr_hypotheses
label_to_hypotheses = agenda_label_maps.label_to_hypotheses
hypothesis_to_label = agenda_label_maps.hypothesis_to_label
ordered_agenda_labels = agenda_label_maps.ordered_agenda_labels

te_label_to_index = agenda_label_maps.te_label_to_index
te_index_to_label = agenda_label_maps.te_index_to_label

# define label string for no agenda
other_none_label = "None of These"

# method to same datasets into json files
def save_json(dataset, filename):
    print("Saving file:\n\t", filename)

    with open(filename, "w") as json_file:
        json.dump(dataset, json_file)

# method to load a csv dataset to memory
def load_csv_as_dict(filename):
    print("Loading file: ", filename)

    dataset = dict()
    with open(filename, 'r', encoding = 'utf-8') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            doc_id = row[0]
            tweet_id = row[1]
            language = row[2]
            labels = [type_to_label[x] for x in ast.literal_eval(row[3])]

            fr_text = row[4]
            en_text = row[5]

            dataset[doc_id] = dict()
            dataset[doc_id]["tweet_id"] = tweet_id
            dataset[doc_id]["language"] = language
            dataset[doc_id]["fr_text"] = fr_text
            dataset[doc_id]["en_text"] = en_text
            dataset[doc_id]["labels"] = labels

    return dataset

# generate hypotheses based on labels
def get_hypotheses(labels):
    en_hypotheses = []
    fr_hypotheses = []

    for label in labels:
        # generate positive hypotheis based on labels
        (en_hypothesis, fr_hypothesis) = label_to_hypotheses[label]

        en_hypotheses.append(en_hypothesis)
        fr_hypotheses.append(fr_hypothesis)

    return en_hypotheses, fr_hypotheses

# method to return negative examples for TE
def get_negative_hypotheses(en_hypotheses, fr_hypotheses):
    en_neg_hypotheses = []
    fr_neg_hypotheses = []

    # get french negative hypotheses
    for hypothesis in ordered_fr_hypotheses:
        if hypothesis in fr_hypotheses:
            continue

        else:
            fr_neg_hypotheses.append(hypothesis)

    # get english negative hypotheses
    for hypothesis in ordered_en_hypotheses:
        if hypothesis in en_hypotheses:
            continue

        else:
            en_neg_hypotheses.append(hypothesis)

    # shuffle so I can "randomly" pop from the list later
    random.shuffle(en_neg_hypotheses)
    random.shuffle(fr_neg_hypotheses)

    return en_neg_hypotheses, fr_neg_hypotheses

# method to predict Textual Entailment and extract confidence scores, and interpret agenda classes
def predict(premises, hypotheses, model, tokenizer):

    # encode premises and hypotheses
    inputs = tokenizer(premises,
                       hypotheses,
                       padding = True,
                       max_length = 512,
                       truncation = True,
                       return_tensors = 'pt').to("cuda:0")

    te_predictions = [] # for rte-style evaluation
    is_predictions = [False] * len(index_to_label) # for agenda classification
    confidences = [0.0] * len(index_to_label)

    with torch.no_grad():
        logits = model(**inputs).logits

        for i in range(len(logits)):
            probability = F.softmax(logits[i], dim = -1)
            predicted_class_index = logits[i].argmax().item()
            prediction = te_index_to_label[predicted_class_index]

            te_predictions.append(prediction)

            # convert rte predictions to agenda
            is_prediction_output = False
            if prediction == "entailment":
                is_prediction_output = True

            # get agenda label
            label = hypothesis_to_label[hypotheses[i]]
            index = label_to_index[label]

            confidences[index] = round(probability[0].item(), 6)
            is_predictions[index] = is_prediction_output

    # # normalize confidences
    # conf = np.array(confidences)
    # conf_norm = (conf - conf.min()) / (conf - conf.min()).sum()
    # confidences = conf_norm.tolist()

    # Nothing entails
    if True not in is_predictions:
        # default agenda label if TE predics not entailement everywhere hardcode cofidence
        confidences[label_to_index[other_none_label]] = 0.500001
        is_predictions[label_to_index[other_none_label]] = True

    # package confidences into a neat dict
    confidence_dict = {"confidences": confidences,
                       "is_predictions": is_predictions}

    return te_predictions, confidence_dict

# method to predict Agenda labels
def predict_agenda(dataset, model, tokenizer):

    # for Agenda classification evaluation
    labels = []
    is_french = []
    confidences = []

    with torch.no_grad():

        for key in dataset.keys():
            te_premises = []
            te_hypotheses = []
            te_predictions = []

            en_premise = dataset[key]["en_text"]
            fr_premise = dataset[key]["fr_text"]
            agenda_labels = dataset[key]["labels"]

            if len(agenda_labels) > 0: # check that we have specified labels
                en_hypotheses, fr_hypotheses = get_hypotheses(agenda_labels)
                en_neg_hypotheses, fr_neg_hypotheses = get_negative_hypotheses(en_hypotheses, fr_hypotheses)

                french_flag = False # ENGLISH: all positive examples that entail
                for en_hypothesis in en_hypotheses:
                    te_premises.append(en_premise)
                    te_hypotheses.append(en_hypothesis)

                # ENGLISH: all positive examples that do not entail
                for en_neg_hypothesis in en_neg_hypotheses:
                    te_premises.append(en_premise)
                    te_hypotheses.append(en_neg_hypothesis)

                # predict french premises and hypotheses and convert to agenda classes
                te_predictions, agenda_confidence = predict(te_premises, te_hypotheses, model, tokenizer)

                dataset[key]["prediction"] = dict()
                dataset[key]["prediction"]["en"] = agenda_confidence

                confidences.append(agenda_confidence)
                is_french.append(french_flag)
                labels.append(agenda_labels)

                # reset these lists
                te_premises = []
                te_hypotheses = []
                te_predictions = []

                french_flag = True # FRENCH: all positive examples that entail
                for fr_hypothesis in fr_hypotheses:
                    te_premises.append(fr_premise)
                    te_hypotheses.append(fr_hypothesis)

                # FRENCH: all positive examples that do not entail
                for fr_neg_hypothesis in fr_neg_hypotheses:
                    te_premises.append(fr_premise)
                    te_hypotheses.append(fr_neg_hypothesis)

                # predict french premises and hypotheses and convert to agenda classes
                te_predictions, agenda_confidence = predict(te_premises, te_hypotheses, model, tokenizer)
                dataset[key]["prediction"]["fr"] = agenda_confidence

                confidences.append(agenda_confidence)
                is_french.append(french_flag)
                labels.append(agenda_labels)

    return dataset, labels, confidences, is_french

### SCRIPT ###
base_dir = "../../"

# run predictions using the 3 trained models on the 3 test sets
for run in ["R1", "R2", "R3"]:

    # output file paths for test files
    en_bert_agenda_test_file = base_dir + "evaluation/" + run + "/agenda_rte_en_bert_test.json"
    bi_mbert_agenda_test_file = base_dir + "evaluation/" + run + "/agenda_rte_bi_mbert_test.json"
    
    mbert_agenda_test_file = base_dir + "evaluation/" + run + "/agenda_mbert_test.json"
    bert_agenda_test_file = base_dir + "evaluation/" + run + "/agenda_bert_test.json"
    
    en_bert_rte_agenda_test_file = base_dir + "evaluation/" + run + "/rte_en_bert_test.json"
    bi_mbert_rte_agenda_test_file = base_dir + "evaluation/" + run + "/rte_bi_mbert_test.json"
    
    # output file paths for dev files
    en_bert_agenda_dev_file = base_dir + "evaluation/" + run + "/agenda_rte_en_bert_dev.json"
    bi_mbert_agenda_dev_file = base_dir + "evaluation/" + run + "/agenda_rte_bi_mbert_dev.json"
    
    mbert_agenda_dev_file = base_dir + "evaluation/" + run + "/agenda_mbert_dev.json"
    bert_agenda_dev_file = base_dir + "evaluation/" + run + "/agenda_bert_dev.json"
    
    en_bert_rte_agenda_dev_file = base_dir + "evaluation/" + run + "/rte_en_bert_dev.json"
    bi_mbert_rte_agenda_dev_file = base_dir + "evaluation/" + run + "/rte_bi_mbert_dev.json"
    
    # trained models file paths
    en_bert_file_path = base_dir + "models/" + run + "/AGENDA-RTE-EN-BERT/"
    bi_mbert_file_path = base_dir + "models/" + run + "/AGENDA-RTE-BI-MBERT/"
    
    mbert_file_path = base_dir + "models/" + run + "/AGENDA-MBERT/"
    bert_file_path = base_dir + "models/" + run + "/AGENDA-BERT/"
    
    en_bert_rte_file_path = base_dir + "models/RTE-EN-BERT/"
    bi_mbert_rte_file_path = base_dir + "models/RTE-BI-MBERT/"
    
    # run prediction for all models
    for model_choice in ["en_bert", "bi_mbert", "bert", "mbert", "rte_en_bert", "rte_bi_mbert"]:
    
        if model_choice == "en_bert":
            model_file_path = en_bert_file_path
            output_dev_file_path = en_bert_agenda_dev_file
            output_test_file_path = en_bert_agenda_test_file
        
        elif model_choice == "bi_mbert":
            model_file_path = bi_mbert_file_path
            output_dev_file_path = bi_mbert_agenda_dev_file
            output_test_file_path = bi_mbert_agenda_test_file
        
        elif model_choice == "mbert":
            model_file_path = mbert_file_path
            output_dev_file_path = mbert_agenda_dev_file
            output_test_file_path = mbert_agenda_test_file
        
        elif model_choice == "bert":
            model_file_path = bert_file_path
            output_dev_file_path = bert_agenda_dev_file
            output_test_file_path = bert_agenda_test_file
        
        elif model_choice == "rte_en_bert":
            model_file_path = en_bert_rte_file_path
            output_dev_file_path = en_bert_rte_agenda_dev_file
            output_test_file_path = en_bert_rte_agenda_test_file
        
        elif model_choice == "rte_bi_mbert":
            model_file_path = bi_mbert_rte_file_path
            output_dev_file_path = bi_mbert_rte_agenda_dev_file
            output_test_file_path = bi_mbert_rte_agenda_test_file
        
        # input file paths
        agenda_dev_file = base_dir + "data/" + run + "/dev.csv"
        agenda_test_file = base_dir + "data/" + run + "/test.csv"
        
        # load data
        dev_data = load_csv_as_dict(agenda_dev_file)
        test_data = load_csv_as_dict(agenda_test_file)
        
        print("Loading model: ", model_file_path)
        
        # bert-base-cased or bert-base-multilingual-cased
        tokenizer = BertTokenizer.from_pretrained(model_file_path)
        model = BertForSequenceClassification.from_pretrained(model_file_path).to("cuda:0")
        model.eval() # lock model in eval mode
        
        print("Predicting...")
        
        # run prediction on dev set
        dev_predictions, _, _, _ = predict_agenda(dev_data, model, tokenizer)
        
        # save dev predictions
        os.makedirs(os.path.dirname(output_dev_file_path), exist_ok = True)
        save_json(dev_predictions, output_dev_file_path)
        
        # run prediction on test set
        test_predictions, labels, confidences, is_french = predict_agenda(test_data, model, tokenizer)
        
        # save test predictions
        os.makedirs(os.path.dirname(output_test_file_path), exist_ok = True)
        save_json(test_predictions, output_test_file_path)

# print("Evaluating model...")

# # encode ground truth as 1-hot vectors
# one_hot_labels = []
# for agenda_labels in labels:
#     one_hot_label = [0] * len(ordered_agenda_labels)

#     for label in agenda_labels:
#         one_hot_label[label_to_index[label]] = 1

#     one_hot_labels.append(one_hot_label)

# # use default threshold for final evaluation report
# one_hot_predictions = []
# for confidence in confidences:
#     probabilities = confidence["confidences"]
#     is_predictions = confidence["is_predictions"]
#     one_hot_prediction = [0] * len(ordered_agenda_labels)

#     for i in range(len(probabilities)):
#         if is_predictions[i]:
#             one_hot_prediction[i] = 1

#     one_hot_predictions.append(one_hot_prediction)

# # classification report for Agenda multi-label eval using default threshold
# report = classification_report(one_hot_labels, one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

# # create output report file
# output_file = base_dir + "evaluation/" + model_choice + "_eval_agenda/" + model_choice + "_agenda_report.txt"
# os.makedirs(os.path.dirname(output_file), exist_ok = True)

# # write overal report
# with open(output_file, "w") as fp:
#     fp.write("\n\nAGENDA EVALUATION RESULTS: \n")
#     fp.write("\t" + report + "\n")

# # get english subset report
# if len(set(is_french)) == 2:
#     en_one_hot_labels = [one_hot_labels[i] for i in range(len(one_hot_labels)) if not is_french[i]]
#     en_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if not is_french[i]]

#     # english subset classification reports
#     report = classification_report(en_one_hot_labels, en_one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

#     # write english report
#     with open(output_file, "a") as fp:
#         fp.write("\n\nENGLISH SUBSET AGENDA EVALUATION RESULTS: \n")
#         fp.write("\t" + report + "\n")

# # get french subset report
# if len(set(is_french)) == 2:
#     fr_one_hot_labels = [one_hot_labels[i] for i in range(len(one_hot_labels)) if is_french[i]]
#     fr_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if is_french[i]]

#     # french subset classification reports
#     report = classification_report(fr_one_hot_labels, fr_one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

#     # write french report
#     with open(output_file, "a") as fp:
#         fp.write("\n\nFRENCH SUBSET AGENDA EVALUATION RESULTS: \n")
#         fp.write("\t" + report + "\n")

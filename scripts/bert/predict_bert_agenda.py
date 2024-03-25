# -*- coding: utf-8 -*-

import os
import sys
import csv
import ast
import json
import torch
from sklearn.metrics import classification_report
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

# define string for no agenda
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

# method to predict agenda labels only
def predict(text, model, tokenizer, threshold = 0.500001):

    # encode text
    inputs = tokenizer(text,
                       padding = True,
                       max_length = 512,
                       truncation = True,
                       return_tensors = 'pt').to("cuda:0")

    is_predictions = [False] * len(index_to_label) # for agenda classification

    with torch.no_grad():
        logits = model(**inputs).logits

        # not softmax, need independent probability of class being true
        probability = torch.sigmoid(logits).squeeze().tolist()

        for i in range(len(probability)):
            if probability[i] >= threshold:
                is_predictions[i] = True

        # Nothing exceeds threshold
        if True not in is_predictions:
            # default agenda label if confidence is low
            probability[label_to_index[other_none_label]] = threshold
            is_predictions[label_to_index[other_none_label]] = True

        # package confidences into a neat dict
        confidence_dict = {"confidences": probability,
                           "is_predictions": is_predictions}

    return confidence_dict

# method to predict Agenda labels
def predict_agenda(dataset, model, tokenizer):

    # for Agenda classification evaluation
    labels = []
    is_french = []
    confidences = []
    with torch.no_grad():

        for key in dataset.keys():
            agenda_labels = dataset[key]["labels"]

            if "fr_text" in dataset[key]:
                french_flag = True
                text = dataset[key]["fr_text"]

                agenda_confidence = predict(text, model, tokenizer)
                confidences.append(agenda_confidence)
                is_french.append(french_flag)
                labels.append(agenda_labels)

                dataset[key]["prediction"] = dict()
                dataset[key]["prediction"]["fr"] = agenda_confidence

            if "en_text" in dataset[key]:
                french_flag = False
                text = dataset[key]["en_text"]

                agenda_confidence = predict(text, model, tokenizer)
                confidences.append(agenda_confidence)
                is_french.append(french_flag)
                labels.append(agenda_labels)

                dataset[key]["prediction"]["en"] = agenda_confidence

    return dataset, labels, confidences, is_french

### SCRIPT ###
base_dir = "../../"

# run predictions using the 3 trained models on the 3 test sets
for run in ["R1", "R2", "R3"]:

    # output file paths for test file
    bert_agenda_test_file = base_dir + "evaluation/" + run + "/agenda_mlc_bert_test.json"
    mbert_agenda_test_file = base_dir + "evaluation/" + run + "/agenda_mlc_mbert_test.json"
    
    # output file paths for dev file
    bert_agenda_dev_file = base_dir + "evaluation/" + run + "/agenda_mlc_bert_dev.json"
    mbert_agenda_dev_file = base_dir + "evaluation/" + run + "/agenda_mlc_mbert_dev.json"
    
    # trained models file paths
    bert_file_path = base_dir + "models/" + run + "/AGENDA-MLC-BERT/"
    mbert_file_path = base_dir + "models/" + run + "/AGENDA-MLC-MBERT/"
    
    # run prediction for both models
    for model_choice in ["mbert", "bert"]:
    
        if model_choice == "bert":
            model_file_path = bert_file_path
            output_dev_file_path = bert_agenda_dev_file
            output_test_file_path = bert_agenda_test_file
        
        elif model_choice == "mbert":
            model_file_path = mbert_file_path
            output_dev_file_path = mbert_agenda_dev_file
            output_test_file_path = mbert_agenda_test_file
        
        # input file paths
        agenda_dev_file = base_dir + "data/" + run + "/dev.csv"
        agenda_test_file = base_dir + "data/" + run + "/test.csv"
        
        # load data
        dev_data = load_csv_as_dict(agenda_dev_file)
        test_data = load_csv_as_dict(agenda_test_file)
        
        print("Loading model: ", model_file_path)
        
        tokenizer = BertTokenizer.from_pretrained(model_file_path)
        model = BertForSequenceClassification.from_pretrained(model_file_path,
                                                              problem_type = "multi_label_classification").to("cuda:0")
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

# # prepare prediction for multi-label evaluation
# one_hot_predictions = []
# for confidence in confidences:
#     is_predictions = confidence["is_predictions"]
#     one_hot_prediction = [0] * len(ordered_agenda_labels)

#     for i in range(len(is_predictions)):
#         if is_predictions[i]:
#             one_hot_prediction[i] = 1

#     one_hot_predictions.append(one_hot_prediction)

# # classification reports for Agenda multi-label eval with default threshold
# report = classification_report(one_hot_labels, one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

# output_file = base_dir + "evaluation/" + model_choice + "_eval_agenda/" + model_choice + "_agenda_report.txt"
# os.makedirs(os.path.dirname(output_file), exist_ok = True)

# with open(output_file, "w") as fp:
#     fp.write("\n\nAGENDA EVALUATION RESULTS: \n")
#     fp.write("\t" + report + "\n")

# # get english subset
# if len(set(is_french)) == 2:
#     en_one_hot_labels = [one_hot_labels[i] for i in range(len(one_hot_labels)) if not is_french[i]]
#     en_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if not is_french[i]]

#     # english subset classification reports
#     report = classification_report(en_one_hot_labels, en_one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

#     with open(output_file, "a") as fp:
#         fp.write("\n\nENGLISH SUBSET AGENDA EVALUATION RESULTS: \n")
#         fp.write("\t" + report + "\n")

# # get french subset
# if len(set(is_french)) == 2:
#     fr_one_hot_labels = [one_hot_labels[i] for i in range(len(one_hot_labels)) if is_french[i]]
#     fr_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if is_french[i]]

#     # french subset classification reports
#     report = classification_report(fr_one_hot_labels, fr_one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

#     with open(output_file, "a") as fp:
#         fp.write("\n\nFRENCH SUBSET AGENDA EVALUATION RESULTS: \n")
#         fp.write("\t" + report + "\n")

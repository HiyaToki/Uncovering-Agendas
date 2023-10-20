# -*- coding: utf-8 -*-

import os
import sys
import csv
import ast
import json
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append('../../scripts')
import agenda_label_maps

type_to_label = agenda_label_maps.type_to_label
label_to_index = agenda_label_maps.label_to_index
index_to_label = agenda_label_maps.index_to_label
hypothesis_to_label = agenda_label_maps.hypothesis_to_label
label_to_hypotheses = agenda_label_maps.label_to_hypotheses
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
def predict(text, model, tokenizer):

    # tokenize text
    inputs = tokenizer(text, return_tensors = "pt", truncation = True, max_length = 512).to("cuda:0")
    is_predictions = [False] * len(index_to_label) # for agenda classification
    confidences = [0.0] * len(index_to_label)

    # generate tokens
    tokens = model.generate(**inputs)[0]
    predicted_tokens = tokenizer.decode(tokens, skip_special_tokens = True)

    predictions = [] # process predictions
    for predicted_token in predicted_tokens.split(", "):
        for token in predicted_token.split("/"):

            # edit these because we do not generate strings to match the entire label
            if "online" in token.lower() or "solidarity" in token.lower():
                predictions.append("Online Solidarity")

            # also, order of checks here matter...
            elif "noncooperation" in token.lower() or "disengagement" in token.lower():
                predictions.append("Noncooperation/Disengagement")

            elif "cooperation" in token.lower() or "engagement" in token.lower():
                predictions.append("Political Cooperation/Engagement")

            elif "nonviolent" in token.lower() or "demonstration" in token.lower():
                predictions.append("Nonviolent Demonstration")

            elif "violent" in token.lower() or "action" in token.lower():
                predictions.append("Violent Action")

            else:
                predictions.append(other_none_label)

    # look for labels
    for label in predictions:
        if label in label_to_index:
            index = label_to_index[label]
            confidences[index] = 0.500001
            is_predictions[index] = True

    if True not in is_predictions:
        print("Coulnd not match prediction to labels: ", label)
        print("Generated tokens: ", predicted_tokens)
        print("Adding default prediction...\n")

        # default agenda label if model predicts nonsense strings
        confidences[label_to_index[other_none_label]] = 0.500001
        is_predictions[label_to_index[other_none_label]] = True

    # package confidences into a neat dict
    confidence_dict = {"confidences": confidences,
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
                source = dataset[key]["fr_text"]

                agenda_confidence = predict(source, model, tokenizer)
                confidences.append(agenda_confidence)
                is_french.append(french_flag)
                labels.append(agenda_labels)

                dataset[key]["prediction"] = dict()
                dataset[key]["prediction"]["fr"] = agenda_confidence

            if "en_text" in dataset[key]:
                french_flag = False
                source = dataset[key]["en_text"]

                agenda_confidence = predict(source, model, tokenizer)
                confidences.append(agenda_confidence)
                is_french.append(french_flag)
                labels.append(agenda_labels)

                dataset[key]["prediction"]["en"] = agenda_confidence

    return dataset, labels, confidences, is_french

### SCRIPT ###
base_dir = "../../"

# output file paths for test set
t5_agenda_test_file = base_dir + "evaluation/mlc_agenda_t5_test.json"
mt5_agenda_test_file = base_dir + "evaluation/mlc_agenda_mt5_test.json"

# output file paths for dev set
t5_agenda_dev_file = base_dir + "evaluation/mlc_agenda_t5_dev.json"
mt5_agenda_dev_file = base_dir + "evaluation/mlc_agenda_mt5_dev.json"

# trained models file paths
t5_file_path = base_dir + "/models/MLC-AGENDA-T5/"
mt5_file_path = base_dir + "/models/MLC-AGENDA-MT5/"

# switcher
model_choice = "mt5"
model_file_path = t5_file_path
output_dev_file_path = t5_agenda_dev_file
output_test_file_path = t5_agenda_test_file

if model_choice == "t5":
    model_file_path = t5_file_path
    output_dev_file_path = t5_agenda_dev_file
    output_test_file_path = t5_agenda_test_file

elif model_choice == "mt5":
    model_file_path = mt5_file_path
    output_dev_file_path = mt5_agenda_dev_file
    output_test_file_path = mt5_agenda_test_file

# input file paths
agenda_dev_file = base_dir + "data/dev.csv"
agenda_test_file = base_dir + "data/test.csv"

# load data
dev_data = load_csv_as_dict(agenda_dev_file)
test_data = load_csv_as_dict(agenda_test_file)

print("Loading model: ", model_file_path)

tokenizer = AutoTokenizer.from_pretrained(model_file_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_file_path).to("cuda:0")
model.eval() # lock model in eval mode

print("Predicting...")

# run prediction on dev set
dev_predictions, _, _, _ = predict_agenda(dev_data, model, tokenizer)

# save dev predictions
save_json(dev_predictions, output_dev_file_path)

# run prediction on test set
test_predictions, labels, confidences, is_french = predict_agenda(test_data, model, tokenizer)

# save test predictions
save_json(test_predictions, output_test_file_path)

print("Evaluating model...")

# encode ground truth as 1-hot vectors
one_hot_labels = []
for agenda_labels in labels:
    one_hot_label = [0] * len(ordered_agenda_labels)

    for label in agenda_labels:
        one_hot_label[label_to_index[label]] = 1

    one_hot_labels.append(one_hot_label)

# prepare prediction for multi-label evaluation
one_hot_predictions = []
for confidence in confidences:
    is_predictions = confidence["is_predictions"]
    one_hot_prediction = [0] * len(ordered_agenda_labels)

    for i in range(len(is_predictions)):
        if is_predictions[i]:
            one_hot_prediction[i] = 1

    one_hot_predictions.append(one_hot_prediction)

# classification reports for Agenda multi-label eval
report = classification_report(one_hot_labels, one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

output_file = base_dir + "evaluation/" + model_choice + "_eval_agenda/" + model_choice + "_agenda_report.txt"
os.makedirs(os.path.dirname(output_file), exist_ok = True)

with open(output_file, "w") as fp:
    fp.write("\n\nAGENDA EVALUATION RESULTS: \n")
    fp.write("\t" + report + "\n")

# get english subset
if len(set(is_french)) == 2:
    en_one_hot_labels = [one_hot_labels[i] for i in range(len(one_hot_labels)) if not is_french[i]]
    en_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if not is_french[i]]

    # french subset classification reports
    report = classification_report(en_one_hot_labels, en_one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

    with open(output_file, "a") as fp:
        fp.write("\n\nENGLISH SUBSET AGENDA EVALUATION RESULTS: \n")
        fp.write("\t" + report + "\n")

# get french subset
if len(set(is_french)) == 2:
    fr_one_hot_labels = [one_hot_labels[i] for i in range(len(one_hot_labels)) if is_french[i]]
    fr_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if is_french[i]]

    # french subset classification reports
    report = classification_report(fr_one_hot_labels, fr_one_hot_predictions, zero_division = 0, target_names = ordered_agenda_labels)

    with open(output_file, "a") as fp:
        fp.write("\n\nFRENCH SUBSET AGENDA EVALUATION RESULTS: \n")
        fp.write("\t" + report + "\n")

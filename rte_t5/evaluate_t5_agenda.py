# -*- coding: utf-8 -*-

import os
import ast
import csv
import sys
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

# method to get token id for our labels
def get_label_token_ids(tokenizer):
    positive_id = tokenizer.convert_tokens_to_ids("positive")
    negative_id = tokenizer.convert_tokens_to_ids("negative")

    # mT5 has more than one "positive" and "negative" tokens, these codes are what are actually generated
    # print("token:", positive_id, "decodes: ", tokenizer.decode(positive_id)) # generated code: 18205
    # print("token:", negative_id, "decodes: ", tokenizer.decode(negative_id)) # generated code: 259

    # mT5 is funny this way, so take alternate token ids...
    if "mt5" in model_choice:
        positive_id = 18205
        negative_id = 259

    return positive_id, negative_id

# method to predict Textual Entailment and extract confidence scores, and interpret agenda classes
def predict(texts, hypotheses, model, tokenizer):

    positive_id, negative_id = get_label_token_ids(tokenizer)
    start_token = [[tokenizer.pad_token_id]] * len(texts)

    # pad token to start generation, this is always the same
    decoder_input_ids = torch.tensor(start_token).to("cuda:0")

    # tokenize and encode text
    inputs = tokenizer(texts,
                       padding = True,
                       max_length = 512,
                       truncation = True,
                       return_tensors = "pt").to("cuda:0")

    te_predictions = [] # for rte-style evaluation
    is_predictions = [False] * len(index_to_label) # for agenda classification
    confidences = [0.0] * len(index_to_label)

    with torch.no_grad():

        # pass text input through the model
        logits = model(**inputs, decoder_input_ids = decoder_input_ids).logits

        for i in range(len(logits)):

            # decode the token with maximum log-likelihood
            token = torch.argmax(logits[i], dim = -1)
            prediction = tokenizer.decode(token, skip_special_tokens = True)

            # convert rte predictions to agenda
            is_prediction_output = False
            if prediction == "positive":
                is_prediction_output = True

            elif prediction == "": # mT5 generates empty string after RTE training?
                is_prediction_output = False
                prediction = "negative"

            te_predictions.append(prediction)

            # get agenda label
            label = hypothesis_to_label[hypotheses[i]]
            index = label_to_index[label]

            is_predictions[index] = is_prediction_output

            # look at the logits to estimate confidence score
            # only take the logits of "positive" and "negative"
            selected_logits = logits[i][:, [positive_id, negative_id]]
            probability = F.softmax(selected_logits[0], dim = 0)
            confidences[index] = round(probability[0].item(), 6)

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
            te_texts = []
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
                    text = "premise: " + en_premise + " hypothesis: " + en_hypothesis
                    te_hypotheses.append(en_hypothesis)
                    te_texts.append(text)

                # ENGLISH: all positive examples that do not entail
                for en_neg_hypothesis in en_neg_hypotheses:
                    text = "premise: " + en_premise + " hypothesis: " + en_neg_hypothesis
                    te_hypotheses.append(en_neg_hypothesis)
                    te_texts.append(text)

                # predict french premises and hypotheses and convert to agenda classes
                te_predictions, agenda_confidence = predict(te_texts, te_hypotheses, model, tokenizer)

                dataset[key]["prediction"] = dict()
                dataset[key]["prediction"]["en"] = agenda_confidence

                # for agenda evaluation
                confidences.append(agenda_confidence)
                is_french.append(french_flag)
                labels.append(agenda_labels)

                # reset these lists
                te_texts = []
                te_hypotheses = []
                te_predictions = []
                french_flag = True # FRENCH: all positive examples that entail
                for fr_hypothesis in fr_hypotheses:
                    text = "premise: " + fr_premise + " hypothesis: " + fr_hypothesis
                    te_hypotheses.append(fr_hypothesis)
                    te_texts.append(text)

                # FRENCH: all positive examples that do not entail
                for fr_neg_hypothesis in fr_neg_hypotheses:
                    text = "premise: " + fr_premise + " hypothesis: " + fr_neg_hypothesis
                    te_hypotheses.append(fr_neg_hypothesis)
                    te_texts.append(text)

                # predict french premises and hypotheses and convert to agenda classes
                te_predictions, agenda_confidence = predict(te_texts, te_hypotheses, model, tokenizer)
                dataset[key]["prediction"]["fr"] = agenda_confidence

                # for agenda evaluation
                confidences.append(agenda_confidence)
                is_french.append(french_flag)
                labels.append(agenda_labels)

    return dataset, labels, confidences, is_french

### SCRIPT ###
base_dir = "../../"

# output file paths for test files
en_t5_agenda_test_file = base_dir + "evaluation/agenda_rte_en_t5_test.json"
bi_mt5_agenda_test_file = base_dir + "evaluation/agenda_rte_bi_mt5_test.json"

mt5_agenda_test_file = base_dir + "evaluation/agenda_mt5_test.json"
t5_agenda_test_file = base_dir + "evaluation/agenda_t5_test.json"

en_t5_rte_agenda_test_file = base_dir + "evaluation/rte_en_t5_test.json"
bi_mt5_rte_agenda_test_file = base_dir + "evaluation/rte_bi_mt5_test.json"

# dev files
en_t5_agenda_dev_file = base_dir + "evaluation/agenda_rte_en_t5_dev.json"
bi_mt5_agenda_dev_file = base_dir + "evaluation/agenda_rte_bi_mt5_dev.json"

mt5_agenda_dev_file = base_dir + "evaluation/agenda_mt5_dev.json"
t5_agenda_dev_file = base_dir + "evaluation/agenda_t5_dev.json"

en_t5_rte_agenda_dev_file = base_dir + "evaluation/rte_en_t5_dev.json"
bi_mt5_rte_agenda_dev_file = base_dir + "evaluation/rte_bi_mt5_dev.json"

# trained models file paths
en_t5_file_path = base_dir + "models/AGENDA-RTE-EN-T5/"
bi_mt5_file_path = base_dir + "models/AGENDA-RTE-BI-MT5/"

mt5_file_path = base_dir + "models/AGENDA-MT5/"
t5_file_path = base_dir + "models/AGENDA-T5/"

en_t5_rte_file_path = base_dir + "models/RTE-EN-T5/"
bi_mt5_rte_file_path = base_dir + "models/RTE-BI-MT5/"

# switcher
model_choice = "rte_en_t5"
model_file_path = en_t5_file_path
output_dev_file_path = en_t5_agenda_dev_file
output_test_file_path = en_t5_agenda_test_file

if model_choice == "en_t5":
    model_file_path = en_t5_file_path
    output_dev_file_path = en_t5_agenda_dev_file
    output_test_file_path = en_t5_agenda_test_file

elif model_choice == "bi_mt5":
    model_file_path = bi_mt5_file_path
    output_dev_file_path = bi_mt5_agenda_dev_file
    output_test_file_path = bi_mt5_agenda_test_file

elif model_choice == "mt5":
    model_file_path = mt5_file_path
    output_dev_file_path = mt5_agenda_dev_file
    output_test_file_path = mt5_agenda_test_file

elif model_choice == "t5":
    model_file_path = t5_file_path
    output_dev_file_path = t5_agenda_dev_file
    output_test_file_path = t5_agenda_test_file

elif model_choice == "rte_en_t5":
    model_file_path = en_t5_rte_file_path
    output_dev_file_path = en_t5_rte_agenda_dev_file
    output_test_file_path = en_t5_rte_agenda_test_file

elif model_choice == "rte_bi_mt5":
    model_file_path = bi_mt5_rte_file_path
    output_dev_file_path = bi_mt5_rte_agenda_dev_file
    output_test_file_path = bi_mt5_rte_agenda_test_file

# input file path
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

# use default threshold for final evaluation report
one_hot_predictions = []
for confidence in confidences:
    probabilities = confidence["confidences"]
    is_predictions = confidence["is_predictions"]
    one_hot_prediction = [0] * len(ordered_agenda_labels)

    for i in range(len(probabilities)):
        if is_predictions[i]:
            one_hot_prediction[i] = 1

    one_hot_predictions.append(one_hot_prediction)

# classification report for Agenda multi-label eval using optimal threshold
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

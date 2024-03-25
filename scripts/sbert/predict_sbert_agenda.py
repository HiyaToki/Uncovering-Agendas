# -*- coding: utf-8 -*-

import os
import sys
import csv
import ast
import json
import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util

# set path and import map file
sys.path.append('../../scripts')
import agenda_label_maps

type_to_label = agenda_label_maps.type_to_label
label_to_index = agenda_label_maps.label_to_index
index_to_label = agenda_label_maps.index_to_label
hypothesis_to_label = agenda_label_maps.hypothesis_to_label
label_to_hypotheses = agenda_label_maps.label_to_hypotheses
ordered_agenda_labels = agenda_label_maps.ordered_agenda_labels

en_hypotheses = agenda_label_maps.en_hypotheses
fr_hypotheses = agenda_label_maps.fr_hypotheses

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

# method to get SBERT predictions based on COS-SIM
def predict(text, hypotheses):
    is_predictions = [False] * len(index_to_label) # for agenda classification

    # compute cos-sim scores
    confidences = util.cos_sim(text, hypotheses)[0]
    index_of_best_cos_sim = np.argmax(confidences)
    is_predictions[index_of_best_cos_sim] = True

    # package confidences into a neat dict
    confidence_dict = {"confidences": confidences.tolist(),
                       "is_predictions": is_predictions}

    return confidence_dict

# method to invoke model and loop over texts
def predict_agenda(dataset, model, mode = "hypotheses"):

    keys = []
    texts = []
    labels = []
    is_french = []
    for key in dataset.keys():
        agenda_labels = dataset[key]["labels"]

        if "fr_text" in dataset[key]:
            french_flag = True
            text = dataset[key]["fr_text"]

            keys.append(key)
            texts.append(text)
            labels.append(agenda_labels)
            is_french.append(french_flag)

        if "en_text" in dataset[key]:
            french_flag = False
            text = dataset[key]["en_text"]

            keys.append(key)
            texts.append(text)
            labels.append(agenda_labels)
            is_french.append(french_flag)

    # start by embedding the text
    text_embeddings = model.encode(texts, convert_to_numpy  = True)

    # then embed the hypotheses that are in natural language
    fr_hypotheses_embeddings = model.encode(fr_hypotheses, convert_to_numpy  = True)

    en_hypotheses_embeddings = model.encode(en_hypotheses, convert_to_numpy  = True)

    # also use labels for comparison baseline
    agenda_labels_embeddings = model.encode(ordered_agenda_labels, convert_to_numpy  = True)

    confidences = []
    for i in range(len(texts)):

        text_emb = text_embeddings[i]
        key = keys[i]

        if is_french[i]:
            # use cosine similarity to find correct prediction
            if mode == "hypotheses":
                agenda_confidence = predict(text_emb, fr_hypotheses_embeddings)

            elif mode == "labels": # predict with labels only
                agenda_confidence = predict(text_emb, agenda_labels_embeddings)

            confidences.append(agenda_confidence)

            # map result to dataset
            if "prediction" not in dataset[key]:
                dataset[key]["prediction"] = dict()
            dataset[key]["prediction"]["fr"] = agenda_confidence

        else:
            # compare with english hypotheses
            # use cosine similarity to find correct prediction
            if mode == "hypotheses":
                agenda_confidence = predict(text_emb, en_hypotheses_embeddings)

            elif mode == "labels": # predict with labels only
                agenda_confidence = predict(text_emb, agenda_labels_embeddings)

            confidences.append(agenda_confidence)

            # map result to dataset
            if "prediction" not in dataset[key]:
                dataset[key]["prediction"] = dict()
            dataset[key]["prediction"]["en"] = agenda_confidence

    return dataset, labels, confidences, is_french

### SCRIPT ###
base_dir = "../../"

# run predictions using the 3 test sets (this is 0-shot)
for run in ["R1", "R2", "R3"]:

    # output file paths for test set
    sbert_hypotheses_agenda_test_file = base_dir + "evaluation/" + run + "/ss_sbert_hypotheses_test.json"
    sbert_labels_agenda_test_file = base_dir + "evaluation/" + run + "/ss_sbert_labels_test.json"
    
    msbert_hypotheses_agenda_test_file = base_dir + "evaluation/" + run + "/ss_msbert_hypotheses_test.json"
    msbert_labels_agenda_test_file = base_dir + "evaluation/" + run + "/ss_msbert_labels_test.json"
    
    # output file paths for dev set
    sbert_hypotheses_agenda_dev_file = base_dir + "evaluation/" + run + "/ss_sbert_hypotheses_dev.json"
    sbert_labels_agenda_dev_file = base_dir + "evaluation/" + run + "/ss_sbert_labels_dev.json"
    
    msbert_hypotheses_agenda_dev_file = base_dir + "evaluation/" + run + "/ss_msbert_hypotheses_dev.json"
    msbert_labels_agenda_dev_file = base_dir + "evaluation/" + run + "/ss_msbert_labels_dev.json"
    
    # for each hypothesis or label format
    for mode in ["hypotheses", "labels"]:
        
        # for multi-lingual and english only models
        for model_choice in ["msbert", "sbert"]:
        
            if model_choice == "sbert":
                model_file_path = "all-mpnet-base-v2"
            
                if mode == "hypotheses":
                    output_dev_file_path = sbert_hypotheses_agenda_dev_file
                    output_test_file_path = sbert_hypotheses_agenda_test_file
            
                elif mode == "labels":
                    output_dev_file_path = sbert_labels_agenda_dev_file
                    output_test_file_path = sbert_labels_agenda_test_file
            
            elif model_choice == "msbert":
                model_file_path = "paraphrase-multilingual-mpnet-base-v2"
                if mode == "hypotheses":
                    output_dev_file_path = msbert_hypotheses_agenda_dev_file
                    output_test_file_path = msbert_hypotheses_agenda_test_file
            
                elif mode == "labels":
                    output_dev_file_path = msbert_labels_agenda_dev_file
                    output_test_file_path = msbert_labels_agenda_test_file
            
            # input file paths
            agenda_dev_file = base_dir + "data/" + run + "/dev.csv"
            agenda_test_file = base_dir + "data/" + run + "/test.csv"
            
            # load data
            dev_data = load_csv_as_dict(agenda_dev_file)
            test_data = load_csv_as_dict(agenda_test_file)
            
            ## Load selected SBERT model
            print("Loading model: ", model_file_path)
            model = SentenceTransformer(model_file_path)
            
            print("Predicting...")
            
            # run prediction on dev set
            dev_predictions, _, _, _ = predict_agenda(dev_data, model, mode)
            
            # save dev predictions
            os.makedirs(os.path.dirname(output_dev_file_path), exist_ok = True)
            save_json(dev_predictions, output_dev_file_path)
            
            # run prediction on test set
            test_predictions, labels, confidences, is_french = predict_agenda(test_data, model, mode)
            
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

# # encode predictions as 1-hot label
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

# output_file = base_dir + "evaluation/" + model_choice + "_" + mode + "_eval_agenda/" + model_choice + "_" + mode + "_agenda_report.txt"
# os.makedirs(os.path.dirname(output_file), exist_ok = True)

# with open(output_file, "w") as fp:
#     fp.write("\n\nAGENDA EVALUATION RESULTS: \n")
#     fp.write("\t" + report + "\n")

# # get english subset
# if len(set(is_french)) == 2:
#     en_one_hot_labels = [one_hot_labels[i] for i in range(len(one_hot_labels)) if not is_french[i]]
#     en_one_hot_predictions = [one_hot_predictions[i] for i in range(len(one_hot_predictions)) if not is_french[i]]

#     # french subset classification reports
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

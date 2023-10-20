# -*- coding: utf-8 -*-

import sys
import ast
import csv
import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

import logging
logging.disable(logging.WARNING)

# set path and import map file
sys.path.append('../../scripts')
import agenda_label_maps

type_to_label = agenda_label_maps.type_to_label
ordered_en_hypotheses = agenda_label_maps.en_hypotheses
ordered_fr_hypotheses = agenda_label_maps.fr_hypotheses
label_to_hypotheses = agenda_label_maps.label_to_hypotheses

label_to_index = agenda_label_maps.te_label_to_index
index_to_label = agenda_label_maps.te_index_to_label

default_num_neg_examples = 2 # up to 12

# generate hypotheses based on labels
def get_hypotheses(labels):
    en_hypotheses = []
    fr_hypotheses = []

    for label in labels:

        (en_hypothesis, fr_hypothesis) = label_to_hypotheses[label]

        en_hypotheses.append(en_hypothesis)
        fr_hypotheses.append(fr_hypothesis)

    return en_hypotheses, fr_hypotheses

# method to return negative examples for TE
def get_negative_hypotheses(en_hypotheses, fr_hypotheses):
    en_neg_hypotheses = []
    fr_neg_hypotheses = []

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

# method of dataset peprocessing
def preprocess(dataset, tokenizer):
    model_inputs = []

    for key in dataset.keys():
        en_premise = dataset[key]["en_text"]
        fr_premise = dataset[key]["fr_text"]
        labels = dataset[key]["labels"]

        if len(labels) > 0: # check that we have specified labels
            en_hypotheses, fr_hypotheses = get_hypotheses(labels)
            en_neg_hypotheses, fr_neg_hypotheses = get_negative_hypotheses(en_hypotheses, fr_hypotheses)

            # ENGLISH: all positive examples that entail
            for en_hypothesis in en_hypotheses:
                model_input = tokenizer(en_premise, en_hypothesis, truncation = True, max_length = 512)
                model_input["labels"] = torch.tensor(label_to_index["entailment"])
                model_inputs.append(model_input)

            # ENGLISH: get negative examples for the same premise that do not entail
            for i in range(min(default_num_neg_examples * len(en_hypotheses), len(en_neg_hypotheses))):
                model_input = tokenizer(en_premise, en_neg_hypotheses[i], truncation = True, max_length = 512)
                model_input["labels"] = torch.tensor(label_to_index["not_entailment"])
                model_inputs.append(model_input)

            # FRENCH: all positive examples that entail
            for fr_hypothesis in fr_hypotheses:
                model_input = tokenizer(fr_premise, fr_hypothesis, truncation = True, max_length = 512)
                model_input["labels"] = torch.tensor(label_to_index["entailment"])
                model_inputs.append(model_input)

            # FRENCH: get negative examples for the same premise that do not entail
            for i in range(min(default_num_neg_examples * len(fr_hypotheses), len(fr_neg_hypotheses))):
                model_input = tokenizer(fr_premise, fr_neg_hypotheses[i], truncation = True, max_length = 512)
                model_input["labels"] = torch.tensor(label_to_index["not_entailment"])
                model_inputs.append(model_input)

    # shuffle training examples
    random.shuffle(model_inputs)

    return model_inputs

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

### SCRIPT ###

# paths to pre-trained models (not agenda)
pre_trained_models = ["../../models/RTE-EN-BERT/",
                      "../../models/RTE-BI-MBERT/",
                      "bert-base-multilingual-cased",
                      "bert-base-cased"
                      ]

output_models = ["../../models/AGENDA-RTE-EN-BERT/",
                 "../../models/AGENDA-RTE-BI-MBERT/",
                 "../../models/AGENDA-MBERT/",
                 "../../models/AGENDA-BERT/"
                 ]

# load training data
agenda_train_file = "../../data/train.csv"
train_data = load_csv_as_dict(agenda_train_file)

# load model and tokenizer
for i in range(len(pre_trained_models)):
    model_output_directory = output_models[i]
    model_path = pre_trained_models[i]

    print("\nLOADING PRE-TRAINED MODEL FROM: ", model_path)
    model = BertForSequenceClassification.from_pretrained(model_path,  num_labels = 2)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # create data collator and prepare data with the tokenizer
    data_collator = DataCollatorWithPadding(tokenizer, padding = "longest")
    train_dataset = preprocess(train_data, tokenizer)

    args = TrainingArguments(
        output_dir = model_output_directory,
        per_device_train_batch_size = 32,

        overwrite_output_dir = True,
        save_strategy  = "no",

        num_train_epochs = 5,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        seed = 42
    )

    trainer = Trainer(
        args = args,
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        data_collator = data_collator,
    )

    trainer.train()
    trainer.save_model(model_output_directory)
    print("\nSAVING TRAINED MODEL INTO: ", model_output_directory)

    del data_collator
    del tokenizer
    del trainer
    del model
    del args

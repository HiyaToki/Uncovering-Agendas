# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:34:26 2022

@author: KATSIG
"""

import sys
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

# set path and import map file
sys.path.append('../../scripts')
import agenda_label_maps

label_to_index = agenda_label_maps.te_label_to_index
index_to_label = agenda_label_maps.te_index_to_label

# class of dataset peprocessing and tokenizing
def preprocess(dataset, tokenizer):

    model_inputs = []
    for key in dataset.keys():
        if "fr_premise" in dataset[key]:
            premise =  dataset[key]["fr_premise"]
            hypothesis =  dataset[key]["fr_hypothesis"]

        else:
            premise =  dataset[key]["en_premise"]
            hypothesis =  dataset[key]["en_hypothesis"]

        model_input = tokenizer(premise, hypothesis, max_length = 512, truncation = True)
        model_input["labels"] = torch.tensor(label_to_index[dataset[key]["label"]])
        model_inputs.append(model_input)

    return model_inputs

# method to load a json dataset to memory
def load_json(filename):
    print("Loading file: ", filename)
    with open(filename, 'r') as json_file:
        dataset = json.load(json_file)

    return dataset

### SCRIPT ###

pre_trained_models = ["bert-base-cased",
                      "bert-base-multilingual-cased",
                      ]

rte_training_files = ["../../data/pretrain/massive_RTE/train.json",
                      "../../data/pretrain/bilingual_RTE/train.json"
                      ]

output_models = ["../../models/RTE-EN-BERT/",
                 "../../models/RTE-BI-MBERT/"
                 ]


# load model and tokenizer
for i in range(len(pre_trained_models)):
    model_output_directory = output_models[i]
    training_file = rte_training_files[i]
    model_path = pre_trained_models[i]

    print("\nLOADING PRE-TRAINED MODEL FROM: ", model_path)

    # Define pretrained tokenizer and model: bert-base-cased or bert-base-multilingual-cased
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels = 2)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer)

    # load and preprocess training data
    train_dataset = preprocess(load_json(training_file), tokenizer)

    args = TrainingArguments(
        output_dir = model_output_directory,
        per_device_train_batch_size = 32,

        overwrite_output_dir = True,
        save_strategy  = "no",

        num_train_epochs = 5,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        seed = 42
    )

    trainer = Trainer(
        args = args,
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        data_collator = data_collator
    )

    trainer.train()
    trainer.save_model(model_output_directory)
    print("\nSAVING TRAINED MODEL INTO: ", model_output_directory)

    del data_collator
    del tokenizer
    del trainer
    del model
    del args

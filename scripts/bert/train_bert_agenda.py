# -*- coding: utf-8 -*-

import sys
import csv
import ast
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, DefaultDataCollator

import logging
logging.disable(logging.WARNING)

sys.path.append('../../scripts')
import agenda_label_maps

type_to_label = agenda_label_maps.type_to_label
label_to_index = agenda_label_maps.label_to_index
index_to_label = agenda_label_maps.index_to_label

# method of dataset peprocessing
def preprocess(dataset, tokenizer):

    model_inputs = []
    for key in dataset.keys():

        # get one hot vectors of labels
        one_hot_labels = [0.0] * len(label_to_index)
        for label in dataset[key]["labels"]:
            one_hot_labels[label_to_index[label]] = 1.0

        if "fr_premise" in dataset[key]:
            text =  dataset[key]["fr_text"]

            # use max_length, because just True does not work... ??
            model_input = tokenizer(text, max_length = 512, truncation = True, padding = 'max_length')
            model_input["labels"] = torch.tensor(one_hot_labels)

            model_inputs.append(model_input)

        # prepare english training examples
        if "en_text" in dataset[key]:
            text =  dataset[key]["en_text"]

            # use max_length, because just True does not work... ??
            model_input = tokenizer(text, max_length = 512, truncation = True, padding = 'max_length')
            model_input["labels"] = torch.tensor(one_hot_labels)

            model_inputs.append(model_input)

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
base_dir = "../../"

# paths to pre-trained models (not agenda)
pre_trained_models = ["bert-base-multilingual-cased",
                      "bert-base-cased"
                      ]

output_models = ["/AGENDA-MLC-MBERT/",
                 "/AGENDA-MLC-BERT/"
                 ]

# train 3 models on diff train samples
for run in ["R1", "R2", "R3"]:
    
    # load training data
    agenda_train_file = base_dir + "data/" + run + "/train.csv"
    train_data = load_csv_as_dict(agenda_train_file)

    # load model and tokenizer
    for i in range(len(pre_trained_models)):
        model_output_directory = base_dir + "models/" + run + output_models[i]
        model_path = pre_trained_models[i]
    
        print("\nLOADING PRE-TRAINED MODEL FROM: ", model_path)
    
        # load rte pretrained models
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path,
                                                              problem_type = "multi_label_classification",
                                                              num_labels = len(label_to_index),
                                                              id2label = index_to_label,
                                                              label2id = label_to_index,
                                                              )
    
        train_dataset = preprocess(train_data, tokenizer)
        data_collator = DefaultDataCollator()
    
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

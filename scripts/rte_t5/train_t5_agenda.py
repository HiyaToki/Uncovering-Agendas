# -*- coding: utf-8 -*-

import sys
import ast
import csv
import random
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

import logging
logging.disable(logging.WARNING)

# set path and import map file
sys.path.append('../../scripts')
import agenda_label_maps

type_to_label = agenda_label_maps.type_to_label
ordered_en_hypotheses = agenda_label_maps.en_hypotheses
ordered_fr_hypotheses = agenda_label_maps.fr_hypotheses
label_to_hypotheses = agenda_label_maps.label_to_hypotheses

# default number of negative hypotheses
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

# class of dataset peprocessing and tokenizing
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
                source = "premise: " + en_premise + " hypothesis: " + en_hypothesis
                model_input = tokenizer(source, max_length = 512, truncation = True)
                label = tokenizer("positive", max_length = 32, truncation = True)
                model_input["labels"] = label["input_ids"]
                model_inputs.append(model_input)

            # ENGLISH: get negative examples for the same premise that do not entail
            for i in range(min(default_num_neg_examples * len(en_hypotheses), len(en_neg_hypotheses))):
                source = "premise: " + en_premise + " hypothesis: " + en_neg_hypotheses[i]
                model_input = tokenizer(source, max_length = 512, truncation = True)
                label = tokenizer("negative", max_length = 32, truncation = True)
                model_input["labels"] = label["input_ids"]
                model_inputs.append(model_input)

            # FRENCH: all positive examples that entail
            for fr_hypothesis in fr_hypotheses:
                source = "premise: " + fr_premise + " hypothesis: " + fr_hypothesis
                model_input = tokenizer(source, max_length = 512, truncation = True)
                label = tokenizer("positive", max_length = 32, truncation = True)
                model_input["labels"] = label["input_ids"]
                model_inputs.append(model_input)

            # FRENCH: get negative examples for the same premise that do not entail
            for i in range(min(default_num_neg_examples * len(fr_hypotheses), len(fr_neg_hypotheses))):
                source = "premise: " + fr_premise + " hypothesis: " + fr_neg_hypotheses[i]
                model_input = tokenizer(source, max_length = 512, truncation = True)
                label = tokenizer("negative", max_length = 32, truncation = True)
                model_input["labels"] = label["input_ids"]
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
base_dir = "../../"

# paths to pre-trained models (not agenda)
pre_trained_models = [base_dir + "models/RTE-EN-T5/",
                      base_dir + "models/RTE-BI-MT5/",
                      "google/mt5-base",
                      "google/t5-v1_1-base"
                      ]

output_models = ["/AGENDA-RTE-EN-T5/",
                 "/AGENDA-RTE-BI-MT5/",
                 "/AGENDA-MT5/",
                 "/AGENDA-T5/"
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
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)
        train_dataset = preprocess(train_data, tokenizer)
    
        args = Seq2SeqTrainingArguments(
            output_dir = model_output_directory,
            per_device_train_batch_size = 32,
            predict_with_generate = True,
    
            overwrite_output_dir = True,
            save_strategy  = "no",
    
            num_train_epochs = 5,
            learning_rate = 1e-4,
            weight_decay = 0.01,
            seed = 42
        )
    
        trainer = Seq2SeqTrainer(
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

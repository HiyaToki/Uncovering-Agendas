# -*- coding: utf-8 -*-

import sys
import csv
import ast
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

import logging
logging.disable(logging.WARNING)

sys.path.append('../../scripts')
import agenda_label_maps

type_to_label = agenda_label_maps.type_to_label
label_to_index = agenda_label_maps.label_to_index
index_to_label = agenda_label_maps.index_to_label

# class of dataset peprocessing and tokenizing
def preprocess(dataset, tokenizer):
    model_inputs = []
    for key in dataset.keys():

        # prepare french training examples
        if "fr_text" in dataset[key]:
            source =  dataset[key]["fr_text"]
            labels =  dataset[key]["labels"]
            target = ", ".join(labels)

            model_input = tokenizer(source, max_length = 512, truncation = True)
            label = tokenizer(target, max_length = 32, truncation = True)

            model_input["labels"] = label["input_ids"]
            model_inputs.append(model_input)

        # prepare english training examples
        if "en_text" in dataset[key]:
            source =  dataset[key]["en_text"]
            labels =  dataset[key]["labels"]
            target = ", ".join(labels)

            model_input = tokenizer(source, max_length = 512, truncation = True)
            label = tokenizer(target, max_length = 32, truncation = True)

            model_input["labels"] = label["input_ids"]
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

# paths to pre-trained models (not agenda)
pre_trained_models = ["google/t5-v1_1-base",
                      "google/mt5-base"
                      ]

output_models = ["../../models/MLC-AGENDA-T5/",
                 "../../models/MLC-AGENDA-MT5/"
                 ]

# load training data
agenda_train_file = "../../data/train.csv"
train_data = load_csv_as_dict(agenda_train_file)

# load model and tokenizer
for i in range(len(pre_trained_models)):
    model_output_directory = output_models[i]
    model_path = pre_trained_models[i]

    print("\nLOADING PRE-TRAINED MODEL FROM: ", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)
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

# -*- coding: utf-8 -*-

import json
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

# pre-processing to decode the predictions into texts
def postprocess(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# class of dataset peprocessing and tokenizing
def preprocess(dataset, tokenizer):
    model_inputs = []
    for key in dataset.keys():
        if "fr_premise" in dataset[key]:
            premise = dataset[key]["fr_premise"]
            hypothesis = dataset[key]["fr_hypothesis"]

        else:
            premise = dataset[key]["en_premise"]
            hypothesis = dataset[key]["en_hypothesis"]

        target_text = dataset[key]["target"]

        if target_text == "":
            continue

        source_text = "premise: " + premise + " hypothesis: " + hypothesis
        model_input = tokenizer(source_text, max_length = 512, truncation = True)
        label = tokenizer(target_text, max_length = 32, truncation = True)

        model_input["labels"] = label["input_ids"]
        model_inputs.append(model_input)

    return model_inputs

# method to load a json dataset to memory
def load_json(filename):
    print("Loading file: ", filename)
    with open(filename, 'r') as json_file:
        dataset = json.load(json_file)

    return dataset

### SCRIPT ###
base_dir = "../../"

pre_trained_models = ["google/t5-v1_1-base",
                      "google/mt5-base",
                      ]

rte_training_files = [base_dir + "data/pretrain/massive_RTE/train.json",
                      base_dir + "data/pretrain/bilingual_RTE/train.json"
                      ]

output_models = [base_dir + "models/RTE-EN-T5/",
                 base_dir + "models/RTE-BI-MT5/"
                 ]

# load model and tokenizer
for i in range(len(pre_trained_models)):
    model_output_directory = output_models[i]
    training_file = rte_training_files[i]
    model_path = pre_trained_models[i]

    print("\nLOADING PRE-TRAINED MODEL FROM: ", model_path)

    # Define pretrained tokenizer and model: "google/t5-v1_1-base" or "google/mt5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)

    # load and preprocess training data
    train_dataset = preprocess(load_json(training_file), tokenizer)

    args = Seq2SeqTrainingArguments(
        output_dir = model_output_directory,
        per_device_train_batch_size = 32,
        predict_with_generate = True,

        overwrite_output_dir = True,
        save_strategy  = "no",

        num_train_epochs = 5,
        learning_rate = 1e-4,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        seed = 42
    )

    trainer = Seq2SeqTrainer(
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

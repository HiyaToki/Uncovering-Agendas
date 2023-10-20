# -*- coding: utf-8 -*-

import json
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# method to same datasets into json files
def save_json(dataset, filename):
    print("Saving file:\n\t", filename)

    with open(filename, "w") as json_file:
        json.dump(dataset, json_file)

# method to load a json dataset to memory
def load_json(filename):
    print("Loading file: ", filename)
    with open(filename, 'r') as json_file:
        dataset = json.load(json_file)

    return dataset

### SCRIPT ###
base_dir = "../"
massive_rte_train_file = base_dir + "data/pretrain/massive_RTE/train.json"
massive_rte_dev_file = base_dir + "data/pretrain/massive_RTE/dev.json"
massive_rte_test_1_file = base_dir + "data/pretrain/massive_RTE/test.json"
massive_rte_test_2_file = base_dir + "data/pretrain/massive_RTE/unseen.json"

bilingual_rte_train_file = base_dir + "data/pretrain/bilingual_RTE/train.json"
bilingual_rte_dev_file = base_dir + "data/pretrain/bilingual_RTE/dev.json"
bilingual_rte_test_1_file = base_dir + "data/pretrain/bilingual_RTE/test.json"
bilingual_rte_test_2_file = base_dir + "data/pretrain/bilingual_RTE/unseen.json"

# load data
dataset = load_json(massive_rte_train_file)

# prepare matrices for translating
indices = []
premises = []
hypotheses = []

# sample data at 30%
for index in dataset.keys():
    if random.random() > 0.3:
        continue

    else:
        premise = dataset[index]["en_premise"]
        hypothesis = dataset[index]["en_hypothesis"]

        indices.append(index)
        premises.append(premise)
        hypotheses.append(hypothesis)

# load translators
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to("cuda:0")

# generate translations
print("Translating premises...")

batch = []
fr_premises = []
for i in range(len(premises)):
    batch.append(premises[i])

    if len(batch) == 32:
        translations_p = model.generate(**tokenizer(batch, return_tensors = "pt", padding = True).to("cuda:0"))
        text_p = [tokenizer.decode(t, skip_special_tokens = True) for t in translations_p]

        fr_premises.extend(text_p)
        batch = []

if len(batch) >= 1:
    translations_p = model.generate(**tokenizer(batch, return_tensors = "pt", padding = True).to("cuda:0"))
    text_p = [tokenizer.decode(t, skip_special_tokens = True) for t in translations_p]

    fr_premises.extend(text_p)
    batch = []

print("Translating hypothesis...")

batch = []
fr_hypotheses = []
for i in range(len(hypotheses)):
    batch.append(hypotheses[i])

    if len(batch) == 32:
        translations_h = model.generate(**tokenizer(batch, return_tensors = "pt", padding = True).to("cuda:0"))
        text_h = [tokenizer.decode(t, skip_special_tokens = True) for t in translations_h]

        fr_hypotheses.extend(text_h)
        batch = []

if len(batch) >= 1:
    translations_h = model.generate(**tokenizer(batch, return_tensors = "pt", padding = True).to("cuda:0"))
    text_h = [tokenizer.decode(t, skip_special_tokens = True) for t in translations_h]

    fr_hypotheses.extend(text_h)
    batch = []

# sanity check
if len(fr_premises) == len(fr_hypotheses):
    print("Translation completed!")

for i in range(len(indices)):
    index = indices[i]
    fr_premise = fr_premises[i]
    fr_hypothesis = fr_hypotheses[i]

    dataset[index]["fr_premise"] = fr_premise
    dataset[index]["fr_hypothesis"] = fr_hypothesis

save_json(dataset, bilingual_rte_train_file)

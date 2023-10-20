# -*- coding: utf-8 -*-

import json

# method to print longest seq length
def longest(dataset):
    max_size = 0
    for key in dataset.keys():

        text = "premise: " + dataset[key]["en_premise"] \
             + " hypothesis: " + dataset[key]["en_hypothesis"]

        words = text.split(" ")
        if len(words) > max_size:
            max_size = len(words)

    print("Longest sequence length: ", max_size)

# method to map nli labels to rte
def nli2rte(nli_label):
    if nli_label == "neutral" or nli_label == "contradiction":
        return "not_entailment"

    elif nli_label == "entailment":
        return "entailment"

    else: # noob error handling
        return "I saw an unexpected label!"

# method to map rte labels to T5-specific tokens
def rte2t5(rte_label):
    if rte_label == "not_entailment":
        return "negative"

    elif rte_label == "entailment":
        return "positive"

    else: # noob error handling
        return "I saw an unexpected label!"


# method to load nli files into memory
def load_nli(datafile, dataset = dict()):
    print("Loading file:\n\t", datafile)

    with open(datafile, encoding = "utf8") as nli:
        for line in nli:

            data = json.loads(line)
            if "language" in data:
                if data["language"] != "fr":
                    continue

            label = nli2rte(data["gold_label"])
            if label == "I saw an unexpected label!":
                continue

            entry_id = len(dataset)
            dataset[entry_id] = dict()
            dataset[entry_id]["label"] = label
            dataset[entry_id]["target"] = rte2t5(label)
            dataset[entry_id]["en_premise"] = data["sentence1"]
            dataset[entry_id]["en_hypothesis"] = data["sentence2"]

    return dataset

# method to load rte files into memory
def load_rte(datafile, dataset = dict()):
    print("Loading file:\n\t", datafile)

    with open(datafile, encoding = "utf8") as rte:
        for line in rte:

            if "sentence1" in line:
                continue

            data = line.split("\t")
            label = data[3].strip()
            target = rte2t5(label)

            if target == "I saw an unexpected label!":
                continue

            entry_id = len(dataset)
            dataset[entry_id] = dict()
            dataset[entry_id]["label"] = label
            dataset[entry_id]["target"] = target
            dataset[entry_id]["en_premise"] = data[1].strip()
            dataset[entry_id]["en_hypothesis"] = data[2].strip()

    return dataset

# method to same datasets into json files
def save_json(dataset, filename):
    print("Saving file:\n\t", filename)

    with open(filename, "w") as json_file:
        json.dump(dataset, json_file)

## script ##
baser_dir = "../"
snli_train_file = baser_dir + "data/pretrain/SNLI/snli_1.0_train.jsonl"
snli_test_file = baser_dir + "data/pretrain/SNLI/snli_1.0_test.jsonl"
snli_dev_file = baser_dir + "data/pretrain/SNLI/snli_1.0_dev.jsonl"

mnli_train_file = baser_dir + "data/pretrain/MNLI/multinli_1.0_train.jsonl"
mnli_dev_1_file = baser_dir + "data/pretrain/MNLI/multinli_1.0_dev_matched.jsonl"
mnli_dev_2_file = baser_dir + "data/pretrain/MNLI/multinli_1.0_dev_mismatched.jsonl"
mnli_test_1_file = baser_dir + "data/pretrain/MNLI/multinli_1.0_test_matched.jsonl"
mnli_test_2_file = baser_dir + "data/pretrain/MNLI/multinli_1.0_test_mismatched.jsonl"

rte_train_file = baser_dir + "data/pretrain/RTE/train.tsv"
rte_test_file = baser_dir + "data/pretrain/RTE/test.tsv"
rte_dev_file = baser_dir + "data/pretrain/RTE/dev.tsv"

massive_rte_train_file = baser_dir + "data/pretrain/massive_RTE/train.json"
massive_rte_dev_file = baser_dir + "data/pretrain/massive_RTE/dev.json"
massive_rte_test_1_file = baser_dir + "data/pretrain/massive_RTE/test.json"
massive_rte_test_2_file = baser_dir + "data/pretrain/massive_RTE/unseen.json"

# create the massive train file
train = dict()
train = load_rte(rte_train_file, dataset = train)
train = load_nli(snli_train_file, dataset = train)
train = load_nli(mnli_train_file, dataset = train)

# create the massive dev file file
valid = dict()
valid = load_rte(rte_dev_file, dataset = valid)
valid = load_nli(snli_dev_file, dataset = valid)
valid = load_nli(mnli_dev_1_file, dataset = valid)

# create the massive test file file
test = dict()
test = load_nli(snli_test_file, dataset = test)
test = load_nli(mnli_test_1_file, dataset = test)

# use mnli mismatched data for another small test file
unseen = dict()
unseen = load_nli(mnli_test_2_file, dataset = unseen)
unseen = load_nli(mnli_dev_2_file, dataset = unseen)

print("train size: ", len(train))
print("dev size: ", len(valid))
print("test size: ", len(test))
print("unseen size: ", len(unseen))

save_json(train, massive_rte_train_file)
save_json(valid, massive_rte_dev_file)
save_json(test, massive_rte_test_1_file)
save_json(unseen, massive_rte_test_2_file)

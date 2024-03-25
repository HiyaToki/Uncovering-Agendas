# -*- coding: utf-8 -*-

import csv
import ast
import random

# method to save a list as csv
def save_list_as_csv(filename, data):
    with open(filename, 'w', encoding = 'utf-8', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = data[0].keys())
        
        # writer.writeheader()
        for item in data:
            writer.writerow(item)

# method to load a csv dataset to memory
def load_csv_as_list(filename):
    print("Loading file: ", filename)

    dataset = []
    with open(filename, 'r', encoding = 'utf-8') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            doc_id = row[0]
            tweet_id = row[1]
            language = row[2]
            labels = [x for x in ast.literal_eval(row[3])]

            fr_text = row[4]
            en_text = row[5]

            data_item = dict()
            data_item["doc_id"] = doc_id
            data_item["tweet_id"] = tweet_id
            data_item["language"] = language
            data_item["labels"] = labels
            data_item["fr_text"] = fr_text
            data_item["en_text"] = en_text
            
            dataset.append(data_item)

    return dataset

# method to get R2 and R3 splits from original split
def get_splits(dataset, test_size = 102):
    
    random.shuffle(dataset)
    
    # R1 split
    r1_dev = dataset[: test_size]
    r1_test = dataset[test_size: test_size * 2]
    r1_train = dataset[test_size * 2: ]
    
    r1_dev = sorted(r1_dev, key = lambda d: int(d['doc_id']))
    r1_test = sorted(r1_test, key = lambda d: int(d['doc_id']))
    r1_train = sorted(r1_train, key = lambda d: int(d['doc_id']))
    
    # R2 split
    r2_dev = dataset[test_size * 2: test_size * 3]
    r2_test = dataset[test_size * 3: test_size * 4]
    r2_train = dataset[: test_size * 2] + dataset[test_size * 4: ]
    
    r2_dev = sorted(r2_dev, key = lambda d: int(d['doc_id']))
    r2_test = sorted(r2_test, key = lambda d: int(d['doc_id']))
    r2_train = sorted(r2_train, key = lambda d: int(d['doc_id']))
    
    # R3 split
    r3_dev = dataset[test_size * 4: test_size * 5]
    r3_test = dataset[test_size * 5: test_size * 6]
    r3_train = dataset[: test_size * 4] + dataset[test_size * 6: ]
    
    r3_dev = sorted(r3_dev, key = lambda d: int(d['doc_id']))
    r3_test = sorted(r3_test, key = lambda d: int(d['doc_id']))
    r3_train = sorted(r3_train, key = lambda d: int(d['doc_id']))

    return r1_train, r1_test, r1_dev, r2_train, r2_test, r2_dev, r3_train, r3_test, r3_dev

# load annotated agenda dataset
agenda_data_file_path = '../data/agenda_dataset.csv'
dataset = load_csv_as_list(agenda_data_file_path)

# get train/dev/test splits with non-overlapping test sets
r1_train, r1_test, r1_dev, r2_train, r2_test, r2_dev, r3_train, r3_test, r3_dev = get_splits(dataset)

# save splits
save_list_as_csv('../data/R1/train_1.csv', r1_train)
save_list_as_csv('../data/R1/test_1.csv', r1_test)
save_list_as_csv('../data/R1/dev_1.csv', r1_dev)

save_list_as_csv('../data/R2/train.csv', r2_train)
save_list_as_csv('../data/R2/test.csv', r2_test)
save_list_as_csv('../data/R2/dev.csv', r2_dev)

save_list_as_csv('../data/R3/train.csv', r3_train)
save_list_as_csv('../data/R3/test.csv', r3_test)
save_list_as_csv('../data/R3/dev.csv', r3_dev)

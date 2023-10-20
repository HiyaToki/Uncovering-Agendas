# Dataset for Agenda Detection

The train, dev and test split of our annotated Agenda dataset: 

```
train.csv, dev.csv, test.csv 
```

The column title line is ommited from the CSV files. For all the csv files, the title line is:

```
document_id, tweet_id, original_language, labels
```

## Instructions
To obtain the text filed of our dataset, please download the un-tagged Twitter 2022 French Presendential Elections coprus from: https://www.kaggle.com/datasets/jeanmidev/french-presidential-online-listener

Then match the ```tweet_id``` column of our annotated subset with the ```tweet_id``` filed of the downloaded files from Kaggle. Then, use the ```original_language``` column to translate the text into English (or French). For example, if the ```original_language``` is ```French```, then the text should be translated from French to English.

The final CSV files should have a schema like:

```
document_id, tweet_id, original_language, labels, french_text, english_text
```

## Datasets for RTE
The pretrain folder is expected to contain the MNLI, SNLI and RTE datasets (not included, download required). 

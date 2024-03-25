# Dataset for Agenda Detection

The annotated Agenda dataset: 

```
agenda_dataset.csv 
```

The column title line is ommited from the CSV files. For these files, the title line is:

```
document_id,tweet_id,original_language,labels
```

In order to adhere to the content sharing policies of X (formerly known as Twitter), we are not making the textual contents of our dataset public. Please read the instructions below on how to construct the complete dataset.

## Instructions
To obtain the text filed of our dataset, please download the un-tagged Twitter 2022 French Presendential Elections coprus from: https://www.kaggle.com/datasets/jeanmidev/french-presidential-online-listener <br>

Then, match the ```tweet_id``` column of our annotated subset with the ```tweet_id``` field of the downloaded files from Kaggle. The next step is to use the ```original_language``` column to translate the text into English (or French). For example, if the ```original_language``` is ```French```, then the text should be translated from French to English. <br>

The MT models we used for the tweet translation task are: https://huggingface.co/Helsinki-NLP/opus-mt-fr-en and https://huggingface.co/Helsinki-NLP/opus-mt-en-fr <br>

The final CSV files should have the following columns:

```
document_id,tweet_id,original_language,labels,french_text,english_text
```

Or (instead of doing all of the above) message the authors of the paper for a direct link to download the full dataset. <br>

Finally, run `../scripts/create_data_splits.py` to obtain three train/dev/test splits with non-overlapping test sets. <br>

## Datasets for RTE
The pretrain folder is expected to contain the MNLI, SNLI and RTE datasets (not included, download required). 

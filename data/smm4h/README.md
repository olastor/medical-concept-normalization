# SMM4H

This folder contains:

- code to load the smm4h dataset, check and normalize the labels to the pt level using a more recent MedDRA version (MedDRA data archive is not included in this repository, thus you need a subscription to reproduce) and to merge the training files into one tsv file
- code to apply german translations obtained from DeepL API in order to generate german training and test sets
- the resulting training and test data

### Attribution

#### SMM4H

The (SMM4H)-2017 data is licenced under CC BY 4.0. The files of subtask3 are included in this repository for convenience, but no ownership is being claimed. This also applies for the resulting tsv files that contain this dataset with small changes to the format and some changed labels (see code). 

Sarker, Abeed (2018), “Data and systems for medication-related text classification and concept normalization from Twitter: Insights from the Social Media Mining for Health (SMM4H)-2017 shared task”, Mendeley Data, V2, doi: 10.17632/rxwfb3tysd.2

#### DeepL

German translations were automatically created using [DeepL API](https://www.deepl.com/en/docs-api/).


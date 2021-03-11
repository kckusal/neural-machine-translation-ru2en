import torchtext
from torchtext.legacy.data import Field, Dataset, Example, BucketIterator

import spacy
from spacy.lang.ru import Russian

import io

nlp_ru = Russian()
nlp_en = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])


def tokenize_ru(text):
    return [tok.text for tok in nlp_ru.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in nlp_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_ru, include_lengths=True, lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>',
            include_lengths=True, lower=True)

fields = [('rus', SRC), ('eng', TRG)]

# Get the dataset
torchtext.utils.download_from_url(
    'https://github.com/bsbor/data/releases/download/test3/1mcorpus.zip', '1mcorpus.zip')
torchtext.utils.extract_archive('1mcorpus.zip')

ru_lines = io.open("corpus.en_ru.1m.ru", encoding='UTF-8').read().splitlines()
en_lines = io.open("corpus.en_ru.1m.en", encoding='UTF-8').read().splitlines()

dataset_size = 200000
temp_ru_lines = ru_lines[:dataset_size]
temp_en_lines = en_lines[:dataset_size]
sentences = list(zip(temp_ru_lines, temp_en_lines))

data = [Example.fromlist(item, fields) for item in sentences]

data = Dataset(data, fields=fields)
SRC.build_vocab(data)
TRG.build_vocab(data)

import re
import numpy as np
import html
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from wrench.dataset import load_dataset, BaseDataset
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import pdb
import torch
from pathlib import Path
import os
import json


def preprocess_text(text, stop_words=None, stemming="porter"):
    if stop_words is not None:
        stop_words = set(stopwords.words(stop_words))
    else:
        stop_words = set()

    if stemming == "porter":
        stemmer = PorterStemmer()
    elif stemming == "snowball":
        stemmer = SnowballStemmer(language="english")
    else:
        stemmer = None

    processed_tokens = []
    tokens = nltk.word_tokenize(text.lower())
    for token in tokens:
        # filter out stopwords and non-words
        if token in stop_words or (re.search("^\w+$", token) is None):
            continue
        if stemmer is not None:
            token = stemmer.stem(token)

        processed_tokens.append(token)

    processed_text = " ".join(processed_tokens)
    return processed_text


def build_revert_index(dataset, stop_words=None, stemming="porter", max_ngram=1, cache_path=None):
    """
    Build reverted index for the given text dataset
    """
    if cache_path is not None:
        if os.path.exists(cache_path):
            with open(cache_path) as infile:
                reverted_index = json.load(infile)
                for phrase in reverted_index:
                    reverted_index[phrase] = np.array(reverted_index[phrase])

                return reverted_index

    # preprocess data
    corpus = [dataset.examples[idx]["text"] for idx in range(len(dataset))]
    if stop_words is not None:
        stop_words = set(stopwords.words(stop_words))
    else:
        stop_words = set()

    if stemming == "porter":
        stemmer = PorterStemmer()
    elif stemming == "snowball":
        stemmer = SnowballStemmer(language="english")
    else:
        stemmer = None

    reverted_index = {}

    for idx, text in enumerate(corpus):
        processed_tokens = []
        text = re.sub("\ufeff", "", text)
        tokens = nltk.word_tokenize(text.lower())
        for token in tokens:
            # filter out stopwords and non-words
            if token in stop_words or (re.search("^\w+$", token) is None):
                continue
            if stemmer is not None:
                token = stemmer.stem(token)

            processed_tokens.append(token)

        for n in range(max_ngram):
            phrases = ngrams(processed_tokens, n+1)
            for t in phrases:
                phrase = " ".join(t)
                if phrase in reverted_index:
                    if reverted_index[phrase][-1] != idx:
                        reverted_index[phrase].append(idx)
                else:
                    reverted_index[phrase] = [idx]

    if cache_path is not None:
        with open(cache_path, "w") as outfile:
            json.dump(reverted_index, outfile)

    for phrase in reverted_index:
        reverted_index[phrase] = np.array(reverted_index[phrase])

    return reverted_index


def extract_features(sentences, tokenizer, model, batch_size=32):
    # Tokenize all sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Initialize a list to store our sentence embeddings
    all_sentence_embeddings = []

    for i in range(0, len(sentences), batch_size):
        # Prepare batch inputs
        batch_input = {k: v[i:i + batch_size] for k, v in encoded_input.items()}

        # Move batch to the same device as model
        batch_input = {k: v.to(model.device) for k, v in batch_input.items()}

        with torch.no_grad():
            # Get model outputs for the current batch
            outputs = model(**batch_input, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Aggregate the hidden states for the batch, e.g., mean of the last layer's output
            batch_embeddings = torch.mean(hidden_states[-1], dim=1)

            all_sentence_embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings to have a single matrix
    sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0).numpy()

    return sentence_embeddings


def load_wrench_data(data_root, dataset_name, feature, stopwords=None, stemming="porter", max_ngram=1,
                     scalar=None, revert_index=False):
    if feature == "bert":
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,
                                                                  extract_feature=True, extract_fn="bert", cache_name="bert")
    elif feature == "sentence-bert":
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,extract_feature=False)
        cache_path = Path(data_root) / dataset_name / "sentence_bert_cache"
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        train_cache_path = cache_path / "train_features.pkl.npy"
        valid_cache_path = cache_path / "valid_features.pkl.npy"
        test_cache_path = cache_path / "test_features.pkl.npy"

        if os.path.exists(train_cache_path):
            train_dataset.features = np.load(train_cache_path, allow_pickle=True)
            valid_dataset.features = np.load(valid_cache_path, allow_pickle=True)
            test_dataset.features = np.load(test_cache_path, allow_pickle=True)
        else:
            embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
            train_sentences = [train_dataset.examples[i]["text"] for i in range(len(train_dataset))]
            valid_sentences = [valid_dataset.examples[i]["text"] for i in range(len(valid_dataset))]
            test_sentences = [test_dataset.examples[i]["text"] for i in range(len(test_dataset))]
            train_dataset.features = embedding_model.encode(train_sentences)
            valid_dataset.features = embedding_model.encode(valid_sentences)
            test_dataset.features = embedding_model.encode(test_sentences)
            np.save(train_cache_path, train_dataset.features)
            np.save(valid_cache_path, valid_dataset.features)
            np.save(test_cache_path, test_dataset.features)

    elif feature == "bertweet-sentiment":
        cache_path = Path(data_root) / dataset_name / "bertweet_sentiment_cache"
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,extract_feature=False)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        train_cache_path = cache_path / "train_features.npy"
        valid_cache_path = cache_path / "valid_features.npy"
        test_cache_path = cache_path / "test_features.npy"

        if os.path.exists(train_cache_path):
            train_dataset.features = np.load(train_cache_path, allow_pickle=True)
            valid_dataset.features = np.load(valid_cache_path, allow_pickle=True)
            test_dataset.features = np.load(test_cache_path, allow_pickle=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
            model = AutoModelForSequenceClassification.from_pretrained(
                "finiteautomata/bertweet-base-sentiment-analysis")
            train_sentences = [train_dataset.examples[i]["text"] for i in range(len(train_dataset))]
            valid_sentences = [valid_dataset.examples[i]["text"] for i in range(len(valid_dataset))]
            test_sentences = [test_dataset.examples[i]["text"] for i in range(len(test_dataset))]
            train_dataset.features = extract_features(train_sentences, tokenizer, model)
            valid_dataset.features = extract_features(valid_sentences, tokenizer, model)
            test_dataset.features = extract_features(test_sentences, tokenizer, model)
            np.save(train_cache_path, train_dataset.features)
            np.save(valid_cache_path, valid_dataset.features)
            np.save(test_cache_path, test_dataset.features)

    else:
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,
                                                                  extract_feature=True, extract_fn=feature)

    if scalar is not None:
        if scalar == "standard":
            scalar = preprocessing.StandardScaler().fit(train_dataset.features)
            train_dataset.features = scalar.transform(train_dataset.features)
            valid_dataset.features = scalar.transform(valid_dataset.features)
            test_dataset.features = scalar.transform(test_dataset.features)

    if revert_index:
        train_cache_path = Path(data_root) / dataset_name / "train_index.json"
        valid_cache_path = Path(data_root) / dataset_name / "valid_index.json"
        test_cache_path = Path(data_root) / dataset_name / "test_index.json"
        train_dataset.revert_index = build_revert_index(train_dataset, stop_words=stopwords, stemming=stemming,
                                                        max_ngram=max_ngram, cache_path=train_cache_path)
        valid_dataset.revert_index = build_revert_index(valid_dataset, stop_words=stopwords, stemming=stemming,
                                                        max_ngram=max_ngram, cache_path=valid_cache_path)
        test_dataset.revert_index = build_revert_index(test_dataset, stop_words=stopwords, stemming=stemming,
                                                       max_ngram=max_ngram, cache_path=test_cache_path)

    return train_dataset, valid_dataset, test_dataset
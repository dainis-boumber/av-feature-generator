from pathlib import Path
import pickle
import logging

import numpy as np
import pandas as pd
import textacy
import nltk

nltk.download('punkt')
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')
from nltk.tokenize.moses import MosesTokenizer

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from data.MLP400AV.mlpapi import MLPVLoader
from data_helper.DataHelpers import DataHelper
from data_helper.Data import DataObject


class DataBuilderML400(DataHelper):
    problem_name = "ML400"

    def __init__(self, embed_dim, target_doc_len, target_sent_len, doc_as_sent=False, doc_level=True):
        super(DataBuilderML400, self).__init__(embed_dim=embed_dim, target_doc_len=target_doc_len,
                                               target_sent_len=target_sent_len)

        logging.info("setting: %s is %s", "doc_as_sent", doc_as_sent)
        self.doc_as_sent = doc_as_sent
        logging.info("setting: %s is %s", "sent_list_doc", doc_level)
        self.doc_level = doc_level

        self.dataset_dir = self.data_path + 'MLP400AV/'
        self.num_classes = 2  # true or false

        self.tokenizer = None

        print("loading nltk model")
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = MosesTokenizer()
        print("nltk model loaded")

        self.load_all_data()

    def str_2_sent_2_token(self, data, sent_split=True, word_split=False):
        content = "".join(data)
        content = content.replace("\n", " ")
        content = textacy.preprocess_text(content, fix_unicode=True, lowercase=True,
                                          transliterate=True, no_numbers=False, no_contractions=True, no_accents=True)

        if sent_split:
            content_sents = self.sent_detector.tokenize(content)
            content_sents = [s for s in content_sents if len(s) > 20]

            if word_split:
                content_tokens = []
                for sent in content_sents:
                    content_tokens.append(self.tokenizer.tokenize(sent))
                return content_tokens

            else:
                return content_sents
        elif word_split:
            self.tokenizer.tokenize(content)
        else:
            return content

    def clean_text(self, content):
        content = content.replace("\n", " ")
        content = textacy.preprocess_text(content, lowercase=True, no_contractions=True)
        return content

    def proc_data(self, data_raw, label, raw_to_vec, sent_split=True, word_split=False):
        vector_sequences = data_raw.applymap(lambda x: raw_to_vec[x])
        doc_label = np.array([1 if la == "YES" else 0 for la in label])

        logging.info("data shape: " + str(vector_sequences.shape))
        logging.info("label shape: " + str(doc_label.shape))

        if self.doc_as_sent:
            raise NotImplementedError

        data_obj = DataObject(self.problem_name, len(doc_label))
        data_obj.raw = data_raw
        data_obj.label_doc = doc_label
        data_obj.value = vector_sequences
        return data_obj

    def load_dataframe(self):
        data_pickle = Path("av400tuple.pickle")
        if not data_pickle.exists():
            logging.info("loading data structure from RAW")
            loader = MLPVLoader("A2", fileformat='pandas', directory=self.dataset_dir)
            train_data, val_data, test_data = loader.get_mlpv()

            train_y = train_data['label'].tolist()
            val_y = val_data['label'].tolist()
            test_y = test_data['label'].tolist()

            train_data.drop(['k_author', 'u_author', 'label'], axis=1, inplace=True)
            val_data.drop(['k_author', 'u_author', 'label'], axis=1, inplace=True)
            test_data.drop(['k_author', 'u_author', 'label'], axis=1, inplace=True)

            logging.info("load data structure completed")

            pickle.dump([train_data, val_data, test_data, train_y, val_y, test_y], open(data_pickle, mode="wb"))
            logging.info("dumped all data structure in " + str(data_pickle))
        else:
            logging.info("loading data structure from PICKLE")
            [train_data, val_data, test_data, train_y, val_y, test_y] = pickle.load(open(data_pickle, mode="rb"))
            logging.info("load data structure completed")

        return (train_data, train_y), (val_data, val_y), (test_data, test_y)

    def build_embedding_matrix(self):
        embedding_matrix = np.zeros((len(self.vocab) + 1, self.embedding_dim))
        for word, i in self.vocab.items():
            embedding_vector = self.glove_dict.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def load_all_data(self):
        (train_data, train_y), (val_data, val_y), (test_data, test_y) = self.load_dataframe()

        all_data = pd.concat([train_data, val_data, test_data])
        uniq_doc = pd.unique(all_data.values.ravel('K'))

        self.tokenizer = Tokenizer(num_words=self.vocabulary_size)
        self.tokenizer.fit_on_texts(uniq_doc)
        uniq_seq = self.tokenizer.texts_to_sequences(uniq_doc)
        uniq_seq = pad_sequences(uniq_seq, maxlen=65535, padding="post", truncating="post")

        raw_to_vec = dict(zip(uniq_doc, uniq_seq))

        self.train_data = self.proc_data(train_data, train_y, raw_to_vec)
        self.val_data = self.proc_data(val_data, val_y, raw_to_vec)
        self.test_data = self.proc_data(test_data, test_y, raw_to_vec)

        self.vocab = self.tokenizer.word_index
        self.embed_matrix =  self.build_embedding_matrix()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    a = DataBuilderML400(embed_dim=300, target_doc_len=64, target_sent_len=1024,
                         doc_as_sent=False, doc_level=True)

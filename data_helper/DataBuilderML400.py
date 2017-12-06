from pathlib import Path
import pickle
import logging

import numpy as np
from spacy.attrs import ORTH
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

    def proc_data(self, data, sent_split=True, word_split=False):
        raw = []
        label_doc = []

        one_row_train = data['k_doc'].append(data['u_doc'])
        uniq_doc = one_row_train.unique()
        k_tokenizer = Tokenizer(num_words=self.vocabulary_size)
        k_tokenizer.fit_on_texts(uniq_doc)
        vector_sequences = k_tokenizer.texts_to_sequences(uniq_doc)

        # uniq_doc = [self.str_2_sent_2_token(x, sent_split=sent_split, word_split=word_split) for x in uniq_doc]
        # data.applymap(lambda x: self.str_2_sent_2_token(x, sent_split=sent_split, word_split=word_split))

        if self.doc_as_sent:
            raise NotImplementedError

        data = DataObject(self.problem_name, len(raw))
        data.raw = raw
        data.label_doc = label_doc
        return data

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

    def load_all_data(self):
        (train_data, train_y), (val_data, val_y), (test_data, test_y) = self.load_dataframe()

        train = self.proc_data(train_data)
        vali = self.proc_data(val_data)
        test = self.proc_data(test_data)


if __name__ == "__main__":
    a = DataBuilderML400(embed_dim=300, target_doc_len=64, target_sent_len=1024,
                         doc_as_sent=False, doc_level=True)

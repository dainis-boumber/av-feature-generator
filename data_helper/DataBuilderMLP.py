import numpy as np
import logging
import textacy
import nltk

nltk.download('punkt')
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')
from nltk.tokenize.moses import MosesTokenizer

from data.MLP400AV.mlpapi import MLPVLoader
from data_helper.DataHelpers import DataHelper
from data_helper.Data import DataObject


class DataBuilderMLP(DataHelper):
    problem_name = "MLP"

    def __init__(self, embed_dim, target_doc_len, target_sent_len, doc_as_sent=False, doc_level=True):
        super(DataBuilderMLP, self).__init__(embed_dim=embed_dim, target_doc_len=target_doc_len,
                                             target_sent_len=target_sent_len)

        logging.info("setting: %s is %s", "doc_as_sent", doc_as_sent)
        self.doc_as_sent = doc_as_sent
        logging.info("setting: %s is %s", "sent_list_doc", doc_level)
        self.doc_level = doc_level

        self.dataset_dir = self.data_path + 'hotel_balance_LengthFix1_3000per/'
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
        else:
            return content

    def proc_data(self, data, sent_split=True, word_split=False):
        raw = []
        label_doc = []

        for data_tuple in data:
            k, u, label = data_tuple
            k = self.str_2_sent_2_token(k, sent_split=sent_split, word_split=word_split)
            u = self.str_2_sent_2_token(u, sent_split=sent_split, word_split=word_split)
            raw.append((k, u))
            if "YES" == label:
                label_doc.append(True)
            else:
                label_doc.append(False)

        if self.doc_as_sent:
            raise NotImplementedError

        data = DataObject(self.problem_name, len(raw))
        data.raw = raw
        data.label_doc = label_doc

        return data

    def to_list_of_sent(self, sentence_data, sentence_count):
        x = []
        index = 0
        for sc in sentence_count:
            one_review = sentence_data[index:index + sc]
            x.append(one_review)
            index += sc
        return np.array(x)

    def load_all_data(self):
        data_loader = MLPVLoader(scheme="A2")
        train, vali, test = data_loader.get_mlpv()

        train = self.proc_data(train)
        vali = self.proc_data(vali)
        test = self.proc_data(test)


if __name__ == "__main__":
    a = DataBuilderMLP(embed_dim=300, target_doc_len=64, target_sent_len=1024,
                       doc_as_sent=False, doc_level=True)

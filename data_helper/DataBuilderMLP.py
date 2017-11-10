import numpy as np
import logging

from data.MLP400AV.mlpapi import MLPVLoader

from data_helper.DataHelpers import DataHelper
from data_helper.Data import DataObject


class DataBuilderMLP(DataHelper):
    problem_name = "TripAdvisor"

    sent_num_file = ["aspect_0.count", "test_aspect_0.count"]
    rating_file = ["aspect_0.rating", "test_aspect_0.rating"]
    content_file = ["aspect_0.txt", "test_aspect_0.txt"]

    def __init__(self, embed_dim, target_doc_len, target_sent_len, doc_as_sent=False, doc_level=True):
        super(DataBuilderMLP, self).__init__(embed_dim=embed_dim, target_doc_len=target_doc_len,
                                             target_sent_len=target_sent_len)

        logging.info("setting: %s is %s", "doc_as_sent", doc_as_sent)
        self.doc_as_sent = doc_as_sent
        logging.info("setting: %s is %s", "sent_list_doc", doc_level)
        self.doc_level = doc_level

        self.dataset_dir = self.data_path + 'hotel_balance_LengthFix1_3000per/'
        self.num_classes = 2  # true or false

        self.load_all_data()

    def str_2_sent(self, data):
        content = []
        paragraph = []
        for line in data:
            line = line.strip()
            if len(line) == 0 and len(paragraph) > 0:  # end of paragraph, split and push
                paragraph = " ".join(paragraph)
                content.extend(DataHelper.split_sentence(paragraph))
                paragraph = []
            # if line is too short and is not end of a para we will cut
            elif len(paragraph) == 0 and len(line) <= 3:
                pass
            else:  # keep adding to paragraph
                paragraph.append(line)

        return content

    def str_2_sent_2(self, data):
        content = "".join(data)
        content = content.replace("\n", " ")
        content = DataHelper.split_sentence(content)
        content = [s for s in content if len(s) > 10]

        return content

    def proc_data(self, data):
        for data_tuple in data:
            k, u, label = data_tuple
            k = self.str_2_sent_2(k)
            u = self.str_2_sent_2(u)
            print(k)

        if self.doc_as_sent:
            x_text = DataHelper.concat_to_doc(sent_list=x_text, sent_count=sent_count)

        x = []
        for train_line_index in range(len(x_text)):
            tokens = x_text[train_line_index].split()
            x.append(tokens)

        data = DataObject(self.problem_name, len(y))
        data.raw = x
        data.label_doc = y_onehot
        data.doc_size = sent_count

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
        data_loader = MLPVLoader()
        train, vali, test = data_loader.get_mlpv()

        train_data = self.proc_data(train)
        self.vocab, self.vocab_inv = self.build_vocab([train_data], self.vocabulary_size)
        self.embed_matrix = self.build_glove_embedding(self.vocab_inv)
        train_data = self.build_content_vector(train_data)
        train_data = self.pad_sentences(train_data)

        if self.doc_level:
            value = self.to_list_of_sent(train_data.value, train_data.doc_size)
            train_data.value = value
            DataHelper.pad_document(train_data, self.target_doc_len)

        self.train_data = train_data
        self.train_data.embed_matrix = self.embed_matrix
        self.train_data.vocab = self.vocab
        self.train_data.vocab_inv = self.vocab_inv
        self.train_data.label_instance = self.train_data.label_doc

        test_data = self.proc_data(1)
        test_data = self.build_content_vector(test_data)
        test_data = self.pad_sentences(test_data)

        if self.doc_level:
            value = self.to_list_of_sent(test_data.value, test_data.doc_size)
            test_data.value = value
            DataHelper.pad_document(test_data, self.target_doc_len)

        self.test_data = test_data
        self.test_data.embed_matrix = self.embed_matrix
        self.test_data.vocab = self.vocab
        self.test_data.vocab_inv = self.vocab_inv
        self.test_data.label_instance = self.test_data.label_doc


if __name__ == "__main__":
    a = DataBuilderMLP(embed_dim=300, target_doc_len=64, target_sent_len=1024,
                       doc_as_sent=False, doc_level=True)

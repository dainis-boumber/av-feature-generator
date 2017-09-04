from nn.input_components.OneSequence import OneSequence
from nn.input_components.OneDocSequence import OneDocSequence
from nn.middle_components.SentenceCNN import SentenceCNN
from nn.middle_components.DocumentCNN import DocumentCNN
from nn.output_components.Output import TripAdvisorOutput

from utils.Data import DataObject


class NetworkBuilder:
    def __init__(
            self, data, document_length, sequence_length, num_aspects, num_classes,
            embedding_size, filter_size_lists, num_filters,
            input_component, middle_component, output_component,
            l2_reg_lambda, dropout, batch_normalize, elu, fc):

        vocab_size = len(data.vocab)

        # input component =====
        self.input_comp = self.get_input_component()

        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y
        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle component =====
        self.middle_comp = self.get_middle_component()

        prev_layer = self.middle_comp.get_last_layer_info()
        l2_sum = self.middle_comp.l2_sum

        # output component =====


        self.scores = self.output.scores
        self.predictions = self.output.predictions
        self.loss = self.output.loss
        self.accuracy = self.output.accuracy

        try:
            self.aspect_accuracy = self.output.aspect_accuracy
        except NameError:
            self.aspect_accuracy = None

    def get_input_component(self, input_code, data):
        # input component =====
        if input_code == "Sentence":
            input_comp = OneSequence(data)
        elif input_code == "Document":
            raise NotImplementedError
            input_comp = OneDocSequence(document_length=document_length, sequence_length=sequence_length,
                                        num_classes=num_classes,
                                        vocab_size=vocab_size, embedding_size=embedding_size,
                                        init_embedding=data.embed_matrix)
        else:
            raise NotImplementedError

        return input_comp

    def get_middle_component(self, middle_code, input_comp, data,
                             filter_size_lists=None, num_filters=None, dropout=None,
                             batch_norm=None, elu=None, fc=[], l2_reg=0.0):
        if middle_code == 'Origin':
            middle_comp = SentenceCNN(previous_component=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      dropout=dropout, batch_normalize=batch_norm, elu=elu,
                                      fc=fc, l2_reg_lambda=l2_reg)
        elif middle_code == "DocumentCNN":
            middle_comp = DocumentCNN(previous_component=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      dropout=dropout, batch_normalize=batch_norm, elu=elu,
                                      fc=fc, l2_reg_lambda=l2_reg)
        else:
            raise NotImplementedError

        try:
            self.is_training = self.middle_comp.is_training
        except NameError:
            self.is_training = None

        return middle_code

    def get_output_component(self, output_code):
        if "TripAdvisor" in output_code:
            self.output = TripAdvisorOutput(self.input_comp.input_y, prev_layer, num_classes, l2_sum, l2_reg_lambda)
        elif "LSAA" in output_code:
            self.output = LSAAOutput(prev_layer=prev_layer, input_y=self.input_comp.input_y,
                                     num_aspects=num_aspects, num_classes=num_classes,
                                     document_length=document_length,
                                     l2_sum=l2_sum, l2_reg_lambda=l2_reg_lambda)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    data = DataObject("test", 100)
    data.vocab = [1, 2, 3]
    cnn = NetworkBuilder(
        data=data,
        document_length=64,
        sequence_length=1024,
        num_aspects=6,
        num_classes=5,
        embedding_size=300,
        input_component="TripAdvisorDoc",
        middle_component="DocumentCNN",
        output_component="LSAA",
        filter_size_lists=[[3, 4, 5]],
        num_filters=100,
        l2_reg_lambda=0.1,
        dropout=0.7,
        batch_normalize=False,
        elu=False,
        fc=[])

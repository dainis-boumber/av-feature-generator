from data_helper.Data import DataObject
from nn.input_components.OneDocSequence import OneDocSequence
from nn.input_components.OneSequence import OneSequence
from nn.middle_components.DocumentCNN import DocumentCNN
from nn.middle_components.SentenceCNN import SentenceCNN
from nn.output_components.Output import Output


class CNNNetworkBuilder:
    """I"m currently calling this CNN builder because i'm not sure if it can handle future
    RNN parameters, and just for flexibility and ease of management the component maker is being made into
    separate function
    """
    def __init__(self, input_comp, middle_comp, output_comp):

        # input component =====
        self.input_comp = input_comp

        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y
        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle component =====
        self.middle_comp = middle_comp

        # output component =====
        self.output_comp = output_comp

        self.scores = self.output_comp.scores
        self.predictions = self.output_comp.predictions
        self.loss = self.output_comp.loss
        self.accuracy = self.output_comp.accuracy

    @staticmethod
    def get_input_component(input_name, data):
        # input component =====
        if input_name == "Sentence":
            input_comp = OneSequence(data)
        elif input_name == "Document":
            raise NotImplementedError
            input_comp = OneDocSequence(document_length=document_length, sequence_length=sequence_length,
                                        num_classes=num_classes,
                                        vocab_size=vocab_size, embedding_size=embedding_size,
                                        init_embedding=data.embed_matrix)
        else:
            raise NotImplementedError

        return input_comp

    @staticmethod
    def get_middle_component(middle_name, input_comp, data,
                             filter_size_lists=None, num_filters=None, dropout=None,
                             batch_norm=None, elu=None, fc=[], l2_reg=0.0):
        if middle_name == 'Origin':
            middle_comp = SentenceCNN(previous_component=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      dropout=dropout, batch_normalize=batch_norm, elu=elu,
                                      fc=fc, l2_reg_lambda=l2_reg)
        elif middle_name == "DocumentCNN":
            middle_comp = DocumentCNN(previous_component=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      dropout=dropout, batch_normalize=batch_norm, elu=elu,
                                      fc=fc, l2_reg_lambda=l2_reg)
        else:
            raise NotImplementedError

        return middle_comp

    @staticmethod
    def get_output_component(output_name, middle_comp, data, l2_reg=0.0):
        if "??" in output_name:
            output_comp = Output(middle_comp, data=data, l2_reg=l2_reg)
        else:
            raise NotImplementedError

        return output_comp


if __name__ == "__main__":
    data = DataObject("test", 100)
    data.vocab = [1, 2, 3]
    cnn = CNNNetworkBuilder(
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

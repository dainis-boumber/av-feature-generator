import tensorflow as tf
from tensorflow.contrib import rnn

## implementing baseline
## see https://arxiv.org/pdf/1605.07725.pdf
def lstm(inputs, hidden=[512]):
    # defining the network
    # create 4 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in hidden]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

     # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=inputs,
                                       dtype=tf.float32)


    return outputs, state

import tensorflow as tf
import numpy as np

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, num_lstm_layers, hidden_unit_dim):
        n_input=embedding_size
        n_steps=sequence_length
        #n_hidden layer_ number of features
        n_hidden=hidden_unit_dim
        #num-layers of lstm n_layers=2 => input(t)-> lstm(1)->lstm(2)->output(t)
        n_layers=num_lstm_layers
        
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        #x = tf.split(0, n_steps, x)
        x = tf.split(x, n_steps, axis = 0)

        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            #print(tf.get_variable_scope().name)
            fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
            lstm_fw_cell_m=tf.contrib.rnn.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
        # Backward direction cell
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            #print(tf.get_variable_scope().name)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
            lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)
        
        # Get lstm cell output
        #try:
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            #         except Exception: # Old TensorFlow version only returns outputs not states
            #             outputs = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,
            #                                             dtype=tf.float32)
        return outputs[-1]
    
    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def __init__(
      self, sequence_length, input_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, num_lstm_layers, hidden_unit_dim):

      # Placeholders for input, output and dropout
      self.input_x1 = tf.placeholder(tf.float32, [None, sequence_length, input_size], name="input_x1")
      self.input_x2 = tf.placeholder(tf.float32, [None, sequence_length, input_size], name="input_x2")
      self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0, name="l2_loss")

      # Create a convolution + maxpool layer for each filter size
      with tf.name_scope("output"):
        self.out1=self.BiRNN(self.input_x1, self.dropout_keep_prob, "side1", input_size, sequence_length, num_lstm_layers=num_lstm_layers, hidden_unit_dim=hidden_unit_dim)
        self.out2=self.BiRNN(self.input_x2, self.dropout_keep_prob, "side2", input_size, sequence_length, num_lstm_layers=num_lstm_layers, hidden_unit_dim=hidden_unit_dim)
        self.distance = tf.reduce_sum(tf.abs(tf.subtract(self.out1,self.out2)),1,keep_dims=True)
        self.distance = tf.reshape(self.distance, [-1])
        self.distance = tf.exp(-self.distance, name="distance")

      with tf.name_scope("loss"):
          self.loss = tf.losses.mean_squared_error(self.input_y, self.distance)/batch_size
          #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input.y, logits=self.distance)/batch_size
      tf.summary.scalar('loss', self.loss) 
     

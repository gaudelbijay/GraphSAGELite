import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import glorot_uniform, Zeros
from tensorflow.keras.layers import Input, Dense, Dropout, Layer, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


def GraphSAGE(feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu,
              aggregator_type='pool', dropout_rate=0.0, l2_reg=0):
    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int64)
    neighbor_input = [Input(shape=(l,), dtype=tf.int64) for l in neighbor_num]
    # print('features: ', features)

    if aggregator_type == 'mean':
        aggregator = MeanAggregator
    else:
        aggregator = PoolingAggregator

    h = features
    for i in range(0, len(neighbor_num)):
        if i > 0:
            feature_dim = n_hidden
        if i == len(neighbor_num) - 1:
            activation = tf.nn.softmax
            n_hidden = n_classes
        h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                       dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type)(
            [h, node_input, neighbor_input[i]])  #

    output = h
    input_list = [features, node_input] + neighbor_input
    model = Model(input_list, outputs=output)
    return model


class MeanAggregator(Layer):
    """docstring for MeanAggregator"""

    def __init__(self, units, input_dim, neigh_max, activation=tf.nn.relu, concat=True, dropout_rate=0.0,
                 l2_reg=0, use_bias=False, seed=1024, aggregator='mean', **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.input_dim = input_dim

    def build(self, input_shapes):

        self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=glorot_uniform(
                                                 seed=self.seed),
                                             regularizer=l2(self.l2_reg),
                                             name="neigh_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=self.units, initializer=Zeros(),
                                        name='bias_weight')

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)

        concat_feat = tf.concat([neigh_feat, node_feat], axis=1)
        concat_mean = tf.reduce_mean(concat_feat, axis=1, keepdims=False)

        output = tf.matmul(concat_mean, self.neigh_weights)

        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output, dim=-1)
        output._uses_learning_phase = True

        return output


class PoolingAggregator(Layer):

    def __init__(self, units, input_dim, neigh_max, aggregator='meanpooling', concat=True,
                 dropout_rate=0.0,
                 activation=tf.nn.relu, l2_reg=0, use_bias=False,
                 seed=1024, ):
        super(PoolingAggregator, self).__init__()
        self.output_dim = units
        self.input_dim = input_dim
        self.concat = concat
        self.pooling = aggregator
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.neigh_max = neigh_max
        self.seed = seed

        # if neigh_input_dim is None:

    def build(self, input_shapes):

        self.dense_layers = [Dense(
            self.input_dim, activation='relu', use_bias=True, kernel_regularizer=l2(self.l2_reg))]

        self.neigh_weights = self.add_weight(
            shape=(self.input_dim * 2, self.output_dim),
            initializer=glorot_uniform(
                seed=self.seed),
            regularizer=l2(self.l2_reg),

            name="neigh_weights")

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=Zeros(),
                                        name='bias_weight')

        self.built = True

    def call(self, inputs, mask=None):

        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        print('node feat:   ', node_feat)
        print('neigh feat before pooling:  ', neigh_feat)

        dims = tf.shape(neigh_feat)
        batch_size = dims[0]
        num_neighbors = dims[1]

        h_reshaped = tf.reshape(
            neigh_feat, (batch_size * num_neighbors, self.input_dim))

        for l in self.dense_layers:
            h_reshaped = l(h_reshaped)
        neigh_feat = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))

        if self.pooling == "meanpooling":
            neigh_feat = tf.reduce_mean(neigh_feat, axis=1, keepdims=False)
        else:
            neigh_feat = tf.reduce_max(neigh_feat, axis=1)

        print('neigh feat after pooling', neigh_feat)

        output = tf.concat(
            [tf.squeeze(node_feat, axis=1), neigh_feat], axis=-1)

        output = tf.matmul(output, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output, dim=-1)

        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'concat': self.concat
                  }

        base_config = super(PoolingAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

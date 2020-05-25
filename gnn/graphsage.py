import tensorflow as tf 
import numpy as np 
from tensorflow.keras.initializers import glorot_uniform,zeros
from tensorflow.keras.layers import Input,Dense,Dropout,Layer,LSTM
from tensorflow.keras.regularizers import l2


def GraphSAGE(feature_dim,neighbor_num,n_hidden,n_classes,use_bias=True,
				activation=tf.nn.relu,aggregator_type='mean',dropout_rate=0.0,l2_reg=0):

	features = Input(shape=(features_dim,))
	node_input = Input(shape=(1,),dtype=tf.int32)
	neighbor_input = [Input(shape=(1,),dtype=tf.int32) for n in neighbor_num]
	if aggregator_type == 'mean':
		aggregator = MeanAggregator
	else: aggregator = PoolingAggregator

	h = features 
	for i in range(len(neighbor_num)):
		if i>0:
			features_dim = n_hidden
		if i == len(neighbor_num)-1:
			activation = tf.nn.softmax
			n_hidden = n_classes
		h = aggregator(units = n_hidden,input_dim=features_dim,activation=activation,
			l2_reg=l2_reg,use_bias=use_bias,dropout_rate=dropout_rate,neigh_max=neighbor_num[i],
			aggregator=aggregator_type)[h,node_input,neighbor_input[i]]
	output = h 
	input_list = [features,node_input]+neighbor_input
	model = Model(input_list,outputs=output)



class MeanAggregator(Layer):
 	"""docstring for MeanAggregator"""
 	def __init__(self, units,input_dim,activation=tf.nn.relu,neigh_max,concat=True,dropout_rate=0.0,
 					l2_reg=0,use_bias=False,seed=1024,aggregator=aggregator_type,**kwargs):
 		super(MeanAggregator, self).__init__(**kwargs)
 		self.units = units 
 		self.neigh_max=neigh_max
 		self.concat = concat 
 		self.dropout_rate = dropout_rate
 		self.l2_reg = l2_reg
 		self.use_bias = use_bias 
 		self.activation = activation 
 		self.seed = seed 
 		self.input_dim = input_dim 

 	def build(self,input_shape):
 		pass 
 		 

class PoolingAggregator(Layer):
	"""docstring for PoolingAggregator"""
	def __init__(self, arg):
		super(PoolingAggregator, self).__init__()
		self.arg = arg
		
		

	




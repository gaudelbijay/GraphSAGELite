import numpy as np 


def sample_neighs(G,nodes,sample_num=None,self_loop=False,shuffle=True):
	_sample = np.random.choice #speed hack (local pointer to the function)
	neighs = [list(G[int(node)]) for node in nodes]
	
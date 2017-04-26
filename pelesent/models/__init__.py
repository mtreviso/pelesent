from __future__ import absolute_import
from .nn import *
from .cnn import *
from .mlp import *
from .rcnn import *
from .crnn import *

map_m = {
	'CNN': CNN,
	'MLP': 	MLP,
	'RCNN': RCNN,
	'CRNN': CRNN
}

def get_model(name, **kwargs):
	return map_m[name](**kwargs)
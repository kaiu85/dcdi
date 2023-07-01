"""
GraN-DAG

Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import sys
import math
sys.path.insert(0, '../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class ForwardModel(BaseModel):
    def __init__(self, num_vars, num_layers, hid_dim, num_params,
                 nonlin="leaky-relu", intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1):

        super(ForwardModel, self).__init__(num_vars, num_layers, hid_dim, num_params,
                                             nonlin=nonlin,
                                             intervention=intervention,
                                             intervention_type=intervention_type,
                                             intervention_knowledge=intervention_knowledge,
                                             num_regimes=num_regimes)
        self.reset_params()
        self.adjacency = torch.ones((self.num_vars, self.num_vars))
        
    def forward(self, x, mask=None, regime=None):
        
        weights, biases = self.get_parameters('wb')        
        x = self.forward_given_params(x, weights, biases)      
        x = torch.stack(x, 1)
        
        return x
    
    
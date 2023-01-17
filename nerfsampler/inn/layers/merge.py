from __future__ import annotations
import torch
from typing import Callable
from nerfsampler.inn.fields import DiscretizedField

from nerfsampler.inn.point_set import Discretization, PointValues
nn=torch.nn
F=nn.functional

class MergeLayer(nn.Module):
    def __init__(self, merge_function: Callable, interpolator=None):
        super().__init__()
        self.merge_function = merge_function
        self.interpolator = interpolator

    def forward(self, inr1: DiscretizedField, inr2: DiscretizedField) -> DiscretizedField:
        x = inr1.coords
        y = inr2.coords
        if len(x) == len(y):
            if torch.all(x == y):
                inr1.values = self.merge_function(inr1.values, inr2.values)
            else:
                x_indices = torch.sort((x[:,0]+2)*x.size(0)/2 + x[:,1]).indices
                y_indices = torch.sort((y[:,0]+2)*y.size(0)/2 + y[:,1]).indices
                inr1.coords = x[x_indices]
                if torch.allclose(self.coords, y[y_indices]):
                    inr1.values = self.merge_function(inr1.values[:,x_indices], inr2.values[:,y_indices])
                else:
                    raise ValueError('coord_conflict')
        else:
            raise NotImplementedError('coord_conflict')
        return inr1

        # pdb.set_trace()
        # coord_diffs = x.unsqueeze(0) - y.unsqueeze(1)
        # matches = (coord_diffs.abs().sum(-1) == 0)
        # y_indices, x_indices = torch.where(matches)
        # X = values1[x_indices]
        # Y = values2[y_indices]
        # merged_outs = self.merge_function(X,Y)
        # extra_x = set_difference(torch.arange(len(x), device=x.device), x_indices)
        # extra_x_vals = self.merge_function(values1[extra_x],
        #     self.interpolator(query_coords=x[extra_x], observed_coords=y, values=values2))
        # extra_y = set_difference(torch.arange(len(y), device=x.device), y_indices)
        # extra_y_vals = self.merge_function(values2[extra_y],
        #     self.interpolator(query_coords=y[extra_y], observed_coords=x, values=values1))
        # self.sampled_coords = torch.cat((x[x_indices], x[extra_x], y[extra_y]), dim=0)
        
        # return torch.cat((merged_outs, values1[extra_x], values2[extra_y]), dim=0)

    def change_discretization(self, mode: str='grid'):
        self.inr1.change_discretization(mode=mode)
        self.inr2.change_discretization(mode=mode)

def merge_domains(d1, d2):
    return (max(d1[0], d2[0]), min(d1[1], d2[1]))
    
def set_difference(x, y):
    combined = torch.cat((x, y))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts == 1]

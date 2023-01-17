from torch import nn
from nerfsampler import inn
from nerfsampler.inn.fields import DiscretizedField
from nerfsampler.inn.layers.other import FlowLayer
from nerfsampler.inn.nets.nerfsampler import nerfsampler

class LearnedSampler(nerfsampler):
    def __init__(self, dense_sampler, query_layers: nn.Module,
        flow_layers: nn.Module, ratio=16, **kwargs):
        """_summary_

        Args:
            dense_sampler (_type_): full QMC sampler
            query_layers (nn.Module): layers whose output should be preserved under
                            dense v. sparse sampling
            flow_layers (nn.Module): layers to estimate sampling pattern
            ratio (int, optional): ratio of points in dense to sparse sampler. Defaults to 16.
        """        
        super().__init__(sampler=None, layers=None, **kwargs)
        self.flow_layers = flow_layers
        self.query_layers = query_layers
        self.dense_sampler = dense_sampler.copy()
        self.sparse_sampler = dense_sampler
        self.sparse_sampler['sample points'] = dense_sampler['sample points']//ratio
        
    def forward(self, inr: DiscretizedField):
        dense_inr = self.discretize_inr(inr, self.dense_sampler)
        sparse_inr = self.discretize_inr(inr, self.sparse_sampler)
        flow = self.flow_layers(sparse_inr)
        sparse_inr.coords = sparse_inr.coords + flow.values
        return self.query_layers(sparse_inr), self.query_layers(dense_inr)

    def loss(self, inr: DiscretizedField):
        sparse, dense = self.forward(inr)
        return (sparse - dense).abs().sum()

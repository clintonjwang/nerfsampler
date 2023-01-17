"""Attention Layer"""
import torch
nn = torch.nn
F = nn.functional

from nerfsampler.inn import functional as inrF

class TokenAttn(nn.Module):
    def __init__(self, in_channels, out_channels, d_k=64, num_heads=1, spacing=.2):
        # in_ch is the output dimensions of the input INR (3 for RGB, 1 for occupancy net)
        # out_ch is the output dimensions of the output INR
        # spatial_dim is the coordinate dimensions (2 for SIREN, 6 for NeRF)
        super().__init__()
        #self.Q = polynomials.LegendreFilter(in_channels, d_k, radius=spacing, input_dims=spatial_dim)
        #self.K = polynomials.LegendreFilter(in_channels, d_k, radius=spacing, input_dims=spatial_dim)
        self.d_k = d_k
        # self.V = DerivedINR(n_outputs=d_k)
        # self.V_weights = nn.Parameter(torch.randn(V_dim, d_k))
        self.num_heads = num_heads
        
    def forward(self, inr):
        pass
        # new_inr = inr.create_derived_inr()
        # new_inr.set_integrator(inrF.tokenization, 'Tokenization', layer=self)
#         new_inr.channels = self.out_channels
        # return new_inr

#         Q_values = inr(self.Q_coords_to_sample) * self.Q_weights
#         K_values = inr(self.K_coords_to_sample) * self.K_weights
#         QK = torch.linalg.matmul(Q_values, K_values.transpose())
#         # output = AttnINR()V * torch.softmax(QK / self.d_k.sqrt(), dim=1)
#         pass

# class SelfAttn(nn.Module):
#     # Given an input INR, produce an output INR that is a weighted combination of "value" INRs
#     # Weights are normalized dot products of key and query values
#     def __init__(self, in_channels, out_channels, num_samples, d_k, V_dim, spatial_dim=2):
#         # in_ch is the output dimensions of the input INR (3 for RGB, 1 for occupancy net)
#         # out_ch is the output dimensions of the output INR
#         # spatial_dim is the coordinate dimensions (2 for SIREN, 6 for NeRF)
#         super().__init__()
#         self.Q = polynomials.LegendreFilter(in_channels, d_k, input_dims=spatial_dim)
#         self.K = polynomials.LegendreFilter(in_channels, d_k, input_dims=spatial_dim)
#         self.d_k = d_k
#         self.V = DerivedINR(n_outputs=d_k)
#         self.V_weights = nn.Parameter(torch.randn(V_dim, d_k))

#     def forward(self, inr):
#         Q_values = inr(self.Q_coords_to_sample) * self.Q_weights
#         K_values = inr(self.K_coords_to_sample) * self.K_weights
#         QK = torch.linalg.matmul(Q_values, K_values.transpose())
#         # output = AttnINR()V * torch.softmax(QK / self.d_k.sqrt(), dim=1)
#         pass

# class MultiHeadSelfAttn(nn.Module):
#     def __init__(self, in_dim, in_ch, out_ch):
#         # in_ch is the output dimensions of the input INR (3 for RGB, 1 for occupancy net)
#         # out_ch is the output dimensions of the output INR
#         # in_dim is the coordinate dimensions (2 for SIREN, 6 for NeRF)
#         super().__init__()
#         self.K_coords_to_sample = nn.Parameter()
#         self.Q_coords_to_sample = nn.Parameter()
#         self.V = INR()

#     def forward(self, inr):
#         pass

# class CrossAttnINR(nn.Module):
#     def __init__(self, in_dim, out_dim, t_dim=0, C=512):
#         super().__init__()
#         self.K_coords_to_sample = nn.Parameter()
#         self.Q_coords_to_sample = nn.Parameter()
#         self.V = INR()

#     def forward(self, inr):
#         pass




# class DoubleINR2INR(nn.Module):
#     def __init__(self, out_ch, in_ch1=3, in_ch2=3, in_dim=2, t_dim=0, C=512):
#         # in_ch1 is the output dimensions of the first input INR (3 for RGB, 1 for occupancy net)
#         # in_ch2 is the output dimensions of the second input INR (3 for RGB, 1 for occupancy net)
#         # out_ch is the output dimensions of the output INR
#         # in_dim is the coordinate dimensions (2 for SIREN, 6 for NeRF)
#         super().__init__()
#         if t_dim > 0:
#             self.time_mlp = nn.Sequential(
#                 SinusoidalPosEmb(t_dim),
#                 nn.Linear(t_dim, t_dim * 4),
#                 nn.GELU(),
#                 nn.Linear(t_dim * 4, t_dim)
#             )
#         else:
#             self.time_mlp = None
#             self.layer1 = CrossAttnINR(in_ch1, in_ch2, out_ch=C, in_dim=in_dim)
#             self.layer2 = SelfAttnINR(C, C)
#             # self.bn = nn.BatchNorm1d(C)
#             self.layer3 = nn.Linear(C, C)
#             self.layer4 = nn.Linear(C, out_dim)

#     def forward(self, x, time):
#         # t = self.time_mlp(time) if exists(self.time_mlp) else None
#         x = F.silu(self.layer1(x), inplace=True)
#         x = F.silu(self.layer2(x), inplace=True) #torch.cat([x,t], dim=1)
#         x = F.silu(self.layer3(x), inplace=True)
#         return self.layer4(x)


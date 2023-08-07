import torch
from torch import nn
from torch.nn import functional as F

from lib.network.AutoEncoder import Encoder, NeckDecoder, Decoder
from lib.network.MLP import MLP


class YVAEModule(nn.Module):
    def __init__(self, opt):
        super(YVAEModule, self).__init__()
        
        self.exp_dim_2d = opt.exp_dim_2d
        self.domains = opt.domains
        self.shared_encoder = Encoder(opt.exp_dim_2d, opt.encoder_dims)
        self.shared_neckdecoder = NeckDecoder(opt.exp_dim_2d, opt.neck_dims)
        self.mapping_mlp = MLP(opt.mapping_dims)

        self.decoder_dict = {}
        for domain in opt.domains:
            self.decoder_dict[domain] = Decoder(opt.decoder_dims)
        self.decoder_dict = nn.ModuleDict(self.decoder_dict)
    
    def encode(self, input):
        exp_code_2d = self.shared_encoder(input - 0.5)
        return exp_code_2d

    def decode(self, exp_code_2d, domain):
        result = self.shared_neckdecoder(exp_code_2d)
        result = self.decoder_dict[domain](result)
        return result

    def mapping(self, exp_code_2d):
        return self.mapping_mlp(exp_code_2d)

    def forward(self, func, data):
        if func == 'encode':
            return self.encode(data)
        elif func == 'decode':
            return self.decode(data[0], data[1])
        elif func == 'mapping':
            return self.mapping(data)
from turtle import forward
import numpy as np
import math
import torch.nn.functional as F
from unicodedata import bidirectional
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class Config:
    def __init__(self):
        self.model_name = 'composite_SA_BLSTM'
        self.save_path = './save_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.dropout = 0.5
        self.num_classes = 12
        self.num_heads = 8
        self.n_vocab = 0
        self.num_epochs = 30 
        self.batch_size = 16 
        self.pad_size = 512
        self.learning_rate = 2e-5
        self.cb_embed = 768
        self.embed = 300
        self.hidden_size = 256 
        self.num_layers = 2 
    

class CodeBert_Blstm(nn.Module):
    def __init__(self, config):
        super(CodeBert_Blstm, self).__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.multi_atten = nn.MultiheadAttention(config.hidden_size*2, config.num_heads)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, 
                            bidirectional=True, batch_first=True, dropout=config.dropout)

        self.W_Q = nn.Linear(config.hidden_size*2, config.hidden_size*2, bias=False)
        self.W_K = nn.Linear(config.hidden_size*2, config.hidden_size*2, bias=False)
        self.W_V = nn.Linear(config.hidden_size*2, config.hidden_size*2, bias=False)

        self.linear0 = nn.Sequential(
                                    nn.Linear(config.cb_embed, config.hidden_size*2),
                                    nn.ReLU(),
                                    nn.Linear(config.hidden_size*2, config.hidden_size),
                                    nn.ReLU()
                                    )
        self.linear1 = nn.Sequential(
                                    nn.MaxPool1d(2, stride=2),
                                    nn.Linear(config.hidden_size, config.hidden_size),
                                    nn.ReLU()
                                    )
        self.linear2 = nn.Sequential(
                                    nn.Linear(config.hidden_size*3, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, config.num_classes))
    
    def sf_attention(self, input):
        Q = self.W_Q(input)
        K = self.W_K(input)
        V = self.W_V(input)
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1,2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        code_context = torch.matmul(alpha_n, V)

        output = code_context.sum(1)
        return output, alpha_n

    def forward(self, s, x1, x2):
        s_code = s
        codebert_out = self.codebert(s_code)[0]
        codebert_out = codebert_out[:, 0, :]
        codebert_out = self.linear0(codebert_out)

        code1 = x1
        out1 = self.embedding(code1)
        out1, _ = self.lstm(out1)
        out1 = torch.cat([out1[:,0,-256:], out1[:,-1,:256]], 1)
        out1 = out1.view(out1.shape[0], -1, out1.shape[1])
        sf_atten_out1 = self.linear1(sf_atten_out1)

        code2 = x2
        out2, _ =self.lstm(code2)
        out2 = torch.cat([out2[:,0,-256:], out2[:,-1,:256]], 1)
        out2 = out2.view(out2.shape[0], -1, out2.shape[1])
        sf_atten_out2, _ = self.sf_attention(out2)
        sf_atten_out2 = self.linear1(sf_atten_out2)

        out = torch.cat([codebert_out, sf_atten_out1, sf_atten_out2], 1)
        return out

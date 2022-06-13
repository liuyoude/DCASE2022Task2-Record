"""
author:liuyoude
date:2021-06-29
"""

import torch
import torch.nn as nn

from .attention import *
from .utils import *

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0., norm=True):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.norm = norm

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(nn.ReLU(inplace=True)((self.w1(output))))
        output = self.dropout(output.transpose(1, 2))
        output += x
        if self.norm:
            output = self.layer_norm(output)
        return output

# class PositionalWiseFeedForward(nn.Module):
#     def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.):
#         super(PositionalWiseFeedForward, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(model_dim, ffn_dim, bias=False),
#             nn.ReLU(),
#             nn.Linear(model_dim, ffn_dim, bias=False),
#             nn.Dropout(dropout),
#         )
#         self.layer_norm = nn.LayerNorm(model_dim)
#     def forward(self, inputs):
#         '''
#         inputs: [batch_size, seq_len, d_model]
#         '''
#         residual = inputs
#         output = self.fc(inputs)
#         return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, out_dim=512, num_heads=8, ffn_dim=2048, dropout=0., norm=True):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, out_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(out_dim, ffn_dim, dropout, norm=norm)
    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention

class Encoder(nn.Module):
    def __init__(self,
                 max_seq_len,
                 vocab_size=None,
                 num_layers=6,
                 model_dim=512,
                 out_dim=512,
                 num_heads=8,
                 dropout=0.,
                 use_single=False):
        super(Encoder, self).__init__()
        self.model_dim = model_dim
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, model_dim, num_heads, 2 * model_dim, dropout, norm=True) for _ in range(num_layers-1)]
        )
        if use_single:
            self.encoder_layers.append(EncoderLayer(model_dim, out_dim, num_heads, 2 * out_dim, dropout, norm=False))
        else:
            self.encoder_layers.append(EncoderLayer(model_dim, out_dim, num_heads, 2 * out_dim, dropout, norm=True))
        self.vocab_size = vocab_size
        if vocab_size:
            self.seq_embedding = nn.Embedding(vocab_size+1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

        self.fc_final = nn.Linear(model_dim, model_dim)

    def forward(self, inputs, inputs_embedding, inputs_len):
        output = inputs_embedding
        if self.vocab_size:
            output = self.seq_embedding(inputs)
        self_attention_mask = padding_mask(inputs, inputs)
        output += self.pos_embedding(inputs_len)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
        output = output.view(output.size(0), 1, self.model_dim)
        output = self.fc_final(output)
        return output, attentions

    # def forward(self, inputs_embedding):
    #     inputs = torch.ones((1, 5), dtype=torch.int).cuda()
    #     inputs_len = torch.ones((1, 1), dtype=torch.int) * 5
    #     inputs_len = inputs_len.cuda()
    #     output = inputs_embedding
    #     if self.vocab_size:
    #         output = self.seq_embedding(inputs)
    #     self_attention_mask = padding_mask(inputs, inputs)
    #     output += self.pos_embedding(inputs_len)
    #     attentions = []
    #     for encoder in self.encoder_layers:
    #         output, attention = encoder(output, self_attention_mask)
    #         attentions.append(attention)
    #     return output, attentions


class TransAE(nn.Module):
    def __init__(self,
                 max_seq_len,
                 frames=5,
                 n_mels=128,
                 n_fft=513,
                 n_head=4,
                 num_classes=4,
                 dropout=0.,
                 lpe=True,
                 cfp=True):
        super(TransAE, self).__init__()
        self.lpe = lpe
        self.cfp = cfp
        frames = frames - 1 if cfp else frames

        self.phase_encode = nn.Sequential(
            nn.Linear(n_fft, n_mels),
            nn.BatchNorm1d(frames),
            nn.Linear(n_mels, n_mels),
            nn.BatchNorm1d(frames),
        )

        self.encoder1 = EncoderLayer(n_mels, n_mels, n_head, frames*n_mels, dropout, norm=True)
        self.encoder2 = EncoderLayer(n_mels, n_mels, n_head, frames*n_mels, dropout, norm=True)
        self.encoder3 = EncoderLayer(n_mels, n_mels, n_head, frames * n_mels, dropout, norm=True)
        self.encoder4 = EncoderLayer(n_mels, n_mels, n_head, frames * n_mels, dropout, norm=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(n_mels, 32),
            # nn.LayerNorm([frames-1, 32]),
        )
        self.expand = nn.Linear(32, n_mels)
        self.decoder1 = EncoderLayer(n_mels, n_mels, n_head, frames*n_mels, dropout, norm=True)
        self.decoder2 = EncoderLayer(n_mels, n_mels, n_head, frames*n_mels, dropout, norm=True)
        self.decoder3 = EncoderLayer(n_mels, n_mels, n_head, frames * n_mels, dropout, norm=True)
        self.decoder4 = EncoderLayer(n_mels, n_mels, n_head, frames * n_mels, dropout, norm=True)
        self.fc_out = nn.Linear(n_mels, n_mels)

        # if cfp:
        #     self.fc = nn.Sequential(
        #         nn.Linear(32*(frames-1), 32*(frames-1)),
        #         nn.LayerNorm(32*(frames-1)),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.2),
        #         nn.Linear(32*(frames-1), num_classes)
        #     )
        # else:
        #     self.fc = nn.Sequential(
        #         nn.Linear(32 * (frames), 32 * (frames)),
        #         nn.LayerNorm(32 * (frames)),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.2),
        #         nn.Linear(32 * (frames), num_classes)
        #     )
        self.fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        self.pos_embedding = PositionalEncoding(128, max_seq_len)
    def forward(self, inputs, inputs_embedding, inputs_len, phase, visual=False, lpe_visual=False):
        output = inputs_embedding
        self_attention_mask = padding_mask(inputs, inputs)
        if self.lpe:
            output = output + self.phase_encode(phase)
        else:
            output = output + self.pos_embedding(inputs_len)
        output = output.float()

        output, attention = self.encoder1(output, self_attention_mask)
        output, attention = self.encoder2(output, self_attention_mask)
        # output, attention = self.encoder3(output, self_attention_mask)
        # output, attention = self.encoder4(output, self_attention_mask)
        features = self.bottleneck(output)

        output = self.expand(features)
        output, attention = self.decoder1(output, self_attention_mask)
        output, attention = self.decoder2(output, self_attention_mask)
        # output, attention = self.decoder3(output, self_attention_mask)
        # output, attention = self.decoder4(output, self_attention_mask)
        # output = torch.mean(output, dim=1, keepdim=True)
        if self.cfp:
            output = torch.mean(output, dim=1, keepdim=True)
        output = self.fc_out(output)

        # features = self.fc_ffn(features)
        # features = features.view(features.size(0), -1)
        (features, _) = torch.max(features, dim=1)
        pre_ids = self.fc(features)

        if visual:
            return output, features
        if lpe_visual:
            lpe_result = self.phase_encode(phase)
            pe_result = self.pos_embedding(inputs_len)
            return lpe_result, pe_result

        return output, pre_ids


if __name__ == '__main__':

    a = torch.randn((64, 4, 128))
    b = torch.randn((64, 4, 513))
    # src_seq = torch.ones((a.size(0), 4), dtype=torch.int)
    # src_len = torch.ones((a.size(0), 1), dtype=torch.int) * (4)
    #
    # net = IDC_Transformer(4)
    # c = net(src_seq, a, src_len, b, type=None)
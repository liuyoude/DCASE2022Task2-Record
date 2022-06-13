"""
author:liuyoude
date:2021-06-29
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
    positional encoding
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        # PE matrix
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)
        ])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding)

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        self.position_encoding = nn.Embedding(max_seq_len+1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        max_len = torch.max(input_len)
        input_pos = torch.tensor(
            [list(range(1, len_idx+1)) + [0] * (max_len - len_idx) for len_idx in input_len]
        )
        input_pos = input_pos.to(input_len.device)
        # if input_len.is_cuda:
        #     input_pos = input_pos.cuda()
        return self.position_encoding(input_pos)

"""
mask
"""

# padding mask
def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask

# sequence_mask
def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask

if __name__ == '__main__':
   input_len = torch.Tensor([[5], [5], [4], [2], [3]]).int()
   pe = PositionalEncoding(max_seq_len=10, d_model=4)
   e = pe(input_len)
   print(e.size())
   #print(np.mean(e.numpy()[0], axis=1))

   # context_attn_mask = torch.ones((10, 5))
   # attn = torch.rand((5, 5))
   # x = attn.masked_fill_(context_attn_mask, float('-inf'))
   # print(x.size(), x)
   # a1 = e[0][0].numpy()
   # a2 = e[0][1].numpy()
   # a3 = e[0][2].numpy()
   # a4 = e[0][3].numpy()
   # a5 = e[0][4].numpy()
   # plt.plot(a1, label='p1')
   # plt.plot(a2, label='p2')
   # plt.plot(a3, label='p3')
   # plt.plot(a4, label='p4')
   # plt.plot(a5, label='p5')
   # plt.legend(['p1', 'p2', 'p3', 'p4', 'p5'])
   # plt.show()
   # for a in input_len:
   #     print(a)


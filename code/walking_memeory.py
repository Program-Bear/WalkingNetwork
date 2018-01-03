from __future__ import division
import abc
import util
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from utils.functions import gumbel_softmax
from KBQA_PyTorch import KBQA
from entity2facts import get_memory

class walking_memory(nn.Module):
    def __init__(self, **kwargs):
        super(walking_memory, self).__init__()
        self.kbqa = KBQA(**kwargs)
        self.project = nn.Linear(self.kbqa.lstm_hidden_size * 2 + self.kbqa.embedding_size, self.kbqa.lstm_hidden_size * 2)
    def forward(self, memory, question, question_lengths):
        logits, attn_ques = self.kbqa(memory, question, question_lengths)
        entity_distribution = gumbel_softmax(logits, hard=True)
        entities = torch.max(entity_distribution, dim = 1)[1]
        mem = get_memory(entities)
        entity_emb = self.kbqa.get_entity_embedding(entity_distribution)
        attn_ques = F.tanh(self.project(torch.cat([attn_ques, entity_emb], 1)))
        attn_ques = F.dropout(attn_ques, p = 0.5)
        logits, attn_ques = self.kbqa.forward2(Variable(torch.LongTensor(mem.astype(int))).cuda(),  attn_ques, question_lengths)
        entity_distribution = gumbel_softmax(logits, hard=True)
        entities = torch.max(entity_distribution, dim = 1)[1]
        mem = get_memory(entities)
        entity_emb = self.kbqa.get_entity_embedding(entity_distribution)
        attn_ques = F.tanh(self.project(torch.cat([attn_ques, entity_emb], 1)))
        attn_ques = F.dropout(attn_ques, p = 0.5)
        logits, attn_ques = self.kbqa.forward2(Variable(torch.LongTensor(mem.astype(int))).cuda(), attn_ques, question_lengths)
        return logits


'''
This is an implementation of Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks with PyTorch. 
'''

from __future__ import division
import abc
import util
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from utils.functions import gumbel_softmax
from torch.nn.functional import softmax

# Truncated Backpropagation 
def detach(states):
    return [state.detach() for state in states] 

class QAbase(nn.Module):
    """
    Base class for Question Ansering
    """

    def __init__(self, entity_vocab_size, embedding_size, hops=3,
                 question_encoder='lstm', use_peepholes=True, load_pretrained_model=False,
                 load_pretrained_vectors=False, pretrained_entity_vectors=None, verbose=False):

        super(QAbase, self).__init__()

        self.entity_vocab_size = entity_vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_size = embedding_size
        self.question_encoder = question_encoder
        self.use_peepholes = use_peepholes
        self.hops = hops

        """Common Network parameters"""
        # projection
        self.Linear1 = nn.Linear(self.embedding_size, 2 * self.embedding_size)
        nn.init.xavier_uniform(self.Linear1.weight)
        nn.init.constant(self.Linear1.bias, 0)
        self.Linear2 = nn.Linear(2 * self.embedding_size, self.embedding_size)
        nn.init.xavier_uniform(self.Linear2.weight)
        nn.init.constant(self.Linear2.bias, 0)
        # weights for each hop of the memory network
        self.R = nn.ModuleList([nn.Linear(2 * self.embedding_size, 2 * self.embedding_size) for h in range(self.hops)])
        for r in self.R:
            nn.init.xavier_uniform(r.weight)
            nn.init.constant(r.bias, 0)
        self.attn_weights_all_hops = []
        # embedding layer
        initializer_op = None
        trainable = False
        if load_pretrained_model:
            if verbose:
                print(
                    'Load pretrained model is set to {0} and hence entity_encoder trainable is set to {0}'.format(
                        load_pretrained_model))
                trainable = True

        self.entity_encoder = nn.Embedding(self.entity_vocab_size, self.embedding_size)
        if load_pretrained_vectors:
            if verbose:
                print('pretrained entity & word embeddings available. Initializing with them.')
            assert (pretrained_entity_vectors is not None)
            self.entity_encoder.weight.data[:-1].copy_(torch.from_numpy(pretrained_entity_vectors))
        else:
            if verbose:
                print('No pretrained entity & word embeddings available. Learning entity embeddings from scratch')
                trainable = True
            nn.init.xavier_uniform(self.entity_encoder.weight)
        self.entity_decoder = nn.Linear(self.embedding_size, self.entity_vocab_size, bias = False)
        self.entity_decoder.weight = self.entity_encoder.weight    

        # dummy memory is set to -inf, so that during softmax for attention weight, we correctly
        # assign these slots 0 weight.
        nn.init.constant(self.entity_encoder.weight.data[-1], 0)
        self.attn_weights_all_hops = []

        # for encoding question
        self.question_lstm = nn.LSTM(input_size = self.embedding_size, 
                                     hidden_size = self.lstm_hidden_size, 
                                     batch_first = True, 
                                     bidirectional = True)
        
    def get_entity_embedding(self, entity_distribution):
        return torch.mm(entity_distribution, self.entity_decoder.weight)
    def get_question_embedding(self, question, question_lengths):
        """ 
        Encodes the question. Current implementation is encoding with biLSTM.
        Input: 
            question: (batch_size, max_question_length)
            question_lengths: (batch_size,)
        Output:
            question_embedding: (batch_size, embedding_dim * 2)
        """
        (batch_size, max_question_length) = question.size()
        # question_word_embedding: [B, max_question_length, embedding_dim]
        question_word_embedding = self.entity_encoder(question)
        question_word_embedding_shape = question_word_embedding.size()
        if self.question_encoder == 'lstm':
            zero_states = (Variable(torch.zeros(2, batch_size, self.embedding_size)).cuda(),
                            Variable(torch.zeros(2, batch_size, self.embedding_size)).cuda())                             
            output, (h_n, c_n) = self.question_lstm(question_word_embedding, zero_states)
            
            question_embedding = output[:, -1, :]
            # # last_fwd = output[range(batch_size), (question_lengths - 1).data, torch.LongTensor(range(self.lstm_hidden_size)).cuda()].view(batch_size, self.embedding_size)
            # last_fwd = output[range(batch_size), (question_lengths - 1).data, torch.LongTensor(range(self.lstm_hidden_size)).cuda()].view(batch_size, self.embedding_size)
            # last_bwd = output[:, 0, self.lstm_hidden_size:]
            # # question_embedding: [B,2D]
            # question_embedding = torch.cat([last_fwd, last_bwd], 1)
        else:
            raise NotImplementedError
        return question_embedding

    def get_key_embedding(self, *args, **kwargs):
        raise NotImplementedError

    def get_value_embedding(self, val_mem):
        # each is [B, max_num_slots, D]
        val_embedding = self.entity_encoder(val_mem)
        return val_embedding

    def seek_attention(self, question_embedding, key, value, C, mask):
        """ 
        Iterative attention. 

        Parameters
        ----------------
        question_embedding: FloatTensor (batch_size, embedding_dim * 2)
        key: FloatTensor (batch_size, memory_size, embedding_dim * 2)
        value: FloatTensor (batch_size, memory_size, embedding_dim)
        C: float
        mask: FloatTensor (batch_size, embedding_dim * 2)

        Returns
        ---------------
        attended_question_embedding: FloatTensor (batch_size, embedding_dim * 2)
        """
        batch_size, _ = question_embedding.size()
        for h in range(self.hops):
            expanded_question_embedding = question_embedding.view(batch_size, 2*self.embedding_size, -1)
            # self.key*expanded_question_embedding [B, M, 2D]; self.attn_weights: [B,M]
            attn_logits = torch.matmul(key, expanded_question_embedding).view(batch_size, -1)
            attn_logits = attn_logits * mask + C * (1 - mask)
            self.attn_weights = softmax(attn_logits)
            # self.attn_weights_all_hops.append(self.attn_weights)

            # attn_weights_reshape: [B, M, 1]
            attn_weights_reshape = self.attn_weights.view(batch_size, 1, -1)
            # self.value * attn_weights_reshape:[B, M, D]; self.attn_value:[B, D]
            attn_value = torch.matmul(attn_weights_reshape, value).view(batch_size, self.embedding_size)
            # attn_value_proj : [B, 2D]
            attn_value_proj = self.Linear1(attn_value)
            sum = question_embedding + attn_value_proj
            # question_embedding: [B, 2D]
            question_embedding = self.R[h](sum)
        return question_embedding

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class KBQA(QAbase):
    """
    Class for KB Question Answering
    TODO(rajarshd): describe input/output behaviour
    """

    def __init__(self, relation_vocab_size,
                 key_encoder='concat', **kwargs):
        super(KBQA, self).__init__(**kwargs)
        self.key_encoder = key_encoder
        self.relation_vocab_size = relation_vocab_size

        """Specialized Network parameters"""
        self.relation_encoder = nn.Embedding(self.relation_vocab_size, self.embedding_size)
        nn.init.xavier_uniform(self.relation_encoder.weight)
        nn.init.constant(self.relation_encoder.weight.data[-1], 0)

    def get_key_embedding(self, entity, relation):
        """TODO(rajarshd): describe various options"""
        # each is [B, max_num_slots, D]
        e1_embedding = self.entity_encoder(entity)
        r_embedding = self.relation_encoder(relation)

        # key shape is [B, max_num_slots, 2D]
        if self.key_encoder == 'concat':
            key = torch.cat([e1_embedding, r_embedding], dim=2)
        else:
            raise NotImplementedError
        return key

    def __call__(self, memory, question, question_lengths):
        # split memory and get corresponding embeddings
        e1 = memory[:, :, 0]
        r = memory[:, :, 1]
        e2 = memory[:, :, 2]
        C = -1000
        mask = (e1 != (self.entity_vocab_size - 1)).float()
        key = self.get_key_embedding(e1, r)
        value = self.get_value_embedding(e2)
        ques = self.get_question_embedding(question, question_lengths)

        # get attention on retrived informations based on the question
        attn_ques = self.seek_attention(ques, key, value, C, mask)

        # output embeddings - share with entity lookup table
        # project down
        model_answer = self.Linear2(attn_ques) # model_answer: [B, D]
        logits = self.entity_decoder(model_answer)  # scores: [B, num_entities]
        return logits, attn_ques

    def forward2(self, memory, ques, question_lengths):
        # split memory and get corresponding embeddings
        e1 = memory[:, :, 0]
        r = memory[:, :, 1]
        e2 = memory[:, :, 2]
        C = -1000
        mask = (e1 != (self.entity_vocab_size - 1)).float()
        key = self.get_key_embedding(e1, r)
        value = self.get_value_embedding(e2)

        # get attention on retrived informations based on the question
        attn_ques = self.seek_attention(ques, key, value, C, mask)

        # output embeddings - share with entity lookup table
        # project down
        model_answer = self.Linear2(attn_ques) # model_answer: [B, D]
        logits = self.entity_decoder(model_answer)  # scores: [B, num_entities]
        return logits, attn_ques

class TextQA(QAbase):
    """
    Class for QA with Text only
    TODO(rajarshd): describe input/output behaviour
    """

    def __init__(self, key_encoder='lstm',
                 separate_key_lstm=False, **kwargs):
        super(TextQA, self).__init__(**kwargs)
        self.key_encoder = key_encoder
        self.separate_key_lstm = separate_key_lstm

        """Specialized Network parameters"""
        # for encoding key
        if self.separate_key_lstm:
            self.key_lstm = nn.LSTM(input_size = self.embedding_size, 
                                    hidden_size = self.lstm_hidden_size, 
                                    batch_first = True, 
                                    bidirectional = True)

    def get_key_embedding(self, key_mem, key_lens):
        """TODO(rajarshd): describe various options"""
        # each is [B, max_num_slots, max_key_len, D]
        batch_size, max_num_slots, max_key_len = key_mem.size()
        key_embedding = self.entity_encoder(key_mem.view(batch_size * max_num_slots, max_key_len))
        # reshape the data to [(B, max_num_slots), max_key_len, D]
        key_len_reshaped = key_lens.view(batch_size)
        if self.key_encoder == 'lstm':
            if self.separate_key_lstm:
                outputs, (h_n, c_n) = self.key_lstm(key_embedding)
            else:
                outputs, (h_n, c_n) = self.question_lstm(key_embedding)
            # [(B, max_num_slots), 2D]
            last_fwd = output[range(batch_size), question_lengths, 0]
            last_bwd = output[:, 0, 1]
            key = torch.concat([last_fwd, last_bwd]).view(batch_size, max_num_slots, self.embedding_size * 2)
        else:
            raise NotImplementedError
        return key

    def __call__(self, key_mem, key_len, val_mem, question, question_lengths):
        # key_mem is [B, max_num_mem, max_key_len]
        # key_len is [B, max_num_mem]
        # val_mem is [B, max_num_mem]

        C = torch.ones(key_len.size()) * -1000
        mask = (key_len != 0)
        key = self.get_key_embedding(key_mem, key_len)
        value = self.get_value_embedding(val_mem)

        ques = self.get_question_embedding(question, question_lengths)

        # get attention on retrived informations based on the question
        attn_ques = self.seek_attention(ques, key, value, C, mask)

        # project down
        model_answer = self.Linear2(attn_ques)  # model_answer: [B, D]
        logits = self.entity_decoder(model_answer)
        return logits    

class TextKBQA(QAbase):
    """
    Class for QA with Text+KB
    """

    def __init__(self, relation_vocab_size,
                 kb_key_encoder='concat',
                 text_key_encoder='lstm',
                 join='concat2',
                 separate_key_lstm=False, **kwargs):
        super(TextKBQA, self).__init__(**kwargs)
        self.join = join
        self.kb_key_encoder = kb_key_encoder
        self.text_key_encoder = text_key_encoder
        self.separate_key_lstm = separate_key_lstm
        self.relation_vocab_size = relation_vocab_size

        """Specialized Network parameters"""
        # projection
        self.relation_encoder = nn.Embedding(self.relation_vocab_size, self.embedding_size)
        nn.init.xavier_normal(self.relation_encoder.weight)
        self.relation_encoder.weight.data[-1].copy_(torch.zeros(1, self.embedding_size))

        # for encoding key
        if self.separate_key_lstm:
            self.key_lstm = nn.LSTM(input_size = self.embedding_size, 
                                    hidden_size = self.lstm_hidden_size, 
                                    batch_first = True, 
                                    bidirectional = True)

    def get_key_embedding(self, entity, relation, key_mem, key_lens):
        # each is [B, max_num_slots, D]
        e1_embedding = self.entity_encoder(entity)
        r_embedding = self.relation_encoder(relation)

        # key shape is [B, max_num_slots, 2D]
        if self.kb_key_encoder == 'concat':
            kb_key = torch.cat([e1_embedding, r_embedding], dim=2)
        else:
            raise NotImplementedError

        batch_size, max_num_slots, max_key_len = key_mem.size()
        key_mem = key_mem.view(batch_size * max_num_slots, max_key_len)
        # reshape the data to [(B, max_num_slots), max_key_len, D]
        key_embedding = self.entity_encoder(key_mem)
        if self.text_key_encoder == 'lstm':
            zero_states = (Variable(torch.zeros(2, batch_size, self.embedding_size)).cuda(),
                            Variable(torch.zeros(2, batch_size, self.embedding_size)).cuda())                             
            if self.separate_key_lstm:
                outputs, (h_n, c_n) = self.key_lstm(key_embedding)
            else:
                outputs, (h_n, c_n) = self.question_lstm(key_embedding)
            # [(B, max_num_slots), 2D]
            text_key = outputs[:, -1, :]
        else:
            raise NotImplementedError
        text_key = text_key.view(batch_size, max_num_slots, 2 * self.embedding_size)    
        return kb_key, text_key

    def __call__(self, memory, key_mem, key_len, val_mem, question, question_lengths):
        """ 
        TextKBQA 

        Parameters
        ----------------
        memory: FloatTensor (batch_size, memory_size, 3)
        key_mem: FloatTensor (batch_size, key_memory_size, max_key_len)
        key_len: FloatTensor (batch_size, key_memory_size)
        val_mem: FloatTensor (batch_size, key_memory_size)
        question: FloatTensor (batch_size, max_len)
        question_lengths: FloatTensor (batch_size, )

        Returns
        ---------------
        logits: FloatTensor (batch_size, self.memory_vocab_size)
        """
        # split memory and get corresponding embeddings
        e1 = memory[:, :, 0] # (batch_size, memory_size)
        r = memory[:, :, 1] # (batch_size, memory_size)
        e2 = memory[:, :, 2] # (batch_size, memory_size)
        kb_C = -1000 # float
        kb_mask = (e1 != (self.entity_vocab_size - 1)).float() # (batch_size, memory_size)
        kb_value = self.get_value_embedding(e2) # (batch_size, memory_size, embedding_size)

        # key_mem is [B, max_num_mem, max_key_len]
        # key_len is [B, max_num_mem]
        # val_mem is [B, max_num_mem]
        text_C = -1000 # float
        text_mask = (key_len != 0).float() # (batch_size, key_memory_size)
        text_value = self.get_value_embedding(val_mem) # (batch_size, key_memory_size, embedding_size)

        kb_key, text_key = self.get_key_embedding(e1, r, key_mem, key_len) # ((batch_size, memory_size, 2 * embedding_size), (batch_size, key_memory_size, 2 * embedding_size))
        ques = self.get_question_embedding(question, question_lengths) # (batch_size, 2 * embedding_size)

        # get attention on retrived informations based on the question
        # kb_attn_ques = self.seek_attention(ques, kb_key, kb_value, kb_C, kb_mask)  # (batch_size, 2 * embedding_size)
        # text_attn_ques = self.seek_attention(ques, text_key, text_value, text_C, text_mask)  # (batch_size, 2 * embedding_size)

        if self.join == 'batch_norm':
            # mean_kb_key = torch.mean(kb_key, dim = -1)
            # var_kb_key = torch.var(kb_key, dim=-1)
            # mean_kb_value = torch.mean(kb_value, dim = -1)
            # var_kb_value = torch.var(kb_value, dim=-1)
            # text_key = F.batch_norm(text_key, mean_kb_key, var_kb_key, weight = var_kb_key, bias = mean_kb_key, training = False, eps=1e-8)
            # #  * var_kb_key.unsqueeze(-1).expand_as(text_key) + mean_kb_key.unsqueeze(-1).expand_as(text_key)
            # text_value = F.batch_norm(text_value, mean_kb_value, var_kb_value, weight = var_kb_value, bias = mean_kb_value, training = False, eps=1e-8)
            # # * var_kb_value.unsqueeze(-1).expand_as(text_value) + mean_kb_value.unsqueeze(-1).expand_as(text_value)

            merged_key = torch.cat([kb_key, text_key], dim=1)
            merged_value = torch.cat([kb_value, text_value], dim=1)
            merged_C = -1000
            merged_mask = torch.cat([kb_mask, text_mask], dim=1)

            # get attention on retrived informations based on the question
            attn_ques = self.seek_attention(ques, merged_key, merged_value, merged_C, merged_mask)  # [B, 2D]
            model_answer = self.Linear2(attn_ques)  # model_answer: [B, D]

        logits = self.entity_decoder(model_answer)
        return logits
    
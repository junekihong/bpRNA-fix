

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from collections import defaultdict
from pprint import pprint
import math, random

from multiprocessing import Process, Queue, Event
#from torch.multiprocessing import Process, Queue, Event
from parse import decode, brackets, bt
from parse import nussinov

import threading
from multiprocessing import Process
from multiprocessing.pool import ThreadPool



import subprocess
def run_RNAfold(RNA):
    output = subprocess.check_output("echo " + str(RNA) + " | RNAfold -p", shell=True, universal_newlines=True)
    lines = output.split('\n')
    dot_bracket = lines[1].split(' ')[0]
    return dot_bracket

def read_RNAfold_dot():
    lines = open("dot.ps", "r").readlines()


    start_string = "%start of base pair probability data\n"
    end_string = "showpage\n"
    start_index = 0
    end_index = len(lines) - 1

    for index, line in enumerate(lines):
        if line == start_string:
            start_index = index + 1
        if line == end_string:
            end_index = index
    lines = [line.strip().split() for line in lines[start_index:end_index]]

    upper = {}
    lower = {}
    for line in lines:
        i, j, value = line[:-1]
        i, j, value = int(i)-1, int(j)-1, float(value)
        if line[3] == "ubox":
            #upper.append(line[:-1])
            upper[i,j] = value
        if line[3] == "lbox":
            #lower.append(line[:-1])
            lower[i,j] = value
    return upper, lower




class PositionalEncoding(nn.Module):
    # Taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class Network(nn.Module):
    def __init__(self,
                 seq_vocab, bracket_vocab,
                 batch_size,
                 dropout=0.1,
    ):
        super(Network, self).__init__()
        self.seq_vocab = seq_vocab
        self.bracket_vocab = bracket_vocab
        self.batch_size = batch_size
        self.dropout = dropout
        self.relu = nn.ReLU()

        self.d_model = 256 #128 #256 #512
        self.transformer_ff = 1024 #512 #1024 #2048
        self.ff_paired_size = 256 #64

        self.transformer_heads = 8
        self.transformer_layers = 4
        
        self.ff_enc = nn.Linear(seq_vocab.size, self.d_model)
        #self.ff_dec = nn.Linear(bracket_vocab.size, self.d_model)

        self.lstm_enc = nn.LSTM(seq_vocab.size + bracket_vocab.size, 
                                self.d_model//2,
                                num_layers=2,
                                dropout=dropout,
                                bidirectional=True)

        #self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        self.transformer = nn.Transformer(
            self.d_model, 
            nhead=self.transformer_heads,
            num_encoder_layers=self.transformer_layers,
            num_decoder_layers=self.transformer_layers,
            dim_feedforward=self.transformer_ff,
            dropout=self.dropout,
        )

        
        #self.ff_paired = nn.Linear(self.d_model*2, 2)
        self.ff_paired = nn.Sequential(
            nn.Linear(self.d_model*2, self.ff_paired_size), 
            nn.Linear(self.ff_paired_size, 2))

        self.ff_final = nn.Linear(self.d_model, self.bracket_vocab.size)
        #self.trainer = optim.Adam(self.parameters(), lr=0.00125, betas=(0.9, 0.98), eps=1e-09)
        self.trainer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-09)


    def encode(self, X):
        # Encoder
        zero = torch.zeros((4, X.shape[1], self.d_model//2), device=X.device)
        init_zeros = (zero, zero)
        X, hidden = self.lstm_enc(X, init_zeros)

        # Transformer
        #Y_masked = self.relu(self.ff_dec(Y_masked))
        #X = self.transformer(X, lstm_out,
        #                     tgt_mask=subsequent_mask(len(X)))

        #X = self.ff_enc(X)
        #X = self.pos_encoder(X)
        X = self.transformer.encoder(X)
        return X




    def forward(self, X):
        X = self.encode(X)

        """
        def subsequent_mask(size):
            "Mask out subsequent positions."
            mask = (torch.triu(torch.ones(size, size, device=X.device)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

        zero = torch.zeros((4, X.shape[1], self.d_model//2), device=X.device)
        init_zeros = (zero, zero)

        # Encoder
        lstm_out, hidden = self.lstm_enc(X, init_zeros)

        #print(X.size())
        #print(lstm_out.size())

        # Transformer
        #Y_masked = self.relu(self.ff_dec(Y_masked))
        X = self.transformer(lstm_out, lstm_out,
                             tgt_mask=subsequent_mask(len(X)))

        #X = self.transformer.encoder(lstm_out)
        #X = self.transformer.encoder(X)

        # Decoder
        #X, hidden = self.lstm_dec(X, zeros)
        #X = self.ff_final(lstm_out)
        #X = F.log_softmax(X, dim=2)
        """

        X = self.ff_final(X)
        return X

 

    def save(self, filename):
        torch.save({
            "state_dict": self.state_dict(),
            "seq_vocab": self.seq_vocab,
            "bracket_vocab": self.bracket_vocab,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
        }, filename)


    @staticmethod
    def load(filename, device=None):
        if device is not None:
            checkpoint = torch.load(filename, map_location=device)
        else:
            checkpoint = torch.load(filename, map_location=torch.device("cpu"))

        seq_vocab     = checkpoint["seq_vocab"]
        bracket_vocab = checkpoint["bracket_vocab"]
        batch_size    = checkpoint["batch_size"]
        dropout       = checkpoint["dropout"]

        network = Network(seq_vocab, bracket_vocab,
                          batch_size,
                          dropout=dropout,
                      ).to(device)
        network.load_state_dict(checkpoint["state_dict"])
        return network


    def matrix_predict(self, examples, X, return_loss=False):
        batch_size = len(examples)
        memory = self.encode(X)

        predicted_matrices = []
        if return_loss:
            loss_function = torch.nn.MSELoss(reduction="sum")
            losses = []

        for batch,(ID,seq,annotation,dotbracket) in enumerate(examples):
            n = len(seq)
            mem = memory[:n,batch,:]
            mem_i = torch.stack([mem]*n, 0)
            mem_j = torch.stack([mem]*n, 1)
            mem_ij = torch.cat([mem_i,mem_j], 2)
            mem_paired = self.ff_paired(mem_ij)
            mem_paired = torch.nn.functional.softmax(mem_paired, dim=2)
            predicted_matrices.append(mem_paired)

            dotbracket = brackets(dotbracket)
            gold_mask = torch.zeros(size=(mem_paired.size(0), mem_paired.size(1), 1), device=memory.device, dtype=torch.bool)


            gold_mask_size = mem_paired.size(0) * mem_paired.size(1)
            negative_weight = (2*len(dotbracket)) / gold_mask_size
            positive_weight = 1.0 - negative_weight



            negative_delta = 0.5 - (negative_weight/2)

            for i,j in dotbracket:
                gold_mask[i,j] = True
                gold_mask[j,i] = True
            gold_matrix = torch.where(gold_mask,
                                      torch.tensor([0.5 - (positive_weight/2), 0.5 + (positive_weight/2)], device=memory.device, dtype=torch.float),
                                      torch.tensor([0.5 + (negative_weight/2), 0.5 - (negative_weight/2)], device=memory.device, dtype=torch.float),
            )



            """
            neg_gold_matrix = torch.ones(size=gold_matrix.size(), device=memory.device) - gold_matrix
            TP = mem_paired * gold_matrix
            FP = mem_paired * neg_gold_matrix
            FN = (torch.ones(size=mem_paired.size(), device=memory.device) - mem_paired) * gold_matrix
            F1 = -2 * TP / (2 * TP + FP + FN)
            loss = torch.sum(F1)
            """



            """
            if return_loss:
                predicted_dotbracket, predicted_score, gold_score = nussinov(mem_paired, dotbracket, return_loss=return_loss)
                predicted_dotbrackets.append(predicted_dotbracket)
                predicted_scores.append(predicted_score)
                gold_scores.append(gold_score)
            else:
                predicted_dotbracket, predicted_score = nussinov(mem_paired, dotbracket, return_loss=return_loss)
                predicted_scores.append(predicted_score)
                predicted_dotbrackets.append(predicted_dotbracket)
            """

            loss = loss_function(mem_paired, gold_matrix)
            losses.append(loss)

            
        #predicted_scores = torch.stack(predicted_scores, 0)
        if return_loss:
            #gold_scores = torch.stack(gold_scores, 0)
            #loss = torch.sum(predicted_scores - gold_scores)
            #return loss, predicted_dotbrackets
            losses = torch.stack(losses, 0)
            loss = torch.sum(losses)
            return loss, predicted_matrices
        else:
            return predicted_matrices



    def matrix_predict_softmax(self, examples, X, return_loss=False):
        batch_size = len(examples)
        memory = self.encode(X)

        loss_augmented_predicted_scores = []
        loss_augmented_predicted_dotbrackets = []

        predicted_scores = []
        predicted_dotbrackets = []
        if return_loss:
            losses = []
            loss_function = torch.nn.modules.loss.MSELoss()

        for batch,(ID,seq,dotbracket) in enumerate(examples):
            n = len(seq)
            mem = memory[:n,batch,:]
            mem_i = torch.stack([mem]*n, 0)
            mem_j = torch.stack([mem]*n, 1)
            mem_ij = torch.cat([mem_i,mem_j], 2)
            mem_paired = self.ff_paired(mem_ij)

            probs = torch.nn.functional.softmax(mem_paired, dim=2)[:,:,1]
            probs = torch.triu(probs, diagonal=1)

            if return_loss:
                dotbracket = brackets(dotbracket)
                gold_matrix = torch.zeros(size=(mem_paired.size(0), mem_paired.size(1)), device=memory.device)
                for i,j in dotbracket:
                    gold_matrix[i,j] = 1
                gold_matrix = torch.triu(gold_matrix, diagonal=1)
                
                loss = loss_function(probs, gold_matrix)
                losses.append(loss)



            quick_predictions = torch.where(probs > 0.5, 
                                            torch.ones(size=probs.size(), device=memory.device), 
                                            torch.zeros(size=probs.size(), device=memory.device),
            )
            nonzero_preds = torch.nonzero(quick_predictions)
            predicted_dotbracket = ["."]*n            
            for i,j in nonzero_preds:
                if predicted_dotbracket[i] != "." or \
                   predicted_dotbracket[j] != ".":
                    continue
                predicted_dotbracket[i] = "("
                predicted_dotbracket[j] = ")"
            predicted_dotbracket = "".join(predicted_dotbracket)
            predicted_dotbrackets.append(predicted_dotbracket)


            predicted_score = torch.sum(probs * quick_predictions)
            predicted_scores.append(predicted_score)

        predicted_scores = torch.stack(predicted_scores, 0)
        if return_loss:
            loss = torch.sum(torch.stack(losses, 0))
            return loss, predicted_dotbrackets
        else:
            return predicted_scores, predicted_dotbrackets

            


    def nussinov_parse_fast(self, examples, X, return_loss=False, use_RNAfold_weights=False):
        batch_size = len(examples)
        memory = self.encode(X)

        nussinov = []
        for _,seq,_,_ in examples:
            n = len(seq)
            value = torch.zeros((n,n), device=memory.device)
            bp    = torch.zeros((n,n), device=memory.device).fill_(-1)
            dp_table = torch.stack([value,bp], 2)
            nussinov.append(dp_table)

        if return_loss:
            gold_parse = []
            for dp_table in nussinov:
                gold_parse.append(dp_table.clone())
            dotbrackets = [brackets(dotbracket) for ID,seq,annotation,dotbracket in examples]
            dotbrackets_rightbracket_lookup = [{j:i for i,j in dotbracket} for dotbracket in dotbrackets]


        for batch,(ID,seq,annotation,dotbracket) in enumerate(examples):
            n = len(seq)

            mem = memory[:n,batch,:]
            mem_i = torch.stack([mem]*n, 0)
            mem_j = torch.stack([mem]*n, 1)
            mem_ij = torch.cat([mem_i,mem_j], 2)
            mem_paired = self.ff_paired(mem_ij)            
            mem_paired = torch.nn.functional.softmax(mem_paired, dim=2)

            if use_RNAfold_weights:
                rnafold = run_RNAfold(seq)
                upper, lower = read_RNAfold_dot()
                
                mem_paired = torch.zeros(size=mem_paired.size(), dtype=torch.float, device=mem.device)
                for i in range(n-1):
                    for j in range(i+1, n):
                        if (i,j) in upper:
                            value = upper[i,j]
                            mem_paired[i,j,0] = 0
                            mem_paired[i,j,1] = value
                            mem_paired[j,i,0] = 0
                            mem_paired[j,i,1] = value
                        else:
                            mem_paired[i,j,0] = 0.0001
                            mem_paired[i,j,1] = 0
                            mem_paired[j,i,0] = 0.0001
                            mem_paired[j,i,1] = 0
                #for (i,j),value in upper.items():
                #    mem_paired[i,j,0] = 0
                #    mem_paired[i,j,1] = value
                dotbrackets[batch] = lower.keys()
                dotbrackets_rightbracket_lookup[batch] = {j:i for i,j in dotbrackets[batch]}


            for span in range(1, n):
                for i in range(n-span):
                    j = i + span

                    if span == 1:
                        unpaired = mem_paired[i, j, 0]
                        paired = mem_paired[i, j, 1]
                        subscores = torch.cat([unpaired.unsqueeze(0), paired.unsqueeze(0)], 0)
                    else:
                        unpaired = nussinov[batch][i, j-1, 0] + mem_paired[i, j, 0]
                        last_paired = nussinov[batch][i, j-1, 0] + mem_paired[j-1, j, 1]

                        if span == 2:
                            first_paired = mem_paired[i, j, 1]
                            subscores = torch.cat([unpaired.unsqueeze(0), first_paired.unsqueeze(0), last_paired.unsqueeze(0)], 0)
                        else:
                            first_paired = nussinov[batch][i+1, j-1, 0] + mem_paired[i, j, 1]
                            mid_paired = nussinov[batch][i,   i:j-2, 0] + \
                                         nussinov[batch][j-1, i+2:j, 0] + \
                                         mem_paired[i+1:j-1, j, 1]
                            subscores = torch.cat([unpaired.unsqueeze(0), first_paired.unsqueeze(0), mid_paired, last_paired.unsqueeze(0)], 0)

                        
                    #print("(i:{},j:{}) {}".format(i,j, subscores))
                    value, index = torch.max(subscores, 0)

                    index = int(index)
                    if index == 0:
                        backpointer = -1
                    else:
                        # The value of k, the split point
                        backpointer = i + index - 1
                    nussinov[batch][i,j,0] = value
                    nussinov[batch][i,j,1] = backpointer
                    nussinov[batch][j,i,0] = value
                    nussinov[batch][j,i,1] = backpointer

                    if return_loss:
                        subscore = None

                        if j in dotbrackets_rightbracket_lookup[batch]:
                            k = dotbrackets_rightbracket_lookup[batch][j]
                            # First index check
                            if i == k:
                                if span > 2:
                                    paired = gold_parse[batch][i+1, j-1][0] + mem_paired[i, j, 1]
                                else:
                                    paired = mem_paired[i, j, 1]
                                subscore = (paired, i)
                            # Last index check
                            elif k == j-1:
                                left = gold_parse[batch][i, k-1][0]
                                paired = left + mem_paired[k, j, 1]
                                subscore = (paired, k)
                            else:
                                left   = gold_parse[batch][i,   k-1][0]
                                right  = gold_parse[batch][k+1, j-1][0]
                                paired = left + right + mem_paired[k,j,1]
                                subscore = (paired, k)
                        else:
                            unpaired = gold_parse[batch][i, j-1][0] + mem_paired[i, j, 0]
                            subscore = (unpaired, -1)

                        value, backpointer = subscore
                        gold_parse[batch][i,j,0] = value
                        gold_parse[batch][i,j,1] = backpointer
                        gold_parse[batch][j,i,0] = value
                        gold_parse[batch][j,i,1] = backpointer

        scores = []
        backpointers = []
        lengths = []
        if return_loss:
            gold_scores = []
            gold_backpointers = []

        def bt(batch, n, dp_table):
            result = [None] * n
            i, j = 0, n-1
            _, bp = dp_table[batch][i, j]

            visited = {}

            queue = [((i,j),bp)]
            while queue:
                (i,j),k = queue.pop()
                assert i <= j, "({},{},{})".format(i,j,k)

                visited[i,j,k] = True

                i = int(i)
                j = int(j)
                k = int(k)

                if k == -1:
                    result[j] = "."
                    if i < j:
                        _, left = dp_table[batch][i, j-1]
                        assert (i,j-1,left) not in visited, "({},{},{}) -> ({},{},{})".format(i,j,k,i,j-1,left)
                        queue.append(((i, j-1), left))
                else:
                    result[k] = "("
                    result[j] = ")"
                    if i <= k-1:
                        _, left  = dp_table[batch][i,   k-1]
                        queue.append(((i,k-1),left))
                    if k+1 <= j-1:
                        _, right = dp_table[batch][k+1, j-1]
                        queue.append(((k+1,j-1),right))
            return "".join(result)


        for batch in range(len(examples)):
            ID,seq,annotation,dot = examples[batch]
            #print("{:3d} seq  {}".format(batch, seq))
            #print("{:3d} dot  {}".format(batch, dot))

            n = len(seq)
            lengths.append(n)
            score, backpointer = nussinov[batch][0,n-1]
            scores.append(score)
            backpointers.append(backpointer)

            tree = bt(batch, n, nussinov)
            #print("{:3d} pred {} {}".format(batch, tree, score))

            if return_loss:
                gold_score, gold_backpointer = gold_parse[batch][0, n-1]
                gold_scores.append(gold_score)
                gold_backpointers.append(gold_backpointer)

                tree = bt(batch, n, gold_parse)
                #print("{:3d} gold {} {}".format(batch, tree, gold_score))
        
        decoded = []
        for batch, n in enumerate(lengths):
            decode = bt(batch, n, nussinov)
            decoded.append(decode)

        scores = torch.stack(scores, 0)
        if return_loss:
            gold_scores = torch.stack(gold_scores, 0)
            loss = torch.sum(scores - gold_scores)
            return loss, decoded
 
        #print(len(results), [len(x) for x in results], lengths)
        return scores, decoded



    def nussinov_parse_fast_threaded(self, examples, X, return_loss=False, use_RNAfold_weights=False, 
                                     thread_wait_delay=2):
        batch_size = len(examples)
        memory = self.encode(X)

        nussinov = []
        for _,seq,_,_ in examples:
            n = len(seq)
            value = torch.zeros((n,n), device=memory.device)
            bp    = torch.zeros((n,n), device=memory.device).fill_(-1)
            dp_table = torch.stack([value,bp], 2)
            nussinov.append(dp_table)

        if return_loss:
            gold_parse = []
            for dp_table in nussinov:
                gold_parse.append(dp_table.clone())
            dotbrackets = [brackets(dotbracket) for ID,seq,annotation,dotbracket in examples]
            dotbrackets_rightbracket_lookup = [{j:i for i,j in dotbracket} for dotbracket in dotbrackets]


        threadpool = ThreadPool((X.size(0)-1)*thread_wait_delay)
        #threadpool = ThreadPool(64)

        for batch,(ID,seq,annotation,dotbracket) in enumerate(examples):
            n = len(seq)

            mem = memory[:n,batch,:]
            mem_i = torch.stack([mem]*n, 0)
            mem_j = torch.stack([mem]*n, 1)
            mem_ij = torch.cat([mem_i,mem_j], 2)
            mem_paired = self.ff_paired(mem_ij)            
            mem_paired = torch.nn.functional.softmax(mem_paired, dim=2)

            if use_RNAfold_weights:
                rnafold = run_RNAfold(seq)
                upper, lower = read_RNAfold_dot()
                mem_paired = torch.zeros(size=mem_paired.size(), dtype=torch.float, device=mem.device)
                for i in range(n-1):
                    for j in range(i+1, n):
                        if (i,j) in upper:
                            value = upper[i,j]
                            mem_paired[i,j,0] = 0
                            mem_paired[i,j,1] = value
                            mem_paired[j,i,0] = 0
                            mem_paired[j,i,1] = value
                        else:
                            mem_paired[i,j,0] = 0.0001
                            mem_paired[i,j,1] = 0
                            mem_paired[j,i,0] = 0.0001
                            mem_paired[j,i,1] = 0
                #for (i,j),value in upper.items():
                #    mem_paired[i,j,0] = 0
                #    mem_paired[i,j,1] = value
                dotbrackets[batch] = lower.keys()
                dotbrackets_rightbracket_lookup[batch] = {j:i for i,j in dotbrackets[batch]}


            #threads_waiting = {span:[] for span in range(1,n)}
            threads_waiting = {}

            for span in range(1, n):

                #for i in range(n - span):
                def worker(i):
                    j = i + span
                    
                    if span == 1:
                        unpaired = mem_paired[i, j, 0]
                        paired = mem_paired[i, j, 1]
                        subscores = torch.cat([unpaired.unsqueeze(0), paired.unsqueeze(0)], 0)
                    else:
                        unpaired = nussinov[batch][i, j-1, 0] + mem_paired[i, j, 0]
                        last_paired = nussinov[batch][i, j-1, 0] + mem_paired[j-1, j, 1]

                        if span == 2:
                            first_paired = mem_paired[i, j, 1]
                            subscores = torch.cat([unpaired.unsqueeze(0), first_paired.unsqueeze(0), last_paired.unsqueeze(0)], 0)
                        else:
                            first_paired = nussinov[batch][i+1, j-1, 0] + mem_paired[i, j, 1]
                            mid_paired = nussinov[batch][i,   i:j-2, 0] + \
                                         nussinov[batch][j-1, i+2:j, 0] + \
                                         mem_paired[i+1:j-1, j, 1]
                            subscores = torch.cat([unpaired.unsqueeze(0), first_paired.unsqueeze(0), mid_paired, last_paired.unsqueeze(0)], 0)

                        
                    #print("(i:{},j:{}) {}".format(i,j, subscores))
                    value, index = torch.max(subscores, 0)

                    index = int(index)
                    if index == 0:
                        backpointer = -1
                    else:
                        # The value of k, the split point
                        backpointer = i + index - 1
                    nussinov[batch][i,j,0] = value
                    nussinov[batch][i,j,1] = backpointer
                    nussinov[batch][j,i,0] = value
                    nussinov[batch][j,i,1] = backpointer

                    if return_loss:
                        subscore = None
                        if j in dotbrackets_rightbracket_lookup[batch]:
                            k = dotbrackets_rightbracket_lookup[batch][j]
                            # First index check
                            if i == k:
                                if span > 2:
                                    paired = gold_parse[batch][i+1, j-1][0] + mem_paired[i, j, 1]
                                else:
                                    paired = mem_paired[i, j, 1]
                                subscore = (paired, i)
                            # Last index check
                            elif k == j-1:
                                left = gold_parse[batch][i, k-1][0]
                                paired = left + mem_paired[k, j, 1]
                                subscore = (paired, k)
                            else:
                                left   = gold_parse[batch][i,   k-1][0]
                                right  = gold_parse[batch][k+1, j-1][0]
                                paired = left + right + mem_paired[k,j,1]
                                subscore = (paired, k)
                        else:
                            unpaired = gold_parse[batch][i, j-1][0] + mem_paired[i, j, 0]
                            subscore = (unpaired, -1)

                        value, backpointer = subscore
                        gold_parse[batch][i,j,0] = value
                        gold_parse[batch][i,j,1] = backpointer
                        gold_parse[batch][j,i,0] = value
                        gold_parse[batch][j,i,1] = backpointer

                        """
                        return (batch, i,
                                nussinov[batch][i,j,0], 
                                nussinov[batch][i,j,1], 
                                gold_parse[batch][i,j,0],
                                gold_parse[batch][i,j,1],
                            )
                    else:
                        return (batch, i,
                                nussinov[batch][i,j,0], 
                                nussinov[batch][i,j,1], 
                            )
                        """


                """
                for i in range(n - span):
                    worker(i)
                    #print(nussinov[batch][i,i+span,0], nussinov[batch][i,i+span,1])
                    #threadpool.apply(worker, (i,))
                    #print(nussinov[batch][i,i+span,0], nussinov[batch][i,i+span,1])
                    #print()
                    #result = threadpool.apply_async(worker, (i,))
                    #result.wait()
                    #thread_results.append(result)
                """

                #pool.map(worker, range(n - span))
                #threadpool.map_async(worker, range(n - span))
                #for result in thread_results:
                #    #returned = result.get(timeout=1)
                #    #print(returned)
                #    result.wait()

                if span > int(math.sqrt(n)):
                    previous_spans = [previous_span for previous_span in threads_waiting]
                    for previous_span in previous_spans:
                        threads_waiting[previous_span].wait()
                        del(threads_waiting[previous_span])
                elif span > thread_wait_delay:
                    threads_waiting[span-thread_wait_delay].wait()
                    del(threads_waiting[span-thread_wait_delay])

                 
                #print(span)
                #print(threads_waiting)
                result = threadpool.map_async(worker, range(n - span))
                threads_waiting[span] = result
                #result.wait()

        threadpool.close()


        scores = []
        backpointers = []
        lengths = []
        if return_loss:
            gold_scores = []
            gold_backpointers = []

        def bt(batch, n, dp_table):
            result = [None] * n
            i, j = 0, n-1
            _, bp = dp_table[batch][i, j]

            visited = {}

            queue = [((i,j),bp)]
            while queue:
                (i,j),k = queue.pop()
                assert i <= j, "({},{},{})".format(i,j,k)

                visited[i,j,k] = True

                i = int(i)
                j = int(j)
                k = int(k)

                if k == -1:
                    result[j] = "."
                    if i < j:
                        _, left = dp_table[batch][i, j-1]
                        assert (i,j-1,left) not in visited, "({},{},{}) -> ({},{},{})".format(i,j,k,i,j-1,left)
                        queue.append(((i, j-1), left))
                else:
                    result[k] = "("
                    result[j] = ")"
                    if i <= k-1:
                        _, left  = dp_table[batch][i,   k-1]
                        queue.append(((i,k-1),left))
                    if k+1 <= j-1:
                        _, right = dp_table[batch][k+1, j-1]
                        queue.append(((k+1,j-1),right))
            return "".join(result)


        for batch in range(len(examples)):
            ID,seq,annotation,dot = examples[batch]
            #print("{:3d} seq  {}".format(batch, seq))
            #print("{:3d} dot  {}".format(batch, dot))

            n = len(seq)
            lengths.append(n)
            score, backpointer = nussinov[batch][0,n-1]
            scores.append(score)
            backpointers.append(backpointer)

            tree = bt(batch, n, nussinov)
            #print("{:3d} pred {} {}".format(batch, tree, score))

            if return_loss:
                gold_score, gold_backpointer = gold_parse[batch][0, n-1]
                gold_scores.append(gold_score)
                gold_backpointers.append(gold_backpointer)

                tree = bt(batch, n, gold_parse)
                #print("{:3d} gold {} {}".format(batch, tree, gold_score))
        
        decoded = []
        for batch, n in enumerate(lengths):
            decode = bt(batch, n, nussinov)
            decoded.append(decode)

        scores = torch.stack(scores, 0)
        if return_loss:
            gold_scores = torch.stack(gold_scores, 0)
            loss = torch.sum(scores - gold_scores)
            return loss, decoded
 
        #print(len(results), [len(x) for x in results], lengths)
        return scores, decoded




    def nussinov_parse(self, examples, X, return_loss=False):
        batch_size = len(examples)
        memory = self.encode(X)

        nussinov = [defaultdict(lambda: (0, -1))] * batch_size

        if return_loss:
            gold_parse = [defaultdict(lambda: (0, -1))] * batch_size
            dotbrackets = [brackets(dotbracket) for ID,seq,annotation,dotbracket in examples]

        for batch,(ID,seq,annotation,dotbracket) in enumerate(examples):
            n = len(seq)

            mem = memory[:n,batch,:]
            mem_i = torch.stack([mem]*n, 0)
            mem_j = torch.stack([mem]*n, 1)
            mem_ij = torch.cat([mem_i,mem_j], 2)
            mem_paired = self.ff_paired(mem_ij)
            
            for span in range(1, n):
                for i in range(n-span):
                    j = i + span
                    unpaired = nussinov[batch][i,j-1][0] + mem_paired[i,j,0]
                    subscores = [unpaired]
                    for k in range(i, j):
                        left  = nussinov[batch][i,   k-1][0]
                        right = nussinov[batch][k+1, j-1][0]
                        paired = left + right + mem_paired[k,j,1]
                        subscores.append(paired)
                    subscores = torch.stack(subscores,0)
                    value, index = torch.max(subscores, 0)

                    index = int(index)
                    if index == 0:
                        backpointer = -1
                    else:
                        # The value of k, the split point
                        backpointer = i + index - 1

                    nussinov[batch][i,j] = (value, backpointer)

                    if return_loss:
                        subscore = None
                        if (i,j) not in dotbrackets:
                            unpaired = gold_parse[batch][i,j-1][0] + mem_paired[i,j,0]
                            subscore = (unpaired, -1)
                        else:
                            for k in range(i, j):
                                if (k,j) in dotbrackets:
                                    left   = gold_parse[batch][i,   k-1][0]
                                    right  = gold_parse[batch][k+1, j-1][0]
                                    paired = left + right + mem_paired[k,j,1]
                                    subscore = (paired, k)
                                    break
                        value, backpointer = subscore
                        gold_parse[batch][i,j] = (value, backpointer)


        scores = []
        backpointers = []
        lengths = []
        if return_loss:
            gold_scores = []
            gold_backpointers = []

        def bt(batch, n):
            result = [None] * n
            i, j = 0, n-1
            _, bp = nussinov[batch][i, j]

            queue = [((i,j),bp)]
            while queue:
                (i,j),k = queue.pop()
                if k == -1:
                    result[j] = "."
                    if i < j:
                        _, left = nussinov[batch][i, j-1]
                        queue.append(((i, j-1), left))
                else:
                    result[k] = "("
                    result[j] = ")"
                    _, left  = nussinov[batch][i,   k-1]
                    _, right = nussinov[batch][k+1, j-1]
                    queue.append(((i,k-1),left))
                    queue.append(((k+1,j-1),right))
            return "".join(result)


        for batch in range(len(examples)):
            ID,seq,_,_ = examples[batch]
            n = len(seq)
            lengths.append(n)
            score, backpointer = nussinov[batch][0,n-1]
            scores.append(score)
            backpointers.append(backpointer)

            #tree = bt(batch, n)
            #print("{:3d} {}".format(batch, tree))
            if return_loss:
                gold_score, gold_backpointer = gold_parse[batch][0, n-1]
                gold_scores.append(gold_score)
                gold_backpointers.append(gold_backpointer)

        decoded = []
        for batch, n in enumerate(lengths):
            decode = bt(batch, n)
            decoded.append(decode)


        scores = torch.stack(scores, 0)
        if return_loss:
            gold_scores = torch.stack(gold_scores, 0)
            loss = torch.sum(scores - gold_scores)
            return loss, decoded
 
        #print(len(results), [len(x) for x in results], lengths)
        return scores, decoded



    def nussinov_parse_softmaxed(self, examples, X, return_loss=False):
        batch_size = len(examples)
        memory = self.encode(X)

        nussinov = [defaultdict(lambda: (0, None))] * batch_size
        if return_loss:
            gold_parse = [defaultdict(lambda: (0, None))] * batch_size
            dotbrackets = [brackets(dotbracket) for ID,seq,dotbracket in examples]

        for batch,(ID,seq,annotation,dotbracket) in enumerate(examples):
            n = len(seq)

            mem = memory[:n,batch,:]
            mem_i = torch.stack([mem]*n, 0)
            mem_j = torch.stack([mem]*n, 1)
            mem_ij = torch.cat([mem_i,mem_j], 2)
            mem_paired = self.ff_paired(mem_ij)
            
            probs = torch.nn.functional.softmax(mem_paired, dim=2)[:,:,1]
            probs = torch.triu(probs, diagonal=1)


            for span in range(1, n):
                for i in range(n-span):
                    j = i + span
                    
                    unpaired = nussinov[batch][i,j-1][0] + (1 - probs[i,j])
                    subscores = [unpaired]

                    for k in range(i, j):
                        left  = nussinov[batch][i,   k-1][0]
                        right = nussinov[batch][k+1, j-1][0]
                        paired = left + right + probs[k,j]
                        subscores.append(paired)
                    subscores = torch.stack(subscores,0)
                    values, index = torch.max(subscores, 0)

                    index = int(index)
                    if index == 0:
                        backpointer = -1
                    else:
                        # The value of k, the split point
                        backpointer = i + index - 1
                    nussinov[batch][i,j] = (values, backpointer)

                    if return_loss:
                        subscore = None
                        if (i,j) not in dotbrackets:
                            unpaired = gold_parse[batch][i,j-1][0] + (1 - probs[i,j])
                            subscore = (unpaired, -1)
                        else:
                            for k in range(i, j):
                                if (k,j) in dotbrackets:
                                    left   = gold_parse[batch][i,   k-1][0]
                                    right  = gold_parse[batch][k+1, j-1][0]
                                    paired = left + right + probs[k,j]
                                    subscore = (paired, k)
                                    break
                        value, backpointer = subscore
                        gold_parse[batch][i,j] = (value, backpointer)


        scores = []
        backpointers = []
        lengths = []
        if return_loss:
            gold_scores = []
            gold_backpointers = []

        def bt(batch, n):
            result = [None] * n
            i, j = 0, n-1
            _, bp = nussinov[batch][i, j]

            queue = [((i,j),bp)]
            while queue:
                (i,j),k = queue.pop()
                if k is None:
                    if result[j] is None:
                        result[j] = "."
                elif k == -1:
                    result[j] = "."
                    _, left = nussinov[batch][i, j-1]
                    queue.append(((i, j-1), left))
                else:
                    result[k] = "("
                    result[j] = ")"
                    _, left  = nussinov[batch][i,   k-1]
                    _, right = nussinov[batch][k+1, j-1]
                    queue.append(((i,k-1),left))
                    queue.append(((k+1,j-1),right))
            return "".join(result)


        for batch in range(len(examples)):
            ID,seq,_,_ = examples[batch]
            n = len(seq)
            lengths.append(n)
            score, backpointer = nussinov[batch][0,n-1]
            scores.append(score)
            backpointers.append(backpointer)

            #tree = bt(batch, n)
            #print("{:3d} {}".format(batch, tree))
            if return_loss:
                gold_score, gold_backpointer = gold_parse[batch][0, n-1]
                gold_scores.append(gold_score)
                gold_backpointers.append(gold_backpointer)

        decoded = []
        for batch, n in enumerate(lengths):
            decode = bt(batch, n)
            decoded.append(decode)
           
        scores = torch.stack(scores, 0)
        if return_loss:
            gold_scores = torch.stack(gold_scores, 0)
            loss = torch.sum(scores - gold_scores)
            return loss, decoded
 
        #print(len(results), [len(x) for x in results], lengths)
        return scores, decoded



    def parse_SR(self, examples, X, return_loss=False):
        batch_size = len(examples)
        memory = self.encode(X)
        #beamsize = 8 #16

        mem_i = torch.stack([memory]*memory.size(0), 0)
        mem_j = torch.stack([memory]*memory.size(0), 1)
        mem_ij = torch.cat([mem_i,mem_j], 3)
        mem_paired = self.ff_paired(mem_ij)
        
        N = X.size(0)

        loss = 0

        """
        if return_loss:
            dotbrackets = [brackets(dotbracket) for seq,dotbracket in examples]
            shortest_spanning_bracket = [{i:(None,None) for i in range(len(seq))} for seq,_ in examples]
            for batch,dotbracket in enumerate(dotbrackets):
                for i,j in sorted(dotbracket):
                    for k in range(i,j+1):
                        shortest_spanning_bracket[batch][k] = (i,j)
        """


        #predicted = [None for _ in range(len(examples))]
        #losses    = [0    for _ in range(len(examples))]
        return_queue = Queue()
        
        """
        for batch in range(len(examples)):
            decoded_loss = decode(batch, predicted, losses)
            #loss += decoded_loss
        """


        """
        for batch in range(len(examples)):
            print("Just Running: {}".format(batch))
            decode(batch, examples[batch], mem_paired[:,:,batch,:], return_queue, return_loss)
            print("Finished {}".format(batch))
        """

        

        
        
        """
        batch = 0
        print(examples[batch][0])
        print(examples[batch][1])
        decode(batch, examples[batch], mem_paired[:,:,batch,:].cpu().data.numpy(), return_queue, True)
        items = return_queue.get()
        _, top_predicted, gold_decode = items
        predicted_dotbracket, _ = bt(top_predicted, mem_paired[:,:,batch,:])
        print(top_predicted)
        print(predicted_dotbracket)

        exit()
        decode(batch, examples[batch], mem_paired[:,:,batch,:].cpu().data.numpy(), return_queue, False)
        items = return_queue.get()
        _, top_predicted = items
        predicted_dotbracket, _ = bt(top_predicted, mem_paired[:,:,batch,:])
        print(top_predicted)
        print(predicted_dotbracket)
        exit()
        """



        processes = []
        for batch in range(len(examples)):
            #print("{}: Starting job".format(batch))
            p = Process(target=decode, args=(batch, examples[batch], mem_paired[:,:,batch,:].cpu().data.numpy(), return_queue, return_loss))
            p.start()
            processes.append(p)



        losses = []
        results = []

        ended_workers = 0
        while ended_workers < len(examples):
            items = return_queue.get()
            batch_index = items[0]
            ended_workers += 1

            top_predicted = items[1]

            #print(batch_index, top_predicted)
            if return_loss:
                gold_decode = items[2]
                predicted_dotbracket, loss = bt(top_predicted, 
                                                mem_paired[:,:,batch_index,:],
                                                gold_decode)
                #_, gold_score = bt(gold_decode, mem_paired[:,:,batch_index,:])
                #loss = top_score - gold_score
                results.append((batch_index, loss, predicted_dotbracket))
            else:
                predicted_dotbracket, top_score = bt(top_predicted, 
                                                     mem_paired[:,:,batch_index,:])

                results.append((batch_index, top_score, predicted_dotbracket))

        if return_loss:
            batch_indices, losses, predicted_dotbrackets = zip(*sorted(results))
            loss = sum(losses)
            return loss, predicted_dotbrackets
        else:
            batch_indices, scores, predicted_dotbrackets = zip(*sorted(results))
            predicted_scores = torch.stack(scores, 0)
            return predicted_scores, predicted_dotbrackets


    def DP_fix(self, examples, X, return_loss=False):
        batch_size = len(examples)
        #memory = self.encode(X)
        memory = self(X)
        memory = memory[:,:,:-1]
        print(memory.size())

        for batch,(ID,seq,annotation,dotbracket) in enumerate(examples):
            n = len(seq)
            mem = memory[:n,batch,:]
            print(mem.size())


        exit()



        nussinov = [defaultdict(lambda: (0, -1))] * batch_size

        if return_loss:
            gold_parse = [defaultdict(lambda: (0, -1))] * batch_size
            dotbrackets = [brackets(dotbracket) for ID,seq,annotation,dotbracket in examples]

        for batch,(ID,seq,annotation,dotbracket) in enumerate(examples):
            n = len(seq)

            mem = memory[:n,batch,:]
            mem_i = torch.stack([mem]*n, 0)
            mem_j = torch.stack([mem]*n, 1)
            mem_ij = torch.cat([mem_i,mem_j], 2)
            mem_paired = self.ff_paired(mem_ij)
            
            for span in range(1, n):
                for i in range(n-span):
                    j = i + span
                    unpaired = nussinov[batch][i,j-1][0] + mem_paired[i,j,0]
                    subscores = [unpaired]
                    for k in range(i, j):
                        left  = nussinov[batch][i,   k-1][0]
                        right = nussinov[batch][k+1, j-1][0]
                        paired = left + right + mem_paired[k,j,1]
                        subscores.append(paired)
                    subscores = torch.stack(subscores,0)
                    value, index = torch.max(subscores, 0)

                    index = int(index)
                    if index == 0:
                        backpointer = -1
                    else:
                        # The value of k, the split point
                        backpointer = i + index - 1

                    nussinov[batch][i,j] = (value, backpointer)

                    if return_loss:
                        subscore = None
                        if (i,j) not in dotbrackets:
                            unpaired = gold_parse[batch][i,j-1][0] + mem_paired[i,j,0]
                            subscore = (unpaired, -1)
                        else:
                            for k in range(i, j):
                                if (k,j) in dotbrackets:
                                    left   = gold_parse[batch][i,   k-1][0]
                                    right  = gold_parse[batch][k+1, j-1][0]
                                    paired = left + right + mem_paired[k,j,1]
                                    subscore = (paired, k)
                                    break
                        value, backpointer = subscore
                        gold_parse[batch][i,j] = (value, backpointer)


        scores = []
        backpointers = []
        lengths = []
        if return_loss:
            gold_scores = []
            gold_backpointers = []

        def bt(batch, n):
            result = [None] * n
            i, j = 0, n-1
            _, bp = nussinov[batch][i, j]

            queue = [((i,j),bp)]
            while queue:
                (i,j),k = queue.pop()
                if k == -1:
                    result[j] = "."
                    if i < j:
                        _, left = nussinov[batch][i, j-1]
                        queue.append(((i, j-1), left))
                else:
                    result[k] = "("
                    result[j] = ")"
                    _, left  = nussinov[batch][i,   k-1]
                    _, right = nussinov[batch][k+1, j-1]
                    queue.append(((i,k-1),left))
                    queue.append(((k+1,j-1),right))
            return "".join(result)


        for batch in range(len(examples)):
            ID,seq,_,_ = examples[batch]
            n = len(seq)
            lengths.append(n)
            score, backpointer = nussinov[batch][0,n-1]
            scores.append(score)
            backpointers.append(backpointer)

            #tree = bt(batch, n)
            #print("{:3d} {}".format(batch, tree))
            if return_loss:
                gold_score, gold_backpointer = gold_parse[batch][0, n-1]
                gold_scores.append(gold_score)
                gold_backpointers.append(gold_backpointer)

        decoded = []
        for batch, n in enumerate(lengths):
            decode = bt(batch, n)
            decoded.append(decode)


        scores = torch.stack(scores, 0)
        if return_loss:
            gold_scores = torch.stack(gold_scores, 0)
            loss = torch.sum(scores - gold_scores)
            return loss, decoded
 
        #print(len(results), [len(x) for x in results], lengths)
        return scores, decoded


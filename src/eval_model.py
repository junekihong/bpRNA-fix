#!/usr/bin/python3

import sys, os

from network import Network

import torch
import torch.nn as nn

import subprocess


from main import prepare_batch

usage = "python3 eval.py [model_filename] [sequence] [input_dotbracket]"
if len(sys.argv) != len(usage.split())-1:
    print(usage)
    exit()


_, model_filename, sequence, in_structure = sys.argv

"""
linearfold_path = "/nfs0/BB/Hendrix_Lab/hongju/LinearFold/linearfold"
linearfold_CMD = "{}".format(linearfold_path)
echo_CMD = "echo -e {}".format(sequence)
echo = subprocess.Popen(echo_CMD.split(),
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
output = subprocess.check_output(linearfold_CMD.split(), stdin=echo.stdout)
echo.wait()
in_structure = output.split()[1].decode("utf-8")
"""

GPU = 0

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(GPU) if use_cuda and GPU is not None else "cpu")
print("Selecting device: {}".format(device))
print("Loading model... ", end="")
network = Network.load(model_filename).to(device)
network.eval()
print("done.")

#while True:
#sequence = input()
#in_structure = input()
IDs, X, Y, gold = prepare_batch([("input", sequence, in_structure, "")], network.seq_vocab, network.bracket_vocab, device)
#print(IDs)
#print(X)
#print(Y)
#print(gold)
output = network(X)

# Eliminate the mini-batch dimension
output = output.view(output.size(0), -1)

#print(output)
# Argmax
values, predictions = torch.max(output, 1)

output_seq = []
for pred in predictions.data:
    pred = network.bracket_vocab.value(int(pred.data))
    output_seq.append(pred)
output = "".join(output_seq)


print("sequence length:", len(sequence))
print("SEQUENCE:    ", sequence)
print("INPUT STRUC: ", in_structure)
print("OUTPUT STRUC:", output)



    


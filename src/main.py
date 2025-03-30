#!/usr/bin/python3

import argparse
import sys, os, math
import time, datetime
import itertools, random
from pprint import pprint, pformat
from collections import defaultdict

import numpy as np
import vocabulary
from network import Network
from const import START, STOP, UNK
from utils import format_elapsed, read_dotbrackets
from utils import read_bpRNA_directory, default_data_split

from Preprocess_Evaluate_bpRNA.evaluate_dotbrackets import add_confusion_matrices, get_position_accuracies, str_position_accuracies, get_bp_accuracies, str_basepair_accuracies, get_PRF, get_MCC, get_sequence_f1scores


import torch
import torch.nn as nn
from matplotlib import pyplot as plt



def out(logfile, x="\n", end="\n"):
    print(x, end=end)
    sys.stdout.flush()
    logfile.write(str(x)+end)
    logfile.flush()

def process_vocabulary(args, data):
    """
    Creates and returns vocabulary objects.
    Only iterates through the first 100 sequences, out of interest of computation.
    """
    out(args.logfile, "initializing vocabularies... ", end="")
    seq_vocab = vocabulary.Vocabulary()
    #seq_vocab.index(START)
    #seq_vocab.index(STOP)
    bracket_vocab = vocabulary.Vocabulary()
    #bracket_vocab.index(START)
    #bracket_vocab.index(STOP)

    for ID,sequence,dotbracket in data[:100]:
        for character in sequence:
            seq_vocab.index(character)
        for character in dotbracket:
            bracket_vocab.index(character)
    seq_vocab.index(UNK)
    seq_vocab.freeze()
    bracket_vocab.index(UNK)
    bracket_vocab.freeze()
    out(args.logfile, "done.")

    def print_vocabulary(name, vocab):
        #special = {START, STOP, UNK}
        special = {UNK}
        out(args.logfile, "{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))
    if args.print_vocabs:
        print_vocabulary("Sequence", seq_vocab)
        print_vocabulary("Brackets", bracket_vocab)

    return seq_vocab, bracket_vocab


def prepare_batch(batch, seq_vocab, bracket_vocab, device,
                  masked=False,
              ):
    IDs, X1, X2, Y = zip(*batch)

    longest_length = max([len(x) for x in X1])
    sequences = []
    predicted_structures = []
    dotbrackets = []

    for x in X1:
        length_diff = longest_length - len(x)
        sequence = [seq_vocab.index(char) if \
                    char in seq_vocab.indices else \
                    seq_vocab.index(UNK) for char in x] + \
            [seq_vocab.index(UNK)] * length_diff
        sequences.append(sequence)
    for x in X2:
        length_diff = longest_length - len(x)
        sequence = [bracket_vocab.index(char) if \
                    char in bracket_vocab.indices else \
                    bracket_vocab.index(UNK) for char in x] + \
            [bracket_vocab.index(UNK)] * length_diff
        predicted_structures.append(sequence)
    for y in Y:
        length_diff = longest_length - len(y)
        dotbracket = [bracket_vocab.index(char) if \
                      (not masked or i < len(y)//2) else bracket_vocab.index(UNK) \
                      for i,char in enumerate(y)] + \
                          [bracket_vocab.index(UNK)] * length_diff
        dotbrackets.append(dotbracket)

    # Initial encoding of X and Y
    X1 = torch.tensor(sequences, device=device)
    X2 = torch.tensor(predicted_structures, device=device)
    Y = torch.tensor(dotbrackets, device=device)
    gold = Y

    # One-Hot encoding
    ones_seq = torch.sparse.torch.eye(seq_vocab.size, device=device)
    ones_dot = torch.sparse.torch.eye(bracket_vocab.size, device=device)

    X1 = torch.cat([ones_seq.index_select(0, x).view(longest_length, 1, seq_vocab.size) for x in X1], 1)
    X2 = torch.cat([ones_dot.index_select(0, x).view(longest_length, 1, bracket_vocab.size) for x in X2], 1)
    Y = torch.cat([ones_dot.index_select(0, y).view(longest_length, 1, bracket_vocab.size) for y in Y], 1)

    X = torch.cat([X1,X2], 2)
    return IDs, X, Y, gold



def run_train(args):
    #dotbrackets = read_bpRNA_directory(args.bpRNA_directory)
    train_data = read_dotbrackets(args.train)
    val_data   = read_dotbrackets(args.val)
    
    train_prediction_data = read_dotbrackets(args.train_predictions)
    val_prediction_data = read_dotbrackets(args.val_predictions)

    
    IDs_to_delete_train = []
    IDs_to_delete_val = []
    for ID in train_data:
        if ID not in train_prediction_data:
            IDs_to_delete_train.append(ID)
    for ID in val_data:
        if ID not in val_prediction_data:
            IDs_to_delete_val.append(ID)
    for ID in IDs_to_delete_train:
        del(train_data[ID])
    for ID in IDs_to_delete_val:
        del(val_data[ID])


    """
    for ID,value in list(train_prediction_data.items()):
        trunc_ID = "|".join(ID.split("|")[:-1])
        train_prediction_data[trunc_ID] = value
    for ID,value in list(val_prediction_data.items()):
        trunc_ID = "|".join(ID.split("|")[:-1])
        val_prediction_data[trunc_ID] = value
    """


    def remove_pk(brackets):
        brackets = brackets.replace("[", ".").replace("]", ".")
        brackets = brackets.replace("{", ".").replace("}", ".")
        brackets = brackets.replace("<", ".").replace(">", ".")
        brackets = brackets.replace("A", ".").replace("a", ".")
        brackets = brackets.replace("B", ".").replace("b", ".")
        brackets = brackets.replace("C", ".").replace("c", ".")
        brackets = brackets.replace("D", ".").replace("d", ".")
        return brackets

    if args.max_length != 200:
        train_data = [(k,v[0],remove_pk(v[1])) for k,v in train_data.items() if len(v[0]) <= args.max_length]
        val_data   = [(k,v[0],remove_pk(v[1])) for k,v in val_data.items()   if len(v[0]) <= args.max_length]
    else:
        train_data = [(k,v[0],remove_pk(v[1])) for k,v in train_data.items() if len(v[0]) <= 200]
        val_data   = [(k,v[0],remove_pk(v[1])) for k,v in val_data.items() if len(v[0]) <= 200]
    seq_vocab, bracket_vocab = process_vocabulary(args, train_data)

    """
    for ID,seq,gold_dot in train_data:
        print(ID)
        print(seq)
        print(gold_dot)
        pred = train_prediction_data[ID][1]
        new_train_data.append((ID,seq,pred,gold_dot))
        exit()
    exit()
    """

    train_data = [(ID,seq,train_prediction_data[ID][1],gold_dot) for ID,seq,gold_dot in train_data]
    val_data   = [(ID,seq,val_prediction_data[ID][1],gold_dot) for ID,seq,gold_dot in val_data]


    #train_data = train_data[:100]
    #val_data = val_data[100:200]

    #print(len(train_data))
    #print(len(val_data))
    #exit()


    # Structured prediction is very slow. Only evaluate 10% of the validation.
    #if args.structured:
    #    val_data = val_data[:len(val_data)//20]




    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.GPU) if use_cuda and args.GPU is not None else "cpu")
    out(args.logfile, "Selecting device: {}".format(device))

    out(args.logfile, "Initializing model...")
    if args.load_model is not None:
        out(args.logfile, "Loading model from: {}".format(args.load_model))
        network = Network.load(args.load_model, device)
    else:
        network = Network(
            seq_vocab, bracket_vocab,
            args.batch_size,
            args.dropout,
        ).to(device)
    network.train()
    out(args.logfile, str(network))


    num_parameters = 0
    for parameter in network.parameters():
        sizes = [x for x in parameter.size()]
        prod = 1
        for x in sizes:
            prod *= x
        num_parameters += prod
    out(args.logfile, "# Number of parameters: {}".format(num_parameters))

    total_processed = 0
    check_every = math.floor(len(train_data) / args.checks_per_epoch)
    best_dev_score = -np.inf
    best_dev_model_path = None
    start_time = time.time()
    out(args.logfile, "# Checking dev {} times an epoch (every {} sequences, or every {} batches)".format(args.checks_per_epoch, check_every, check_every//args.batch_size))


    def check_dev():
        nonlocal best_dev_score
        nonlocal best_dev_model_path
        nonlocal val_data
        label = "Dev"

        network.eval()
        dev_start_time = time.time()
        pred_data, tgt_data = [], []

        batched_val = [val_data[start_index:start_index+args.batch_size] \
                       for start_index in range(0, len(val_data), args.batch_size)]
        for batch in batched_val:
            IDs, X, _, gold = prepare_batch(batch, seq_vocab, bracket_vocab, device)
            _, _, _, Y = zip(*batch)

            if args.structured:
                #predicted_scores, predictions = network.nussinov_parse(batch, X)
                #predicted_scores, predictions = network.nussinov_parse_softmaxed(batch, X)
                #predicted_scores, predictions = network.parse_SR(batch, X)


                predicted_scores, predictions = network.nussinov_parse_fast_threaded(batch, X, 
                                                                                     return_loss=False, 
                                                                                     use_RNAfold_weights=args.use_RNAfold_weights,
                                                                                     thread_wait_delay=1,
                )
                pred_data += predictions
                tgt_data += Y

                #batch_loss, predicted_matrices = network.matrix_predict(batch, X, return_loss=True)
                #batch_loss = batch_loss.detach()
                #pred_data.append(batch_loss)
            else:
                output = network(X)

                # Remove <UNK> predictions, Argmax.
                values, predictions = torch.max(output[:,:,:-1], 2)
                
                predictions = ["".join([network.bracket_vocab.value(int(col.data)) for col in row][:len(y)]) for row,y in zip(predictions.transpose(0,1).data, Y)]
                #predictions = [network.bracket_vocab.value(int(pred.data)) for pred in predictions.data]

                pred_data += predictions
                tgt_data += Y

        network.train()

        """
        if args.structured:
            loss = torch.sum(torch.stack(pred_data, 0)) / len(val_data)
        
            out(args.logfile, "# Validation Sequence Average Loss: {}".format(float(loss)))

            if -loss > best_dev_score:
                if best_dev_model_path is not None:
                    path = "{}_dev={:.4f}".format(args.model_path_base, -best_dev_score)
                    if os.path.exists(path):
                        out(args.logfile, "* Removing previous model file {}...".format(path))
                        os.remove(path)

                best_dev_score = -loss
                best_dev_model_path = "{}_dev={:.4f}".format(
                    args.model_path_base, float(loss))
                out(args.logfile, "* Saving new best model to {}...".format(best_dev_model_path))
                network.save(best_dev_model_path)
        else:
        """
        if True:
            out(args.logfile, "# First 3 examples:")
            for i, (pred, tgt) in enumerate(zip(pred_data, tgt_data[:3])):
                ID, seq, prev_pred, dot = val_data[i]
                out(args.logfile, "# ID:   {}".format(ID))
                out(args.logfile, "# Seq:  {}".format(seq))
                out(args.logfile, "# Previous Model Prediction:\n#       {}".format(prev_pred))
                out(args.logfile, "# Pred: {}".format(pred))
                out(args.logfile, "# Tgt : {}".format(tgt))
                out(args.logfile, "# ")
        
            confusion, sequence_accuracy_info = get_position_accuracies(pred_data, tgt_data)
            out(args.logfile, str_position_accuracies(confusion, sequence_accuracy_info), end="")
            bp_summary, balanced_sequences, bracket_counts = get_bp_accuracies(pred_data, tgt_data)
            out(args.logfile, 
                str_basepair_accuracies(bp_summary, balanced_sequences, bracket_counts))
            
            TP, FP, FN = bp_summary
            precision, recall, f1score = get_PRF(TP, FP, FN)
            #network.train()

            if f1score > best_dev_score:
                if best_dev_model_path is not None:
                    path = "{}_dev={:.4f}".format(args.model_path_base, best_dev_score)
                    if os.path.exists(path):
                        out(args.logfile, "* Removing previous model file {}...".format(path))
                        os.remove(path)

                best_dev_score = f1score
                best_dev_model_path = "{}_dev={:.4f}".format(
                    args.model_path_base, f1score)
                out(args.logfile, "* Saving new best model to {}...".format(best_dev_model_path))
                network.save(best_dev_model_path)



    #loss_function = nn.CrossEntropyLoss(reduction="sum")
    loss_function = nn.MSELoss(reduction="sum")
    softmax = nn.Softmax()
    out(args.logfile, "# Training starting. Training set contains {} sequences.".format(len(train_data)))
    out(args.logfile, "# Validation set contains {} sequences.".format(len(val_data)))
    current_processed = 0

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        out(args.logfile, "# Epoch starting. ({} sequences)".format(len(train_data)))
        random.shuffle(train_data)
        batched_train = [train_data[start_index:start_index+args.batch_size] \
                         for start_index in range(0, len(train_data), args.batch_size)]

        epoch_start_time = time.time()
        for batch_index, batch in enumerate(batched_train):
            network.trainer.zero_grad()

            IDs, X, Y, gold = prepare_batch(batch, seq_vocab, bracket_vocab, device)
            total_processed += len(batch)
            current_processed += len(batch)


            if args.structured:
                #batch_loss, predicted_dotbrackets = network.matrix_predict_softmax(batch, X, return_loss=True)
                #batch_loss, predicted_dotbrackets = network.nussinov_parse(batch, X, return_loss=True)




                batch_loss, predicted_dotbrackets = network.nussinov_parse_fast_threaded(batch, X, return_loss=True, use_RNAfold_weights=args.use_RNAfold_weights, thread_wait_delay=3)




                #batch_loss, predicted_dotbrackets = network.parse_SR(batch, X, return_loss=True)
                #batch_loss = network.parse(batch, X, return_loss=True)


                #batch_loss = torch.sum(batch_loss)
                _, _, _, Y = zip(*batch)
                bp_summary, balanced_sequences, bracket_counts = get_bp_accuracies(predicted_dotbrackets, Y)


                """
                batch_loss, predicted_matrices = network.matrix_predict(batch, X, return_loss=True)
                out(args.logfile,
                    "epoch {:,} "
                    "batch {:,}/{:,} "
                    "processed {:,} "
                    "batch-loss {:.4f} "
                    "epoch-elapsed {} "
                    "total-elapsed {} "
                    "".format(
                        epoch,
                        batch_index + 1, len(batched_train),
                        total_processed,
                        float(batch_loss) / len(batch),
                        format_elapsed(epoch_start_time),
                        format_elapsed(start_time),
                    )
                )
                """
                out(args.logfile,
                    "epoch {:,} "
                    "batch {:,}/{:,} "
                    "processed {:,} "
                    "batch-loss {:.4f} "
                    "epoch-elapsed {} "
                    "total-elapsed {} "
                    "   bp-acc {} "
                    "bal-seqs {} "
                    "brckt-cnts {} "
                    "".format(
                        epoch,
                        batch_index + 1, len(batched_train),
                        total_processed,
                        float(batch_loss),
                        format_elapsed(epoch_start_time),
                        format_elapsed(start_time),
                        " ".join([str(x) for x in bp_summary]),
                        " ".join([str(x) for x in balanced_sequences]),
                        " ".join([str(x) for x in bracket_counts]),
                    )
                )

                



                if args.use_RNAfold_weights:
                    continue


                if batch_loss > 0:
                    batch_loss.backward()
                    network.trainer.step()

            else:
                batch_loss = 0
                gold = gold.transpose(0, 1)
                output = network(X)
                
                # Loss augmented margin-based decoding
                loss_margin = torch.ones(Y.size(), device=device) - Y
                output = output + loss_margin

                # Train-time decoding, argmax.
                values, predictions = torch.max(output[:,:,:-1], 2)
                
                _, _, _, Y_str = zip(*batch)
                predicted_dotbrackets = ["".join([network.bracket_vocab.value(int(col.data)) for col in row][:len(y)]) for row,y in zip(predictions.transpose(0,1).data, Y_str)]
                bp_summary, balanced_sequences, bracket_counts = get_bp_accuracies(predicted_dotbrackets, Y_str)

                for out_batch, gold_batch in zip(output, gold):
                    batch_loss += loss_function(out_batch, gold_batch)

                out(args.logfile,
                    "epoch {:,} "
                    "batch {:,}/{:,} "
                    "processed {:,} "
                    "batch-loss {:.4f} "
                    "epoch-elapsed {} "
                    "total-elapsed {} "
                    "   bp-acc {} "
                    "bal-seqs {} "
                    "brckt-cnts {} "
                    "".format(
                        epoch,
                        batch_index + 1, len(batched_train),
                        total_processed,
                        float(batch_loss),
                        format_elapsed(epoch_start_time),
                        format_elapsed(start_time),
                        " ".join([str(x) for x in bp_summary]),
                        " ".join([str(x) for x in balanced_sequences]),
                        " ".join([str(x) for x in bracket_counts]),
                    )
                )
                if batch_loss > 0:
                    batch_loss.backward()
                    network.trainer.step()


            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()

    check_dev()
    out(args.logfile, "# Training Finished")
    out(args.logfile, "# Total Time: {}".format(format_elapsed(start_time)))
    


def run_test(args):
    args.outfile = open(args.outfile, "w")
    #dotbrackets = read_bpRNA_directory(args.bpRNA_directory)
    
    testing_prediction_data = read_dotbrackets(args.input_predictions)
    testing_data = read_dotbrackets(args.input)

    IDs_to_delete_test = []
    for ID in testing_data:
        if ID not in testing_prediction_data:
            IDs_to_delete_test.append(ID)
    for ID in IDs_to_delete_test:
        del(testing_data[ID])

    def remove_pk(brackets):
        brackets = brackets.replace("[", ".").replace("]", ".")
        brackets = brackets.replace("{", ".").replace("}", ".")
        brackets = brackets.replace("<", ".").replace(">", ".")
        brackets = brackets.replace("A", ".").replace("a", ".")
        brackets = brackets.replace("B", ".").replace("b", ".")
        brackets = brackets.replace("C", ".").replace("c", ".")
        brackets = brackets.replace("D", ".").replace("d", ".")
        return brackets


    if args.max_length != 200:
        testing_data = [(k,v[0],remove_pk(v[1])) for k,v in testing_data.items() if len(v[0]) <= args.max_length]
    else:
        testing_data = [(k,v[0],remove_pk(v[1])) for k,v in testing_data.items() if len(v[0]) <= 200]
    testing_data = [(ID,seq,testing_prediction_data[ID][1],gold_dot) for ID,seq,gold_dot in testing_data]

    bpRNAID_to_RNAtype = {}
    RNA_type_results = defaultdict(list)
    if os.path.exists("analysis/RNA_type_list/bpRNAID_to_RNAtype.txt"):
        f = open("analysis/RNA_type_list/bpRNAID_to_RNAtype.txt", "r")
        for line in f.readlines()[1:]:
            line = line.strip().split("\t")
            bpRNA_ID, RNA_type = line
            bpRNAID_to_RNAtype[bpRNA_ID] = RNA_type.replace(" ", "_")

        RNA_type_counts = defaultdict(int)
        
        testing_with_RNAtype = []
        for ID, seq, annotation, bracket in testing_data:
            bpRNA_ID = ID.split("|")[-1].split(",")[-1]
            RNAtype = bpRNAID_to_RNAtype[bpRNA_ID]
            RNA_type_counts[RNAtype] += 1
            testing_with_RNAtype.append((ID + "|" + RNAtype, seq, annotation, bracket))
        #RNA_type_sorted_counts = sorted(RNA_type_counts.items(), key=lambda x:x[1], reverse=True)
        #pprint(RNA_type_sorted_counts)
        #print(sum(RNA_type_counts.values()))
        testing_data = testing_with_RNAtype
    

    #_, val, test = default_data_split(dotbrackets)
    label = "Test"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.GPU) if use_cuda and args.GPU is not None else "cpu")
    out(args.logfile, "Selecting device: {}".format(device))
    out(args.logfile, "Loading model... ", end="")
    network = Network.load(args.model_path_base, device)
    network.eval()
    out(args.logfile, "done.")


    if args.first is not None:
        testing_data = testing_data[:args.first]
    out(args.logfile, "# Number of sequences: {}".format(len(testing_data)))
    pred_data, tgt_data = [], []

    batched_testing = [testing_data[start_index:start_index+args.batch_size] \
                       for start_index in range(0, len(testing_data), args.batch_size)]


    start_time = time.time()
    #for i,example in enumerate(testing):
    for i,batch in enumerate(batched_testing):
        
        #if (i*args.batch_size) % 10 == 0 and i > 0:
        print("{}\r".format(i*args.batch_size), end="")
        
        IDs, X, _, gold = prepare_batch(batch, network.seq_vocab, network.bracket_vocab, device)
        _, _, _, Y = zip(*batch)

        if args.structured:
            #predicted_scores, predictions = network.matrix_predict(batch, X)
            #predicted_scores, predictions = network.parse_SR(batch, X)
            #predicted_scores, predictions = network.nussinov_parse_fast_threaded(batch, X, return_loss=False, use_RNAfold_weights=args.use_RNAfold_weights, thread_wait_delay=1)

            #print(predicted_scores)
            predicted_scores, predictions = network.DP_fix(batch, X, return_loss=False)

        else:
            output = network(X)
            values, predictions = torch.max(output[:,:,:-1], 2)

            predictions = ["".join([network.bracket_vocab.value(int(col.data)) for col in row][:len(y)]) for row,y in zip(predictions.transpose(0,1).data, Y)]


        #for ID,seq,annotation,gold_dot in batch:
        #    print(ID)

        

        """
        # Eliminate the mini-batch dimension
        output = output.view(output.size(0), -1)

        # Remove <UNK> predictions
        output = output[:,:-1]

        # Argmax
        values, predictions = torch.max(output, 1)
        predictions = [network.bracket_vocab.value(int(pred.data)) for pred in predictions.data]
        Y           = [network.bracket_vocab.value(int(y.data)) for y in gold.view(-1).data]
        """

        #Y = [[network.bracket_vocab.value(int(y.data)) for y in seq[:len(batch[batch_index][1])]] for batch_index,seq in enumerate(gold.data)]


        #pred_data.append(predictions)
        #tgt_data.append(Y)
        pred_data += predictions
        tgt_data += Y

        for example,pred in zip(batch,predictions):
            args.outfile.write("".join(example[0]) + "\n")
            args.outfile.write("".join(example[1]) + "\n")
            args.outfile.write("".join(pred) + "\n")


    sequences = [x[0] for x in testing_data]
    get_MCC(sequences, pred_data, tgt_data)

    confusion, sequence_accuracy_info = get_position_accuracies(pred_data, tgt_data)
    out(args.logfile, str_position_accuracies(confusion, sequence_accuracy_info), end="")

    bp_summary, balanced_sequences, bracket_counts = get_bp_accuracies(pred_data, tgt_data)
    out(args.logfile, 
        str_basepair_accuracies(bp_summary, balanced_sequences, bracket_counts),
        end="")

    averaged_sequence_f1score = get_sequence_f1scores(pred_data, tgt_data)
    out(args.logfile, "# Avg sequence f1score: {}".format(sum(averaged_sequence_f1score)/ len(averaged_sequence_f1score)))

    out(args.logfile)
    out(args.logfile, "# RESULTS: <SEQ ID> <RNA Type> <F1> <PREDICTED DOTBRACKET>")
    RNAtype_predictions = defaultdict(list)
    for i, ((ID, seq, annotation, bracket), pred) in enumerate(zip(testing_data, pred_data)):
        F1 = averaged_sequence_f1score[i]
        RNAtype = ID.split("|")[-1]
        out(args.logfile, "\t".join([ID, RNAtype, str(F1), pred]))
        RNAtype_predictions[RNAtype].append(F1)
    out(args.logfile, "# END OF RESULTS")
    out(args.logfile)

    #print(RNAtype_predictions)
    RNAtype_predictions = [(k,v, sum(v)/len(v)) for k,v in RNAtype_predictions.items() if len(v) >= 20]
    RNAtype_predictions = sorted(RNAtype_predictions, key=lambda x:len(x[1]), reverse=True)[:10]


    plt.figure(figsize=(10,10))
    RNAtypes, F1scores, avg_F1score = zip(*RNAtype_predictions)
    plt.boxplot(F1scores, labels=RNAtypes)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.25)
    plt.title("RNA Type Distribution\n{}".format(args.model_path_base.split("/")[-1]))
    plt.xlabel("RNA Type")
    plt.ylabel("F1 Score")
    plt.savefig(args.logfile.name + ".rnatype_dist.pdf")
    plt.close()


    """
    RNAtype_predictions = defaultdict(list)
    for (ID, seq, annotation, dotbracket),pred in zip(testing_data, pred_data):
        RNAtype = ID.split("|")[-1]
        item = (ID, seq, dotbracket, pred)
        RNAtype_predictions[RNAtype].append(item)

    RNAtype_results = {}
    for RNAtype,values in RNAtype_predictions.items():
        IDs, seqs, dotbrackets, preds = zip(*values)
        RNAtype_results[RNAtype] = get_sequence_f1scores(pred_data=preds, tgt_data=dotbrackets)
    
    RNAtype_f1scores = {k:(len(v), sum(v)/len(v)) for k,v in RNAtype_results.items()}
    out(args.logfile, pformat(RNAtype_f1scores))
    out(args.logfile)
    top_20 = sorted(RNAtype_f1scores.items(), reverse=True, key=lambda x: x[1][0])[:20]
    out(args.logfile, pformat(top_20))
    out(args.logfile)
    out(args.logfile, pformat(sorted(top_20, reverse=True, key=lambda x: x[1][1])))
    """


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--model-path-base", required=True)
    #subparser.add_argument("--bpRNA-directory", required=True)

    subparser.add_argument("--train", 
                           help="For Training. This should be a fasta-style file with 3 lines: ID, sequence, structure.", required=True)
    subparser.add_argument("--val", 
                           help="For Validation. This should be a fasta-style file with 3 lines: ID, sequence, structure.", required=True)
    subparser.add_argument("--train-predictions",
                           help="An annotation created by a previous model. This is another faster style file with 3 lines: ID, sequence, predicted structure.", required=True)
    subparser.add_argument("--val-predictions",
                           help="An annotation created by a previous model. This is another faster style file with 3 lines: ID, sequence, predicted structure.", required=True)
    subparser.add_argument("--dropout", type=float, default=0.1)
    subparser.add_argument("--silver-aug", default=False, action="store_true", 
                           help="If turned on, after the first epoch will start to augment the training data with sampled 'silver' data taken from mutating a sequence in the training data and running the model on it.")

    subparser.add_argument("--batch-size", type=int, default=16)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=float, default=4)
    subparser.add_argument("--GPU", type=int, default=None)
    subparser.add_argument("--logfile", default="train_log.txt")
    subparser.add_argument("--load-model", 
                           help="Load a pretrained-model file. This is used for Pretraining, Transfer Learning, or Curriculum Learning.",
    )
    subparser.add_argument("--structured", default=False, action="store_true",
                           help="Will use structured decoding in training a model.")
    subparser.add_argument("--max-length", type=int, default=200, 
                           help="Filters the length of the sequences in the training and validation down to a max length.")
    subparser.add_argument("--use-RNAfold-weights", default=False, action="store_true",
                           help="Will use RNAfold weights.")


    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=True)
    #subparser.add_argument("--bpRNA-directory", required=True)

    subparser.add_argument("--input", help="For Testing. This should be a fasta-style file with 3 lines: ID, sequence, structure.", required=True)
    subparser.add_argument("--input-predictions",
                           help="An annotation created by a previous model. This is another faster style file with 3 lines: ID, sequence, predicted structure.", required=True)

    subparser.add_argument("--batch-size", type=int, default=16)

    subparser.add_argument("--GPU", type=int, default=None)
    subparser.add_argument("--logfile", default=None) # default="test_eval_log.txt")

    subparser.add_argument("--outfile", default="out.txt")
    subparser.add_argument("--structured", default=False, action="store_true",
                           help="Will use structured decoding in testing a model.")
    subparser.add_argument("--max-length", type=int, default=200, 
                           help="Filters the length of the sequences in the testing data down to a max length.")
    subparser.add_argument("--use-RNAfold-weights", default=False, action="store_true",
                           help="Will use RNAfold weights.")
    subparser.add_argument("--first", default=None, type=int, 
                           help="Just parses the first N sequences, instead of the entire input file.")



    args = parser.parse_args()
    if args.logfile is None:
        args.logfile = args.model_path_base + ".log"
        args.logfile = open(args.logfile, "w")
    else:
        args.logfile = open(args.logfile, "a")

    
    out(args.logfile)
    out(args.logfile, datetime.datetime.now())
    out(args.logfile, "# python3 " + " ".join(sys.argv))

    args.callback(args)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()

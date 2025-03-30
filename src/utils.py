#!/usr/bin/python3

import sys, os, pickle
from itertools import zip_longest
import time
from Bio import SeqIO

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def evaluation(predictions,golds):
    positive_predictions = [0,0]
    negative_predictions = [0,0]
    for pred,gold in zip(predictions,golds):
        pred = int(pred)
        #gold = 1 if gold[0] == 0 else 0
        if gold == 1:
            if pred == 1:
                positive_predictions[0] += 1
            else:
                assert pred == 0
                negative_predictions[1] += 1
        else:
            assert gold == 0
            if pred == 0:
                negative_predictions[0] += 1
            else:
                assert pred == 1
                positive_predictions[1] += 1
    return positive_predictions, negative_predictions

def summary_scores(positive_predictions, negative_predictions):
    total = sum(positive_predictions) + sum(negative_predictions)
    accuracy = (positive_predictions[0] + negative_predictions[0]) / total
    precision, recall, fscore = 0,0,0
    if sum(positive_predictions) > 0:
        precision = positive_predictions[0] / sum(positive_predictions)
    if (positive_predictions[0] + negative_predictions[1]) > 0:
        recall = positive_predictions[0] / (positive_predictions[0] + negative_predictions[1])
    if precision + recall > 0:
        fscore = 2 * (precision * recall) / (precision + recall)
    return accuracy, (precision,recall,fscore)

def read_dotbrackets(dotbracket_filename):
    dotbrackets = {}
    #IDs = []
    def grouper(iterable, n):
        args = [iter(iterable)] * n
        return zip_longest(*args)


    with open(dotbracket_filename, "r") as infile:
        augmented_seq_count = 0
        
        i = 0
        lines = infile.readlines()
        while i < len(lines):
            line = lines[i]

            if "Unrecognized sequence" in line:
                #print(lines[i-1:i+2])
                #exit()
                del(lines[i])
                del(lines[i-1])

                continue
            i += 1

        for ID, seq, annotation in grouper(lines, 3):
            if ID is None or seq is None or annotation is None:
                break
            ID = ID.strip()
            if ID[0] == ">":
                ID = ID[1:]
            if ID == "augmented_seq":
                ID = "{}.{}".format(ID, augmented_seq_count)
                augmented_seq_count += 1

            if "|" in ID:
                ID = ID.split("|")[-1]
                if "," in ID:
                    ID = ID.split(",")[0]

            seq = seq.strip().upper().replace("T", "U")
            annotation = annotation[:len(seq)].strip()
            assert ID not in dotbrackets, "{}\n{}\n{}\n\n{}\n{}".format(ID, seq, annotation, dotbrackets[ID][0], dotbrackets[ID][1])
            dotbrackets[ID] = (seq.strip(), annotation.strip())
            #IDs.append(ID)
    return dotbrackets


def read_bpRNA_directory(directory_name):
    sys.stderr.write("Reading bpRNA directory {}\n".format(directory_name))
    if os.path.exists(directory_name + "/dotbrackets.pkl"):
        sys.stderr.write("Found pickle file. Reading... ")
        dotbrackets = pickle.load(open(directory_name + "/dotbrackets.pkl", "rb"))
        sys.stderr.write("done\n")
        return dotbrackets

    def read_bpRNA_file(filename):
        lines = open(filename).readlines()
        name = lines[0].split()[1].strip()
        sequence = lines[3].strip().upper().replace("T", "U")
        dotbracket = lines[4][:len(sequence)].strip()
        return name, (sequence, dotbracket)

    dotbrackets = {}
    for i,filename in enumerate(os.listdir(directory_name)):
        if ".dbn" not in filename:
            continue
        ID, (sequence, dotbracket) = read_bpRNA_file(directory_name + "/" + filename)
        if i % 1000 == 0:
            sys.stderr.write("{}\r".format(i))
        dotbrackets[ID] = (sequence.strip(), dotbracket.strip())

    sys.stderr.write("Finished Reading the bpRNA directory, creating a pickle file... ")
    pickle.dump(dotbrackets, open(directory_name + "/dotbrackets.pkl", "wb"))
    sys.stderr.write("done\n")
    return dotbrackets



def default_data_split(dotbrackets, fasta_file=None):
    
    """
    for i,ID in enumerate(dotbrackets):
        print(i, ID, dotbrackets[ID])
        if i == 10:
            break


    if fasta_file is not None:
        keys = {}
        i = 0
        for record in SeqIO.parse(fasta_file, "fasta"):
            assert record.id in dotbrackets, "{} {}".format(record.id, record.seq)
            keys[record.id] = str(record.seq)


            print(record.id, record.seq)
            print(dotbrackets[record.id])
            i += 1
            if i == 10:
                exit()
    """

    keys = list([ID for ID in dotbrackets.keys() if len(dotbrackets[ID][0]) <= 200])
    LEN = len(keys)
    div1, div2 = int(LEN*0.8), int(LEN*0.9)

    # Get rid of pseudoknots. (For now, until we move to a more advanced task)
    for ID in dotbrackets:
        brackets = dotbrackets[ID][1]
        brackets = brackets.replace("[", ".").replace("]", ".")
        brackets = brackets.replace("{", ".").replace("}", ".")
        brackets = brackets.replace("<", ".").replace(">", ".")
        brackets = brackets.replace("A", ".").replace("a", ".")
        brackets = brackets.replace("B", ".").replace("b", ".")
        brackets = brackets.replace("C", ".").replace("c", ".")

        assert len(set(brackets)) <= 3, brackets

        dotbrackets[ID] = (dotbrackets[ID][0], brackets)

    data = [(tuple(dotbrackets[ID][0]), tuple(dotbrackets[ID][1])) for ID in keys]

    """
    ID_counts = defaultdict(int)
    for ID in dotbrackets:
        #if not (len(dotbrackets[ID][0]) <= 200 and len(dotbrackets[ID][0]) >= 50):
        #if not (len(dotbrackets[ID][0]) <= 200):
        #    continue
        ID = ID.split("_")[1]
        ID_counts[ID] += 1
    for ID in sorted(ID_counts.keys()):
        print(ID, ID_counts[ID])
    exit()
    """


    train = data[:div1]
    val   = data[div1:div2]
    test  = data[div2:]
    return train, val, test



def calculate_class_imbalance(batched_dataset):
    """ Calculates the sequence and site imbalance for a given dataset.
    Inputs: batched_dataset
    Output: Sequence positives and negatives, Site positives and negatives
    """
    sequence_positives, sequence_negatives = 0, 0
    site_positives, site_negatives = 0, 0
    for batch in batched_dataset:
        for example in batch:
            sites = []
            for methylation, expression in zip(example.methylation, example.position_reads):
                if expression != 0 and methylation > 0:
                    sites.append(1)
                else:
                    sites.append(0)

            SUM = sum(sites)
            site_positives += SUM
            site_negatives += len(sites) - SUM
            if SUM > 0:
                sequence_positives += 1
            else:
                sequence_negatives += 1

    return (sequence_positives, sequence_negatives), (site_positives, site_negatives)

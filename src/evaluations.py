#!/usr/bin/python3
"""
Evaluation-related functions.
Such as site-accuracy, base-pair accuracy, etc.
"""
from math import sqrt

from const import START, STOP, UNK
from collections import defaultdict


def evaluate_site_accuracy(predicted_dotbrackets, gold_dotbrackets):
    """
    Takes two dotbracket sequences (predicted and gold)
    And returns site-accuracy confusion matrices.
    """
    confusion = defaultdict(lambda: defaultdict(int))
    confusion_left = defaultdict(lambda: defaultdict(int))
    confusion_right = defaultdict(lambda: defaultdict(int))
    for i, (pred, y) in enumerate(zip(predicted_dotbrackets, gold_dotbrackets)):
        if y == START or y == STOP:
            continue
        confusion[y][pred] += 1
        if i < len(predicted_dotbrackets)//2:
            confusion_left[y][pred] += 1
        else:
            confusion_right[y][pred] += 1
    return confusion, confusion_left, confusion_right


def add_confusion_matrices(matrix1, matrix2):
    """
    Takes two site-wide confusion matrices,
    and returns a new confusion matrix
    that is the sum of those two.
    """
    result = defaultdict(lambda: defaultdict(int))
    for key1 in matrix1:
        for key2 in matrix1[key1]:
            result[key1][key2] += matrix1[key1][key2]
    for key1 in matrix2:
        for key2 in matrix2[key1]:
            result[key1][key2] += matrix2[key1][key2]
    return result


def get_accuracy(matrix):
    """
    Takes a site-wide confusion matrix, 
    and calculates the overall accuracy.
    """
    CORRECT = 0
    SUM = 0
    for key in matrix:
        CORRECT += matrix[key].get(key,0)
        SUM += sum(matrix[key].values())
    return CORRECT, SUM, CORRECT / SUM


def get_basepairs(dotbrackets, CUTOFF=None):
    basepairs = []
    stack = []
    for j, symb in enumerate(dotbrackets):
        if symb == "(":
            stack.append(j)
        if symb == ")":
            if stack:
                i = stack.pop()
                if CUTOFF is None or j > int(CUTOFF*len(dotbrackets)):
                    basepairs.append((i,j))
    return set(basepairs)

def is_balanced(dotbrackets):
    stack = []
    for j, symb in enumerate(dotbrackets):
        if symb == "(":
            stack.append(j)
        if symb == ")":
            if stack:
                i = stack.pop()
            else:
                return False
    return not stack 
    

def get_basepair_counts(sequence, pred_dotbrackets, gold_dotbrackets, CUTOFF=None):
    """
    The CUTOFF denotes what percentage of the prefix of the sequence to ignore, for evaluation.
    This is used when we give the left half of the Y values, and mask out the right half (making CUTOFF=0.5).
    """
    basepairs_pred = get_basepairs(pred_dotbrackets, CUTOFF)
    basepairs_gold = get_basepairs(gold_dotbrackets, CUTOFF)

    TP, FP = 0, 0
    TN, FN = 0, 0
    for bp in basepairs_pred:
        if bp in basepairs_gold:
            TP += 1
        else:
            FP += 1
    for bp in basepairs_gold:
        if bp not in basepairs_pred:
            FN += 1

    n_a = len([x for x in sequence if x == "A"])
    n_c = len([x for x in sequence if x == "C"])
    n_g = len([x for x in sequence if x == "G"])
    n_u = len([x for x in sequence if x == "U"])
    TN = (n_a * n_u) + (n_u * n_g) + (n_g * n_c) - (TP + FP + FN)
    return TP, FP, FN, TN

def get_PRF(TP, FP, FN):
    precision, recall = 0, 0
    f1score = 0

    if TP + FP > 0:
        precision = TP / (TP + FP)
    if TP + FN > 0:
        recall = TP / (TP + FN)
    if precision+recall > 0:
        f1score = 2 * (precision*recall) / (precision+recall)
    #print("precision:", "{:.3f}".format(precision))
    #print("recall:", "{:.3f}".format(recall))
    #print("F1:", "{:.3f}".format(f1score))
    return precision, recall, f1score
    


def get_evaluations(sequence, pred_dotbrackets, gold_dotbrackets):

    TP, FP, FN, TN = get_basepair_counts(sequence, pred_dotbrackets, gold_dotbrackets)
    compatible = 0

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + (FP - compatible))
    
    PPV = TP / (TP + (FP - compatible)) if \
          (TP + (FP - compatible)) > 0 else 0
    balancedAcc = (sensitivity + specificity) / 2
    ACC = (TP + TN) / (TP + TN + (FP - compatible) + FN) if \
          (TP + TN + (FP - compatible) + FN) > 0 else 0
    MCC = ((TP * TN) - ((FP - compatible) * FN)) / sqrt((TP + FN) * (TP + (FP - compatible)) * (TN + (FP - compatible)) * (TN + FN)) if \
          ((TP + FN) * (TP + (FP - compatible)) * (TN + (FP - compatible)) * (TN + FN)) > 0 else 0
    
    """
    print("Acc", ACC)
    print("Sensitivity", sensitivity)
    print("Specificity", specificity)
    print("PPV", PPV)
    print("MCC", MCC)
    """
    return ACC, sensitivity, specificity, PPV, MCC
            

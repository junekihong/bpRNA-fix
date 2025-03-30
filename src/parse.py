#!/usr/bin/python
from collections import defaultdict 
import torch

class State:
    # State: score, (k,j), (action,backpointer), leftpointer

    __slots__ = ["insidescore", "prefixscore",
                 "kj", "action", "lr", "backpointer", "leftpointers",
                 "loss_augment",
                 "action_index",
    ]
    def __init__(self,
                 prefixscore=0,
                 insidescore=0,
                 kj=(0,0),
                 action=0,
                 lr=(0,0),
                 backpointer=None,
                 leftpointers=[],
    ):
        self.prefixscore = prefixscore
        self.insidescore = insidescore

        self.kj = kj
        self.action = action
        self.lr = lr
        self.backpointer = backpointer
        self.leftpointers = leftpointers
        self.action_index = 0
        self.loss_augment = 0


    # TODO: For when you want to add the full DP.
    # Will need to just change the beam to a dict, and merge left pointers. 
    def __hash__(self):
        return (self.kj, self.lr)

    def __eq__(self, other):
        return self.kj == other.kj and self.lr == other.lr

    def __lt__(self, other):
        return self.prefixscore < other.prefixscore

    def __repr__(self):
        #action_index = ""
        #if self.action_index is not None:
        action_index = "{:3d}: ".format(self.action_index)

        backpointer = "None             "
        if self.backpointer is not None:
            backpointer = "({:3d},{:3d} {:3d},{:3d})".format(
                self.backpointer[0].kj[0],
                self.backpointer[0].kj[1],
                self.backpointer[1].kj[0],
                self.backpointer[1].kj[1],
            )
        leftpointer_str = ""
        for leftpointer in self.leftpointers:
            leftpointer_str += "({:3d},{:3d}[{}{}])".format(
                leftpointer.kj[0],
                leftpointer.kj[1],
                leftpointer.lr[0],
                leftpointer.lr[1],
            )


        return "{}(Score:{:10.3f},{:10.3f} + {} (ij:{:3d},{:3d} [{}{}]) action:{} back:{} left:{})".format(
            action_index,
            float(self.prefixscore), float(self.insidescore),
            self.loss_augment,
            self.kj[0], self.kj[1],
            self.lr[0], self.lr[1],
            self.action, 
            backpointer, 
            leftpointer_str,
        )
    def __str__(self):
        return self.__repr__()


def brackets(dotbracket):
    stack = []
    results = set()
    for j,x in enumerate(dotbracket):
        if x == "(":
            stack.append(j)
            continue
        if x == ")":
            i = stack.pop()
            results.add((i,j))
            continue
    return results


def can_shift(j, n):
    return j < n-1
def can_reduce(k):
    return k > 0


def decode(batch_index, example, encoding, return_queue, return_loss=False):
    beamsize = 32
    ID,example,dotbracket = example

    base_state = State()
    beams = defaultdict(dict)
    beams[0] = [base_state]

    if return_loss:
        gold_decode = base_state
        max_violation = 0
        fallen_off_beam = False
        dotbrackets = brackets(dotbracket)
        shortest_spanning_bracket = {i:(None,None) for i in range(len(example))}
        for i,j in sorted(dotbrackets):
            for k in range(i,j+1):
                shortest_spanning_bracket[k] = (i,j)
    
    n = len(example)

    actions = 2*(n-1) - 1
    for action_index in range(actions):
        beam = beams[action_index]
        for state in beam:
            #print("state:", state)

            k,j = state.kj
            if can_shift(j, n):
                mem = encoding[j,j+1,0]
                shift_score = mem
                loss_augment = 0

                # Loss-Augmented Decoding
                if return_loss and can_reduce(k):
                    # If we are currently sitting on the right edge of a gold bracket
                    # We will miss the opportunity to form it if we shift.
                    left_j, right_j = shortest_spanning_bracket[j]

                    #if (left_j is not None) and \
                    #   (right_j == j and left_j <= k):
                    if (left_j is not None) and (right_j == j):
                        loss_augment += 1
                        shift_score = mem + loss_augment

                if ((j,j+1),(state.lr[1],0)) not in beams[action_index+1]:
                    shifted = State(
                        state.prefixscore + shift_score,
                        shift_score,
                        (j,j+1), 
                        action=0,
                        lr=(state.lr[1],0),
                        backpointer=None, 
                        leftpointers=[state],
                    )
                    shifted.action_index = state.action_index + 1
                    shifted.loss_augment = loss_augment
                    #beams[action_index+1].append(shifted)
                    beams[action_index+1][(j,j+1),(state.lr[1],0)] = shifted
                else:
                    shifted = beams[action_index+1][(j,j+1),(state.lr[1],0)]
                    assert shifted.loss_augment == loss_augment, "{} {}\n{}\n{}".format(shifted.loss_augment, loss_augment, shifted, state)
                    assert shifted.action == 0
                    shifted.leftpointers.append(state)





                if state.lr[1] == 0:
                    # Shift a base-paired unit-span.
                    mem = encoding[j,j+1,1]
                    shift_score = mem

                    # Loss-Augmented Decoding
                    if return_loss:
                        left_j,right_j = shortest_spanning_bracket[j]
                        left_j1,right_j1 = shortest_spanning_bracket[j+1]

                        # If (j,j+1) violates a gold bracket, we add 1 more
                        j_check = (left_j is not None) and \
                                  (right_j == j or \
                                   (left_j == j and right_j != j+1))
                        j1_check = (left_j1 is not None) and \
                                   (left_j1 == j+1 or \
                                    (left_j1 != j and right_j1 == j+1))
                        if j_check or j1_check:
                            loss_augment += 1
                            shift_score = mem + loss_augment


                    if ((j,j+1),(1,1)) not in beams[action_index+1]:
                        shifted = State(
                            state.prefixscore + shift_score,
                            shift_score,
                            (j,j+1), 
                            action=1,
                            lr=(1,1),
                            backpointer=None,
                            leftpointers=[state],
                        )
                        shifted.action_index = state.action_index + 1
                        shifted.loss_augment = loss_augment
                        #beams[action_index+1].append(shifted)
                        beams[action_index+1][(j,j+1),(1,1)] = shifted
                    else:
                        shifted = beams[action_index+1][(j,j+1),(1,1)]
                        assert shifted.loss_augment == loss_augment, "{} {}".format(shifted.loss_augment, loss_augment)
                        assert shifted.action == 1, "{} {}".format(shifted.action, action)
                        shifted.leftpointers.append(state)





            if can_reduce(k):

                for leftpointer in state.leftpointers:
                    i = leftpointer.kj[0]
                    assert leftpointer.kj[1] == k
                    mem = encoding[i,j,0]
                    reduce_score = state.insidescore + mem
                    
                    # Loss-Augmented Decoding
                    loss_augment = 0
                    if return_loss:
                        # If k is part of a gold span, we cannot form a bracket with it.
                        left, right = shortest_spanning_bracket[k]
                        if (left is not None and left == k) or \
                           (right is not None and right == k):
                            loss_augment += 1
                            reduce_score = state.insidescore + mem + loss_augment



                    action = 0
                    lr = (leftpointer.lr[0], state.lr[1])

                    if ((i,j),lr) not in beams[action_index+1]:
                        reduced = State(
                            leftpointer.prefixscore + reduce_score, 
                            leftpointer.insidescore + reduce_score,
                            (i,j), 
                            action=action,
                            lr=lr,
                            backpointer=(leftpointer, state), 
                            leftpointers=leftpointer.leftpointers, 
                        )
                        reduced.action_index = state.action_index + 1
                        reduced.loss_augment = loss_augment
                        #beams[action_index+1].append(reduced)
                        beams[action_index+1][(i,j),lr] = reduced
                    else:
                        reduced = beams[action_index+1][(i,j),lr]
                        newprefixscore = leftpointer.prefixscore + reduce_score
                        if newprefixscore > reduced.prefixscore:


                            """
                            print(newprefixscore, reduced.prefixscore)
                            print(reduced)

                            print(newprefixscore)
                            print(state)
                            print(leftpointer)


                            proposed_state = State(
                                leftpointer.prefixscore + reduce_score, 
                                leftpointer.insidescore + reduce_score,
                                (i,j), 
                                action=action,
                                lr=lr,
                                backpointer=(leftpointer, state), 
                                leftpointers=leftpointer.leftpointers, 
                            )
                            proposed_state.action_index = state.action_index + 1
                            proposed_state.loss_augment = loss_augment
                            print(proposed_state)
                            exit()
                            """


                            assert reduced.kj == (i,j)
                            assert reduced.lr == lr
                            assert reduced.action_index == state.action_index + 1

                            reduced.prefixscore = newprefixscore
                            reduced.insidescore = leftpointer.insidescore + reduce_score
                            reduced.backpointer = (leftpointer, state)
                            reduced.leftpointers = leftpointer.leftpointers
                            reduced.action = action
                            reduced.loss_augment = loss_augment



                    if lr == (0, 0):
                        # We can possibly reduce with a pairing here.
                        mem = encoding[i,j,1]
                        reduce_score = state.insidescore + mem

                        # Loss-Augmented Decoding
                        loss_augment = 0
                        if return_loss:
                            #if (i,j) violates a gold bracket, we add 1 more
                            left_i,right_i = shortest_spanning_bracket[i]
                            left_j,right_j = shortest_spanning_bracket[j]
                            i_check = (right_i is not None) and \
                                      (right_i == i or \
                                       (left_i == i and right_i != j))
                            j_check = (left_j is not None) and \
                                      (left_j == j or \
                                       (right_j == j and left_j != i))
                            if i_check or j_check:
                                loss_augment += 1
                                reduce_score = state.insidescore + mem + loss_augment

                        action = 1
                        lr = (1, 1)
                        
                        if ((i,j),lr) not in beams[action_index+1]:
                            reduced = State(
                                leftpointer.prefixscore + reduce_score, 
                                leftpointer.insidescore + reduce_score,
                                (i,j), 
                                action=action,
                                lr=lr,
                                backpointer=(leftpointer, state), 
                                leftpointers=leftpointer.leftpointers, 
                            )
                            reduced.action_index = state.action_index + 1
                            reduced.loss_augment = loss_augment
                            #beams[action_index+1].append(reduced)
                            beams[action_index+1][(i,j),lr] = reduced
                        else:
                            reduced = beams[action_index+1][(i,j),lr]
                            newprefixscore = leftpointer.prefixscore + reduce_score
                            if newprefixscore > reduced.prefixscore:

                                assert reduced.kj == (i,j)
                                #assert reduced.action == action, "{} {}".format(reduced.action, action)
                                assert reduced.lr == lr
                                assert reduced.action_index == state.action_index + 1
                                #assert reduced.loss_augment == loss_augment, "{} {}".format(reduced.loss_augment, loss_augment)

                                reduced.prefixscore = newprefixscore
                                reduced.insidescore = leftpointer.insidescore + reduce_score
                                reduced.backpointer = (leftpointer, state)
                                reduced.leftpointers = leftpointer.leftpointers
                                reduced.action = action
                                reduced.loss_augment = loss_augment
                                



        beams[action_index+1] = sorted(beams[action_index+1].values(), 
                                       reverse=True)[:beamsize]


        if return_loss:
            k,j = gold_decode.kj
            #print("gold: ",gold_decode)

            i = None
            if can_reduce(k):
                i = gold_decode.leftpointers[0].kj[0]

            left_boundary,right_boundary = shortest_spanning_bracket[j]
            if can_shift(j,n) and \
               ((left_boundary is None) or \
                ((k > 0 and i < left_boundary) or \
                 j < right_boundary) or \
                not can_reduce(k)
               ):
                # If there is no encompassing pairing here, and we can still shift.
                # Or if there is an encompassing pairing around us.
                # And the gold right boundary is further along.
                # Or if the gold left boundary is closer than k.
                if (j,j+1) in dotbrackets:
                    # It's possible if the span we shift is itself paired.
                    # In fact it is an edge case, that j is touching the span (j,j+1)
                    shift_score = encoding[j,j+1,1]
                    action = 1
                    lr = (1,1)
                else:
                    shift_score = encoding[j,j+1,0]
                    action = 0
                    lr = (gold_decode.lr[1], 0)
                    shifted = State(
                        gold_decode.prefixscore + shift_score, 
                        shift_score,
                        (j,j+1), 
                        action=action, 
                        lr=lr, 
                        backpointer=None,
                        leftpointers=[gold_decode],
                    )
                shifted.action_index = gold_decode.action_index + 1
                gold_decode = shifted
                #print("gold shifted: {}".format(shifted))
            else:
                assert right_boundary is None or left_boundary <= i or j == n-1, "lr:({},{}), ij:({},{}) {}".format(left_boundary, right_boundary, i, j, n-1)
                if i == left_boundary and j == right_boundary:
                    # We are on top of a gold span. Reduce with a bracket.
                    #reduce_score = gold_decode.insidescore + mem_paired[i,j,batch_index,1]
                    reduce_score = gold_decode.insidescore + encoding[i,j,1]
                    action = 1
                    lr=(1, 1)
                else:
                    # We need to extend our span back, by Reducing.
                    assert left_boundary is None or left_boundary < k or j == n-1
                    #reduce_score = gold_decode.insidescore + mem_paired[i,j,batch_index,0]
                    reduce_score = gold_decode.insidescore + encoding[i,j,0]
                    action = 0
                    lr = (gold_decode.leftpointers[0].lr[0], gold_decode.lr[1])

                reduced = State(
                    gold_decode.leftpointers[0].prefixscore + reduce_score,
                    gold_decode.leftpointers[0].insidescore + reduce_score,
                    (i,j), 
                    action=action,
                    lr=lr,
                    backpointer=(gold_decode.leftpointers[0], gold_decode), 
                    leftpointers=[gold_decode.leftpointers[0].leftpointers[0]],
                )
                reduced.action_index = gold_decode.action_index + 1
                gold_decode = reduced


            violation = beams[action_index+1][0].prefixscore - gold_decode.prefixscore
            if violation >= max_violation:
                max_violation = violation



            """
            # Track if the gold hypothesis has fallen off the beam.
            if gold_decode.prefixscore < beams[action_index+1][-1].prefixscore:
                fallen_off_beam = True

            # Early Update. If the gold hypothesis falls off of the beam, and the violation starts to decrease, we update immediately.
            # Originally a Max-Violation update. We changed to this for faster training.
            # Early Update / Max Violation hybrid.
            if fallen_off_beam and gold_decode.prefixscore > beams[action_index+1][0].prefixscore:
                #if gold_decode.score < beams[action_index+1][-1].score and violation < max_violation:
                losses[batch_index] = max_violation
                predicted[batch_index] = beams[action_index+1][0]
                #loss += max_violation
                #predicted.append(beams[action_index+1][0])
                #break
                return
    #if return_loss and not beams[actions]:
    #    continue
            """


    top_predicted = beams[actions][0]
    if return_loss:
        #return max_violation
        #loss += max_violation

        #print("{} Putting a top_predicted and gold_decode".format(batch_index))
        return_queue.put((batch_index, top_predicted, gold_decode))
    else:
        #print("{}: Putting a top_predicted".format(batch_index))
        return_queue.put((batch_index, top_predicted))


def bt(state, memory, gold_state=None):
    n = state.kj[1] + 1
    result = ["."]*n
    scores = []
    bt_stack = [state]

    states_visited = []

    return_loss = gold_state is not None
    if return_loss:
        gold_scores = []
        gold_bt_stack = [gold_state]


    #print("return loss:", return_loss)
    while bt_stack:
        state = bt_stack.pop()
        #print("bt", state)
        states_visited.insert(0, state)

        k,j = state.kj
        if not (k == 0 and j == 0):

            if state.action == 1:
                result[k] = "("
                result[j] = ")"
                
            mem = memory[k,j,state.action]
            scores.insert(0, mem + state.loss_augment)

            if return_loss:
                gold_state = gold_bt_stack.pop()
                k,j = gold_state.kj
                gold_mem = memory[k,j,gold_state.action]
                gold_scores.insert(0, gold_mem)

        if state.backpointer is not None:
            leftstate,parentstate = state.backpointer
            bt_stack.append(leftstate)
            bt_stack.append(parentstate)
        #state = state.backpointer

        if return_loss:
            if gold_state.backpointer is not None:
                leftstate,parentstate = gold_state.backpointer
                gold_bt_stack.append(leftstate)
                gold_bt_stack.append(parentstate)
            #gold_state = gold_state.backpointer


    """
    DEBUG_SCORES = ["{:.3f}".format(float(sum(scores[:i+1]))) for i in range(len(scores))]
    for state, score in zip(reversed(states_visited), reversed(DEBUG_SCORES)):        
        if abs(state.prefixscore - float(score)) < 0.01:
            print(" ", score, state)
        else:
            print("*", score, state)
    print(float(sum(scores)))
    exit()
    """

    result = "".join(result)
    if return_loss:
        # Max-Violation is calculated and returned.4
        max_violation = 0
        traj, gold_traj = 0, 0
        for score, gold_score in zip(scores, gold_scores):
            traj += score
            gold_traj += gold_score
            violation = traj - gold_traj
            if violation > max_violation:
                max_violation = violation
        return result, max_violation
    return result, sum(scores)



def nussinov(mem_paired, dotbrackets, return_loss=False):
    nussinov = defaultdict(lambda: (0, None))
    n = mem_paired.size(0)
    if return_loss:
        gold_parse = defaultdict(lambda: (0, None))

    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            unpaired = nussinov[i,j-1][0] + mem_paired[i,j,0]
            subscores = [unpaired]
            for k in range(i, j):
                left  = nussinov[i,   k-1][0]
                right = nussinov[k+1, j-1][0]
                paired = left + right + mem_paired[k,j,1]
                subscores.append(paired)
            subscores = torch.stack(subscores, 0)
            values, index = torch.max(subscores, 0)

            backpointers = None
            index = int(index)
            if index == 0:
                backpointer = -1
            else:
                # The value of k, the split point
                backpointer = i + index - 1
            nussinov[i,j] = (values, backpointer)


            if return_loss:
                subscore = None
                if (i,j) not in dotbrackets:
                    unpaired = gold_parse[i,j-1][0] + mem_paired[i,j,0]
                    subscore = (unpaired, -1)
                else:
                    for k in range(i, j):
                        if (k,j) in dotbrackets:
                            left  = gold_parse[i,   k-1][0]
                            right = gold_parse[k+1, j-1][0]
                            paired = left + right + mem_paired[k,j,1]
                            subscore = (paired, k)
                            break
                value, backpointer = subscore
                gold_parse[i,j] = (value, backpointer)



    score = 0
    if return_loss:
        gold_score = 0

    def bt():
        result = [None] * n
        i, j = 0, n-1
        _, bp = nussinov[i, j]
        queue = [((i,j), bp)]
        while queue:
            (i,j), k = queue.pop()
            if k is None:
                if result[j] is None:
                    result[j] = "."
            elif k == -1:
                result[j] = "."
                _, left = nussinov[i, j-1]
                queue.append(((i, j-1), left))
            else:
                result[k] = "("
                result[j] = ")"
                _, left = nussinov[i, k-1]
                _, right = nussinov[k+1, j-1]
                queue.append(((i, k-1), left))
                queue.append(((k+1,j-1), right))
        return "".join(result)


    score, backptr = nussinov[0, n-1]
    decode = bt()
    if return_loss:
        gold_score, gold_backptr = gold_parse[0, n-1]
        return decode, score, gold_score
    return decode, score

from nltk.translate.meteor_score import meteor_score
import sys

sys.path.insert(0, './submodules/')
from pycocoevalcap.meteor.meteor import Meteor

refs = [" an apple on a tree"]
hypo = "an apple on a tree"
single_word_hypo = "apple"

refs2 = [["an", "apple", "on", "a", "tree"]]
hypo2 = ["an", "apple", "on", "a", "tree"]

def test_meteor_print(ref, hypo):
    print("Refs: ", ref)
    print("Hypo: ", hypo)
    print(meteor_score(ref, hypo))


#test_meteor_print(refs2, hypo)


def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Meteor(),"METEOR")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

pyco_refs = {1: refs}
pyco_hypos = {1: [hypo]}

test_meteor_print(refs, single_word_hypo)
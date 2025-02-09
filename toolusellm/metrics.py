import re
import string
from collections import Counter
from typing import List

def normalize_answer(s):
    """Normalize QA answer
    codes adapted from: https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py#L14

    Args:
        s (str): a string to normalize
    
    Returns:
        str: a normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    # codes adapted from: https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py#L53
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def exact_match_score(prediction, ground_truth):
    # codes adapted from: https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py#L49C1-L50C74
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def rough_match_score(prediction, ground_truth):
    normalized_pred = normalize_answer(prediction)
    normalized_gt = normalize_answer(ground_truth)
    result = re.search(normalized_gt, normalized_pred)
    if result is not None:
        return 1
    return 0

def f1_score(prediction, ground_truth):
    """Compute F1 score.
    codes adapted from: https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py#L36

    Args:
        pred (str): model prediction
        gt_list (list): a list of ground truth labels

    Returns:
        float: f1 score
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    """ Check whether the model answer contains one of the ground truth answers
    codes adapted from: https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py#L30

    Args:
        prediction (str): model prediction
        ground_truths (List[str]): a list of ground truth labels

    Returns:
        float: 1 for match and 0 for not match
    """
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def exact_match(pred, gt_list):
    """Compute exact match score, i.e. whether the prediction are exactly one of the ground truth answers

    Args:
        pred (str): model prediction
        gt_list (list): a list of ground truth labels
        
    Returns:
        int: 1 for match and 0 for not match
    """
    em_for_this_question = metric_max_over_ground_truths(exact_match_score, pred, gt_list)
    return em_for_this_question

def exact_match_score_math(prediction, ground_truth):
    # not normalize
    return prediction == ground_truth

def exact_match_math(pred, gt_list):
    """Compute exact match score, i.e. whether the prediction are exactly one of the ground truth answers

    Args:
        pred (str): model prediction
        gt_list (list): a list of ground truth labels
        
    Returns:
        int: 1 for match and 0 for not match
    """
    em_for_this_question = metric_max_over_ground_truths(exact_match_score_math, pred, gt_list)
    return em_for_this_question

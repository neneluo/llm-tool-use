import json
import argparse
from toolusellm.metrics import *
from toolusellm.utils import *
import re

def evaluate_qa(result):
    with open(result, "r") as result_file:
        lines = [json.loads(line) for line in result_file]
    sample_amount = len(lines)
    exact_match_score, accuracy = 0.0, 0.0
    empty_amount = 0.0
    for line in lines:
        model_answer = str(line["model_answer"])
        if model_answer == "" or re.fullmatch(r'\(.*\)', model_answer) is not None:
            empty_amount += 1
        gt_answer_list = list(line["gt_answer"])

        exact_match_score += exact_match(model_answer, gt_answer_list)
        accuracy += best_subspan_em(model_answer, gt_answer_list)

    exact_match_score = (exact_match_score / sample_amount) * 100
    accuracy = (accuracy / sample_amount) * 100
    return exact_match_score, accuracy, sample_amount, empty_amount

def evaluate_math(result):
    with open(result, "r") as result_file:
        lines = [json.loads(line) for line in result_file]
    sample_amount = len(lines)
    exact_match_score, accuracy = 0.0, 0.0
    empty_amount = 0.0
    for line in lines:
        model_answer = str(line["model_answer"])
        if model_answer == "" or re.fullmatch(r'\(.*\)', model_answer) is not None:
            empty_amount += 1
        gt_answer_list = list(line["gt_answer"])
        exact_match_score += exact_match(model_answer, gt_answer_list)

        # extract the last number as model answer if matched and apply regex
        model_answer, gt_answer_list = regex_math_answer(model_answer, gt_answer_list)
        regex_score = exact_match(model_answer, gt_answer_list)
        accuracy += regex_score

    exact_match_score = (exact_match_score / sample_amount) * 100
    accuracy = (accuracy / sample_amount) * 100
    return exact_match_score, accuracy, sample_amount, empty_amount

def main():
    # parse arguments
    parser = argparse.ArgumentParser(prog='compute_score.py', \
                        description='This is a python program to compute QA accuracy.')
    parser.add_argument('--json', type=str, required=True, help="result file name")
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset in ["triviaqa", "popqa", "nq_open"]:
        exact_match_score, accuracy, sample_amount, empty_amount = evaluate_qa(args.json)
    elif args.dataset == "gsm8k":
        exact_match_score, accuracy, sample_amount, empty_amount = evaluate_math(args.json)
    print("\nAnalysing result file %s, sample amount %d, empty answers amount %d:\nExact Match %.1f, Accuracy %.1f" % (args.json, sample_amount, empty_amount, exact_match_score, accuracy))

if __name__ == "__main__":
    main()

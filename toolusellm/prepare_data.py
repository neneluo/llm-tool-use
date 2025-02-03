import sys
import jsonlines
import argparse
from transformers import AutoTokenizer
from toolusellm.metrics import *
from toolusellm.utils import *


def load_tokenizer(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

def compute_match(model_answer, gt_answer_list, dataset):
    if dataset in ["triviaqa", "popqa", "nq_open"]:
        match_or_not = best_subspan_em(model_answer, gt_answer_list)
    elif dataset == "gsm8k":
        # extract the last number as model answer if matched and apply regex
        model_answer, gt_answer_list = regex_math_answer(model_answer, gt_answer_list)
        match_or_not = exact_match_math(model_answer, gt_answer_list)
    return match_or_not

def create_sft_dataset(input_json, output_json, metric, dataset):
    with jsonlines.open(input_json, "r") as reader:
        lines = [line for line in reader]
    print("Read input json file: %s, %d lines" % (input_json, len(lines)))

    sample_amount = len(lines)
    match = 0.0
    with jsonlines.open(output_json, 'w') as writer:
        for line in lines:
            response = str(line["model_answer"])
            model_answer  = extract_answer(response)
            gt_answer_list = list(line["gt_answer"])

            if metric == "exact_match":
                match_or_not = exact_match(model_answer, gt_answer_list)
            elif metric == "acc":
                match_or_not = compute_match(model_answer, gt_answer_list, dataset)
            else:
                print("Metric {} not supported.".format(metric))
                sys.exit(1)

            match += match_or_not
            if match_or_not == 1:
                outline = {"messages": line["conversations"]}
                writer.write(outline)
        print("Output json file has been written to: %s, %d lines" % (output_json, match))
    return (match / sample_amount) * 100

def create_sft_dataset_w_tools(input_json, output_json, metric, dataset):
    with jsonlines.open(input_json, "r") as reader:
        lines = [line for line in reader]
    print("Read input json file: %s, %d lines" % (input_json, len(lines)))

    sample_amount = len(lines)
    match = 0.0
    with jsonlines.open(output_json, 'w') as writer:
        for line in lines:
            conversations = line["conversations"]
            # not using tools
            if len(conversations) < 4:
                continue

            response = str(line["model_answer"])
            model_answer  = extract_answer(response)
            gt_answer_list = list(line["gt_answer"])

            if metric == "exact_match":
                match_or_not = exact_match(model_answer, gt_answer_list)
            elif metric == "acc":
                match_or_not = compute_match(model_answer, gt_answer_list, dataset)
            else:
                print("Metric {} not supported.".format(metric))
                sys.exit(1)

            match += match_or_not
            if match_or_not == 1:
                outline = {"messages": line["conversations"]}
                writer.write(outline)
        print("Output json file has been written to: %s, %d lines" % (output_json, match))
    return (match / sample_amount) * 100

def create_pft_dataset_w_tools(input_json_chosen, input_json_rejected, output_json, metric, dataset):
    # compose a triplet of data: prompt, chosen, rejected
    # currently strategy: 
    #   read in two json files, one of no tool use traces, one for single-step tool use traces. 
    # system: single-step tool use prompt
    # prompt: question
    # chosen: single-step tool use, and correctly use tool to get exact match result
    # rejected: no tool use, and do not get exact match result

    with jsonlines.open(input_json_chosen, "r") as reader:
        chosen_lines = [line for line in reader]
    print("Read input_json_chosen file: %s, %d lines" % (input_json_chosen, len(chosen_lines)))
    with jsonlines.open(input_json_rejected, "r") as reader:
        rejected_lines = [line for line in reader]
    print("Read input_json_rejected file: %s, %d lines" % (input_json_rejected, len(rejected_lines)))

    assert len(chosen_lines) == len(rejected_lines)
    count = 0.0

    with jsonlines.open(output_json, "w") as writer:
        for chosen_line, rejected_line in zip(chosen_lines, rejected_lines):
            # filtering criterion: 
            # (1) no tool use EM = 0
            # (2) single-step tool EM = 1
            # (3) tool action and response not empty
            chosen_response = extract_answer(str(chosen_line["model_answer"]))
            rejected_response = extract_answer(str(rejected_line["model_answer"]))
            gt_answer_list = list(chosen_line["gt_answer"])
            if metric == "exact_match":
                chosen_response_match = exact_match(chosen_response, gt_answer_list)
                rejected_response_match = exact_match(rejected_response, gt_answer_list)
            elif metric == "acc":
                chosen_response_match = compute_match(chosen_response, gt_answer_list, dataset)
                rejected_response_match = compute_match(rejected_response, gt_answer_list, dataset)
            else:
                print("Metric {} not supported.".format(metric))
                sys.exit(1)
            if chosen_response_match == 1 and rejected_response_match == 0:
                ### choose tool-using traces only
                
                # not using tools
                if len(chosen_line["conversations"]) < 4:
                    continue

                tool_response = chosen_line["conversations"][3]
                assert tool_response["role"] == "user" and "Response from tool" in tool_response["content"]
                # use tools wrong
                if "Error" in tool_response["content"]:
                    continue

                """
                ### choose tool-using traces & w/o using tools
                """

                chosen_conversation = chosen_line["conversations"]
                rejected_conversation = rejected_line["conversations"]
                tokenizer = load_tokenizer()
                outline = {
                    "prompt": tokenizer.apply_chat_template([chosen_conversation[1]], tokenize=False),
                    "chosen": tokenizer.apply_chat_template([chosen_conversation[2]], tokenize=False),
                    "rejected": tokenizer.apply_chat_template([rejected_conversation[2]], tokenize=False)
                }
                writer.write(outline) 
                writer._fp.flush()
                count += 1
    print("Output json file has been written to: %s, %d lines" % (output_json, count))


def create_pft_dataset_w_tools_conversation(input_json_chosen, input_json_rejected, output_json, metric, dataset):
    # compose a triplet of data: prompt, chosen, rejected
    # currently strategy: 
    #   read in two json files, one of no tool use traces, one for single-step tool use traces. 
    # system: single-step tool use prompt
    # prompt: question
    # chosen: single-step tool use, and correctly use tool to get exact match result
    # rejected: no tool use, and do not get exact match result

    with jsonlines.open(input_json_chosen, "r") as reader:
        chosen_lines = [line for line in reader]
    print("Read input_json_chosen file: %s, %d lines" % (input_json_chosen, len(chosen_lines)))
    with jsonlines.open(input_json_rejected, "r") as reader:
        rejected_lines = [line for line in reader]
    print("Read input_json_rejected file: %s, %d lines" % (input_json_rejected, len(rejected_lines)))

    assert len(chosen_lines) == len(rejected_lines)
    count = 0.0

    with jsonlines.open(output_json, "w") as writer:
        for chosen_line, rejected_line in zip(chosen_lines, rejected_lines):
            # filtering criterion: 
            # (1) no tool use EM = 0
            # (2) single-step tool EM = 1
            # (3) tool action and response not empty
            chosen_response = extract_answer(str(chosen_line["model_answer"]))
            rejected_response = extract_answer(str(rejected_line["model_answer"]))
            gt_answer_list = list(chosen_line["gt_answer"])
            if metric == "exact_match":
                chosen_response_match = exact_match(chosen_response, gt_answer_list)
                rejected_response_match = exact_match(rejected_response, gt_answer_list)
            elif metric == "acc":
                chosen_response_match = compute_match(chosen_response, gt_answer_list, dataset)
                rejected_response_match = compute_match(rejected_response, gt_answer_list, dataset)
            else:
                print("Metric {} not supported.".format(metric))
                sys.exit(1)
            if chosen_response_match == 1 and rejected_response_match == 0:
                ### choose tool-using traces only
                
                # not using tools
                if len(chosen_line["conversations"]) < 4:
                    continue

                tool_response = chosen_line["conversations"][3]
                assert tool_response["role"] == "user" and "Response from tool" in tool_response["content"]
                # use tools wrong
                if "Error" in tool_response["content"]:
                    continue
                
                chosen_conversation = chosen_line["conversations"]
                rejected_conversation = rejected_line["conversations"]
                tokenizer = load_tokenizer()
                outline = {
                    "prompt": tokenizer.apply_chat_template([chosen_conversation[1]], tokenize=False),
                    "chosen": tokenizer.apply_chat_template(chosen_conversation[2:], tokenize=False),
                    "rejected": tokenizer.apply_chat_template(rejected_conversation[2:], tokenize=False)
                }
                writer.write(outline) 
                writer._fp.flush()
                count += 1
    print("Output json file has been written to: %s, %d lines" % (output_json, count))


def create_pft_dataset(input_json_chosen, input_json_rejected, output_json, metric, dataset):
    # compose a triplet of data: prompt, chosen, rejected
    # currently strategy: 
    #   read in two json files, one of no tool use traces, one for single-step tool use traces. 
    # system: single-step tool use prompt
    # prompt: question
    # chosen: single-step tool use, and correctly use tool to get exact match result
    # rejected: wrong answer generated by LLM

    with jsonlines.open(input_json_chosen, "r") as reader:
        chosen_lines = [line for line in reader]
    print("Read input_json_chosen file: %s, %d lines" % (input_json_chosen, len(chosen_lines)))
    with jsonlines.open(input_json_rejected, "r") as reader:
        rejected_lines = [line for line in reader]
    print("Read input_json_rejected file: %s, %d lines" % (input_json_rejected, len(rejected_lines)))

    assert len(chosen_lines) == len(rejected_lines)
    count = 0.0

    with jsonlines.open(output_json, "w") as writer:
        for chosen_line, rejected_line in zip(chosen_lines, rejected_lines):
            # filtering criterion: 
            # (1) rejected EM = 0
            # (2) accepted EM = 1
            chosen_response = str(chosen_line["model_answer"])
            rejected_response = str(rejected_line["model_answer"])
            gt_answer_list = list(chosen_line["gt_answer"])
            if metric == "exact_match":
                chosen_response_match = exact_match(chosen_response, gt_answer_list)
                rejected_response_match = exact_match(rejected_response, gt_answer_list)
            elif metric == "acc":
                chosen_response_match = compute_match(chosen_response, gt_answer_list, dataset)
                rejected_response_match = compute_match(rejected_response, gt_answer_list, dataset)
            else:
                print("Metric {} not supported.".format(metric))
                sys.exit(1)
            if chosen_response_match == 1 and rejected_response_match == 0:
                ### choose tool-using traces & w/o using tools
                chosen_conversation = chosen_line["conversations"]
                rejected_conversation = rejected_line["conversations"]
                tokenizer = load_tokenizer()
                outline = {
                    "prompt": tokenizer.apply_chat_template([chosen_conversation[1]], tokenize=False),
                    "chosen": tokenizer.apply_chat_template([chosen_conversation[2]], tokenize=False),
                    "rejected": tokenizer.apply_chat_template([rejected_conversation[2]], tokenize=False)
                }
                writer.write(outline) 
                writer._fp.flush()
                count += 1
    print("Output json file has been written to: %s, %d lines" % (output_json, count))


def main():
    # parse arguments
    parser = argparse.ArgumentParser(prog='prepare_data.py', \
                        description='This is a python program to prepare fine-tuning dataset.')
    parser.add_argument('--input_json', type=str, required=False, help="for sft data: input json")
    parser.add_argument('--input_json_chosen', type=str, required=False, help="for pft data: input json to select chosen")
    parser.add_argument('--input_json_rejected', type=str, required=False, help="for pft data: input json to select rejected")
    parser.add_argument('--output_json', type=str, required=True, help="prepared json data")
    parser.add_argument('--data_type', type=str, required=True, help="sft/pft")
    parser.add_argument('--metric', type=str, required=True, help="exact_match/acc")
    parser.add_argument('--dataset', type=str, required=True, help="triviaqa/popqa/nq_open/gsm8k")
    parser.add_argument('--tools', type=bool, required=False, help="keep tool-using data only")
    args = parser.parse_args()

    if args.data_type == "sft":
        if args.tools:
            acc = create_sft_dataset_w_tools(args.input_json, args.output_json, args.metric, args.dataset)
        else:
            acc = create_sft_dataset(args.input_json, args.output_json, args.metric, args.dataset)
        print("Produced %s dataset based on %s metric, accuracy are %.1f" % (args.data_type, args.metric, acc))
    elif args.data_type == "pft":
        if args.tools:
            create_pft_dataset_w_tools(
                args.input_json_chosen,
                args.input_json_rejected,
                args.output_json,
                args.metric,
                args.dataset
            )
        else:
            create_pft_dataset(
                args.input_json_chosen,
                args.input_json_rejected,
                args.output_json,
                args.metric,
                args.dataset
            )
    else:
        print("Error: %s data type not supported." % args.data_type)
        sys.exit(1)

if __name__ == "__main__":
    main()

import json
import jsonlines
import argparse
from toolusellm.metrics import *
from toolusellm.utils import *


def compute_rates(result):
    with open(result, "r") as result_file:
        lines = [json.loads(line) for line in result_file]

    sample_amount = len(lines)
    tool_usage_amount = 0.0
    tool_pass_amount = 0.0
    answerable_amount = 0.0 

    tool_amount = {}
    
    # note: only compatible with single-step tool usage results
    for line in lines:
        conversations = line["conversations"]
        # extract first assistant response
        response = conversations[2]
        assert response["role"] == "assistant"
        tools = extract_tool_usage(response["content"])

        if len(conversations) > 3:
            # if conversations are more than three turn, then tool-using must be involved
            assert len(tools) > 0

            # check tool usage and tool pass
            tool_usage_amount += len(tools)
            user_response = conversations[3]
            assert user_response["role"] == "user"

            # ignore the first item (empty string)
            tool_responses = user_response["content"].split("Response from tool")[1:]
            assert len(tools) == len(tool_responses)

            for tool_response in tool_responses:
                tool = tool_response.split("[")[0].strip()
                tool_amount[tool] = tool_amount.get(tool, 0) + 1
                if tool_response is not None and "Error" not in tool_response \
                and tool_response != "None" and tool_response != "":
                    tool_pass_amount += 1
                    # check whether the tool response contains answer when using tools
                    tool_answer = tool_response.split("] are:")[1].strip()
                    answerable_amount += best_subspan_em(tool_answer, list(line["gt_answer"]))

            assert tool_usage_amount == sum(tool_amount.values())
    
    # rates
    invoke_rate = (tool_usage_amount / sample_amount) * 100
    pass_rate = (tool_pass_amount / tool_usage_amount) * 100
    answerable_rate = (answerable_amount / tool_usage_amount) * 100
    return sample_amount, invoke_rate, pass_rate, answerable_rate, \
        tool_amount, tool_usage_amount


def main():
    # parse arguments
    parser = argparse.ArgumentParser(prog='compute_rate.py', \
                        description='This is a python program to compute tool invoke rate and pass rate.')
    parser.add_argument('--json', type=str, required=True, help="result file name")
    args = parser.parse_args()

    file_name = args.json
    sample_amount, invoke_rate, pass_rate, answerable_rate, \
        tool_amount, \
        tool_usage_amount = compute_rates(file_name)
    print("\nAnalysing result file %s, sample_amount %d." % (file_name, sample_amount))
    print("Tool Invoke Rate %.1f, Tool Pass Rate %.1f, Answerable Rate %.1f" % (invoke_rate, pass_rate, answerable_rate))
    print("Tool usage summary: ", tool_amount, tool_usage_amount)

if __name__ == "__main__":
    main()


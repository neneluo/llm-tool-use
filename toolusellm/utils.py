import re

def extract_answer(response):
    try:
        answer = response.split("\nAnswer:")[1].strip()
    except Exception as e:
        # print(e)
        answer = response
    return answer

def extract_tool_usage(response):
    """extract tool usage information from llm response, matching all patterns in the format \
    of "characters[any characters except[]]"

    Args:
        response (str): response from llm

    Returns:
        list: a list of extracted tool usage patterns
    """
    try:
        action = response.split("\nAction:")[1].split("\nRationale:")[0]
    except Exception as e:
        # print(e)
        action = ""
    tools = re.findall(r'\w+\[[^\[\]]+\]', action)
    return tools

def regex_math_answer(model_answer, gt_answer_list):
    # extract the last number as model answer if matched
    # regex pattern adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml#L41
    try:
        pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        matched = re.findall(pattern, model_answer)
        if matched:
            model_answer = matched[-1][0] if matched[-1][0] else matched[-1][1]
    except Exception as e:
        # print(e)
        pass
    return model_answer, gt_answer_list

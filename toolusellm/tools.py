from functools import cache
from pyserini.search.lucene import LuceneSearcher
import json
import numexpr

@cache
def Calculator(query: str) -> str:
    """NumExpr based calculator, which is a fast numerical expression evaluator for NumPy.
    codes adapted from: https://github.com/ernie-research/Tool-Augmented-Reward-Model/blob/main/src/tools/calculator.py#L1
    
    Example usage:
    answer = Calculator("2+3")
    print(answer) # 5.0

    Args:
        query (str): a math formula, supports +, -, * amd /

    Returns:
        str: calculated answer
    """
    try:
        tool_response = str(numexpr.evaluate(query))
    except Exception as e:
        print(e)
        tool_response = "Error: failed to calculate {}.".format(query)
    return tool_response


@cache
def WikipediaRetriever(query: str) -> str:
    """ Wikipedia Retriever
    codes adapted from https://huggingface.co/docs/trl/en/learning_tools

    Args:
        query (str): the sentence to search

    Returns:
        str: the most relevant document
    """
    try:
        searcher = LuceneSearcher.from_prebuilt_index("wikipedia-kilt-doc")
        hits = searcher.search(query, k=1)
        tool_response = json.loads(hits[0].lucene_document.get('raw'))['contents']
    except Exception as e:
        print(e)
        tool_response = "Error: didn't find a corresponding Wikipedia document for {}.".format(query)
    return tool_response


@cache
def MachineTranslator(query: str, mt_tokenizer, mt_model) -> str:
    """Machine Translator to translate input query into English using model NLLB-600M
    codes adapted from: https://github.com/lucidrains/toolformer-pytorch/blob/main/toolformer_pytorch/tools.py#L83

    Args:
        query (str): the sentence to translate
        mt_tokenizer (hf tokenizer): hugging face tokenizer
        mt_model (hf model): hugging face model

    Returns:
        str: corresponding English translation of the provided query
    """
    try:
        input_ids = mt_tokenizer(query, return_tensors='pt')
        outputs = mt_model.generate(
            **input_ids,
            forced_bos_token_id=mt_tokenizer.convert_tokens_to_ids("eng_Latn")
        )
        tool_response = mt_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        print(e)
        tool_response = "Error: failed to translate {}.".format(query)
    return tool_response

from datasets import load_dataset
import jsonlines
import random
import json

def main():
    # download opensource datasets: triviaqa, gsm8k, popqa
    triviaqa = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
    for split, split_dataset in triviaqa.items():
        split_dataset.to_json("datasets/triviaqa-{}.jsonl".format(split))
    
    gsm8k = load_dataset("openai/gsm8k", "main")
    for split, split_dataset in gsm8k.items():
        split_dataset.to_json("datasets/gsm8k-{}.jsonl".format(split))

    popqa = load_dataset("akariasai/PopQA")
    for split, split_dataset in popqa.items():
        split_dataset.to_json("datasets/popqa-{}.jsonl".format(split))

    # construct our own training, validation and test split
    triviaqa_training_set_size = 10000
    triviaqa_test_set_size = 1000

    gsm_test_set_size = 650

    # for trivia_qa
    with jsonlines.open("datasets/triviaqa-train.jsonl", "r") as reader:
        lines = [line for line in reader]
    random.shuffle(lines)
    questions = []
    gt_answers = []
    count = 0
    for line in lines:
        questions.append(line["question"])
        gt_answers.append(line["answer"]["aliases"])
        count += 1
        if count == triviaqa_training_set_size:
            break

    with jsonlines.open("data/triviaqa-subset-train.jsonl", "w") as writer:
        for question, gt_answer in zip(questions, gt_answers):
            outline = {"question": question, "gt_answer": gt_answer} 
            writer.write(outline)
    
    with jsonlines.open("datasets/triviaqa-validation.jsonl", "r") as reader:
        lines = [line for line in reader]
    random.shuffle(lines)
    questions = []
    gt_answers = []
    count = 0
    for line in lines:
        questions.append(line["question"])
        gt_answers.append(line["answer"]["aliases"])
        count += 1
        if count == (triviaqa_test_set_size * 2):
            break

    with jsonlines.open("data/triviaqa-subset-valid.jsonl", "w") as writer:
        for question, gt_answer in zip(questions[:triviaqa_test_set_size], gt_answers[:triviaqa_test_set_size]):
            outline = {"question": question, "gt_answer": gt_answer} 
            writer.write(outline)
    with jsonlines.open("data/triviaqa-subset-test.jsonl", "w") as writer:
        for question, gt_answer in zip(questions[triviaqa_test_set_size:], gt_answers[triviaqa_test_set_size:]):
            outline = {"question": question, "gt_answer": gt_answer} 
            writer.write(outline)
    

    # for gsm8k
    with jsonlines.open("datasets/gsm8k-train.jsonl", "r") as reader:
        lines = [line for line in reader]
    random.shuffle(lines)
    questions = []
    gt_answers = []
    for line in lines:
        questions.append(line["question"])
        gt_answers.append(line["answer"])

    with jsonlines.open("data/gsm8k-subset-train.jsonl", "w") as writer:
        for question, gt_answer in zip(questions, gt_answers):
            process, result = gt_answer.split("\n#### ")
            outline = {"question": question, "gt_answer": [result], "process": process} 
            writer.write(outline)
    
    with jsonlines.open("datasets/gsm8k-test.jsonl", "r") as reader:
        lines = [line for line in reader]
    random.shuffle(lines)
    questions = []
    gt_answers = []
    for line in lines:
        questions.append(line["question"])
        gt_answers.append(line["answer"])

    with jsonlines.open("data/gsm8k-subset-valid.jsonl", "w") as writer:
        for question, gt_answer in zip(questions[:gsm_test_set_size], gt_answers[:gsm_test_set_size]):
            process, result = gt_answer.split("\n#### ")
            outline = {"question": question, "gt_answer": [result], "process": process} 
            writer.write(outline)
    with jsonlines.open("data/gsm8k-subset-test.jsonl", "w") as writer:
        for question, gt_answer in zip(questions[gsm_test_set_size:], gt_answers[gsm_test_set_size:]):
            process, result = gt_answer.split("\n#### ")
            outline = {"question": question, "gt_answer": [result], "process": process} 
            writer.write(outline)


    with jsonlines.open("datasets/popqa-test.jsonl", "r") as reader:
        lines = [line for line in reader]
    random.shuffle(lines)
    questions = []
    gt_answers = []
    count = 0
    for line in lines:
        questions.append(line["question"])
        gt_answers.append(json.loads(line["possible_answers"]))
        count += 1
        if count == (triviaqa_test_set_size * 2):
            break

    with jsonlines.open("data/popqa-subset-valid.jsonl", "w") as writer:
        for question, gt_answer in zip(questions[:triviaqa_test_set_size], gt_answers[:triviaqa_test_set_size]):
            outline = {"question": question, "gt_answer": gt_answer} 
            writer.write(outline)
    with jsonlines.open("data/popqa-subset-test.jsonl", "w") as writer:
        for question, gt_answer in zip(questions[triviaqa_test_set_size:], gt_answers[triviaqa_test_set_size:]):
            outline = {"question": question, "gt_answer": gt_answer} 
            writer.write(outline)


    # for nq_open
    nq_open = load_dataset("google-research-datasets/nq_open")
    for split, split_dataset in nq_open.items():
        split_dataset.to_json("datasets/nq_open-{}.jsonl".format(split))
    
    training_set_size = 10000
    dev_set_size = 1000

    with jsonlines.open("datasets/nq_open-train.jsonl", "r") as reader:
        lines = [line for line in reader]
    random.shuffle(lines)
    questions = []
    gt_answers = []
    count = 0
    for line in lines:
        questions.append(line["question"])
        gt_answers.append(line["answer"])
        count += 1
        if count == training_set_size:
            break
    with jsonlines.open("data/nq_open-subset-train.jsonl", "w") as writer:
        for question, gt_answer in zip(questions, gt_answers):
            outline = {"question": question, "gt_answer": gt_answer} 
            writer.write(outline)
    
    with jsonlines.open("datasets/nq_open-validation.jsonl", "r") as reader:
        lines = [line for line in reader]
    random.shuffle(lines)
    questions = []
    gt_answers = []
    count = 0
    for line in lines:
        questions.append(line["question"])
        gt_answers.append(line["answer"])
        count += 1
        if count == (dev_set_size * 2):
            break

    with jsonlines.open("data/nq_open-subset-valid.jsonl", "w") as writer:
        for question, gt_answer in zip(questions[:dev_set_size], gt_answers[:dev_set_size]):
            outline = {"question": question, "gt_answer": gt_answer} 
            writer.write(outline)
    with jsonlines.open("data/nq_open-subset-test.jsonl", "w") as writer:
        for question, gt_answer in zip(questions[dev_set_size:], gt_answers[dev_set_size:]):
            outline = {"question": question, "gt_answer": gt_answer} 
            writer.write(outline)

if __name__ == "__main__":
    main()

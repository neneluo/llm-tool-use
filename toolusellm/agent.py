import os, sys
import argparse
import jsonlines

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, set_seed
import torch
from trl import DataCollatorForCompletionOnlyLM, setup_chat_format
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import PartialState
from toolusellm.prompts import *
from toolusellm.tools import *
from toolusellm.utils import *

device_string = PartialState().process_index

class ToolUseAgent:
    def __init__(self, args) -> None:
        # llm decoding related parameters
        self.test_llm_max_new_tokens = args.test_llm_max_new_tokens
        self.test_llm_do_sample = args.test_llm_do_sample
        self.test_llm_temperature = args.test_llm_temperature
        self.test_llm_top_p = args.test_llm_top_p
    
    def prompt_engineering(self, prompt_type):
        """choose the appropriate prompt for llm

        Args:
            prompt_type (str): specify the type of prompt

        Returns:
            str: the prompt sent to llm
        """
        prompt = ""
        if prompt_type == "empty":
            pass
        elif prompt_type == "no_tool_zeroshot":
            prompt = prompt_no_tool_zeroshot
        elif prompt_type == "no_tool_zeroshot_cot":
            prompt = prompt_no_tool_zeroshot_cot
        elif prompt_type == "three_tools_single_step_zeroshot_cot":
            prompt = prompt_three_tools_single_step_zeroshot_cot
        elif prompt_type == "three_tools_single_step_zeroshot_cot_worational":
            prompt = prompt_three_tools_single_step_zeroshot_cot_worational
        elif prompt_type == "three_tools_multi_step_zeroshot_cot":
            prompt = prompt_three_tools_multi_step_zeroshot_cot
        else:
            print("Error: prompt_type %s not supported." % prompt_type)
            sys.exit(1)
        return prompt
    
    def set_mt_model(self, mt_tokenizer, mt_model):
        self.mt_tokenizer = mt_tokenizer
        self.mt_model = mt_model

    def invoke_tool(self, func):
        """Invoke an external tool and get its response

        Args:
            func (str): usage of the tool, in the format of toolname[query]

        Returns:
            str: response from the tool
        """
        tool = func.split("[")[0]
        query = func.split("[")[1].split("]")[0] 
        if tool == "Calculator":
            response = Calculator(query)
        elif tool == "WikipediaSearch":
            response = WikipediaRetriever(query)
        elif tool == "MachineTranslator":
            response = MachineTranslator(query, self.mt_tokenizer, self.mt_model)
        else:
            response = "{} does not exist.".format(tool)
        return response    
    
    def call_llm(self, tokenizer, model, conversations):
        """Inference llm on a given list of conversations
        codes adpated from: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

        Args:
            tokenizer (hf tokenizer): hugging face model tokenizer
            model (hf model): hugging face model
            conversations (list): a list of conversation

        Returns:
            str: response from llm
        """
        input_ids = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        # set pad token to suppress warning: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/40
        outputs = model.generate(
            input_ids,
            max_new_tokens=self.test_llm_max_new_tokens,
            eos_token_id=terminators,
            do_sample=self.test_llm_do_sample,
            temperature=self.test_llm_temperature,
            top_p=self.test_llm_top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        output = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(output, skip_special_tokens=True)
        return response
    
    def inference_single_step_llm(self, prompt_type, question, tokenizer, model):
        """Inference llm on a given dataset

        Args:
            prompt_type (str): specify the type of prompt
            question (str): the question sent to llm
            tokenizer (hf tokenizer): hugging face model tokenizer
            model (hf model): hugging face model

        Returns:
            str: response from llm
        """
        prompt = self.prompt_engineering(prompt_type)
        conversations = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        llm_response = self.call_llm(tokenizer, model, conversations)
        conversations.append({"role": "assistant", "content": llm_response})
        return conversations, llm_response
    
    def inference_single_step_llm_with_tools(self, prompt_type, question, tokenizer, model):
        prompt = self.prompt_engineering(prompt_type)
        conversations = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        llm_response = self.call_llm(tokenizer, model, conversations)
        conversations.append({"role": "assistant", "content": llm_response})
        tools = extract_tool_usage(llm_response)
        if len(tools) > 0:
            tool_responses = []
            for func in tools:
                tool_response = "Response from tool {} are: {}".format(func, self.invoke_tool(func))
                tool_responses.append(tool_response)
            formatted_tool_responses = "\n".join(tool_responses)
            
            conversations.append({"role": "user", "content": formatted_tool_responses})
            llm_response = self.call_llm(tokenizer, model, conversations)
            conversations.append({"role": "assistant", "content": llm_response})
        return conversations, llm_response

    def inference_multi_steps_llm_with_tools(self, prompt_type, question, tokenizer, model):
        prompt = self.prompt_engineering(prompt_type)
        conversations = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        llm_response = self.call_llm(tokenizer, model, conversations)
        conversations.append({"role": "assistant", "content": llm_response})
        tools = extract_tool_usage(llm_response)
        while len(tools) > 0:
            tool_responses = []
            for func in tools:
                tool_response = "Response from tool {} are: {}".format(func, self.invoke_tool(func))
                tool_responses.append(tool_response)
            formatted_tool_responses = "\n".join(tool_responses)
            
            conversations.append({"role": "user", "content": formatted_tool_responses})
            llm_response = self.call_llm(tokenizer, model, conversations)
            conversations.append({"role": "assistant", "content": llm_response})
            tools = extract_tool_usage(llm_response)
        return conversations, llm_response
    
    def test(self, args):
        # load dataset
        with jsonlines.open(args.test_data, "r") as reader:
            to_test_data = [line for line in reader]

        # load valid data
        questions = []
        gt_answers = []
        for item in to_test_data:
            questions.append(item["question"])
            gt_answers.append(item["gt_answer"])

        # load tokenizer and pre-trained model
        tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_id
        )
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        }                                                
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_id,
            device_map="auto",
            **model_kwargs
        )
            
        # load tool model (for machine translator)
        mt_model_id = "facebook/nllb-200-distilled-600M"
        mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_id)
        mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_id)
        self.set_mt_model(mt_tokenizer, mt_model)

        if os.path.exists(args.test_output_json_file):
            print("Warning: output json file %s exists, overwriting ..." % args.test_output_json_file)
        with jsonlines.open(args.test_output_json_file, mode='w') as writer:
            for question, gt_answer in zip(questions, gt_answers):
                if args.tool_usage_type == "no_tool":
                    conversations, response = self.inference_single_step_llm(args.prompt_type, question, tokenizer, model)
                elif args.tool_usage_type == "single_step":
                    conversations, response = self.inference_single_step_llm_with_tools(args.prompt_type, question, tokenizer, model)
                elif args.tool_usage_type == "multi_step":
                    conversations, response = self.inference_multi_steps_llm_with_tools(args.prompt_type, question, tokenizer, model)
                else:
                    print("Error: tool usage type %s not supported." % args.tool_usage_type)
                    sys.exit(1)
                model_answer = extract_answer(response=response)
                outline = {"conversations": conversations, "question": question, "model_answer": model_answer, "gt_answer": gt_answer}
                writer.write(outline)
                writer._fp.flush()
    
    def train(self, args):
        # supervised fine-tuning
        #   input: conversations
        if args.ft_type == "sft_lora":
            # load dataset 
            train_data = load_dataset("json", data_files=args.ft_train_data, split="train")
            train_data.shuffle()

            # load pre-trained model
            tokenizer = AutoTokenizer.from_pretrained(
                args.llm_model_id
            )
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2"
            }          
            # set use_cache as Fasle for ft
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_id,
                device_map={'':device_string},
                use_cache=False,
                **model_kwargs
            )

            tokenizer.pad_token = tokenizer.eos_token

            # set to conversational format
            # model, tokenizer = setup_chat_format(model, tokenizer)

            # define parameter efficient fine-tuning config
            peft_config = LoraConfig(
                r=16,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            # model = get_peft_model(model, peft_config)
            # model.print_trainable_parameters()

            # train on completions only
            data_collator=DataCollatorForCompletionOnlyLM(
                instruction_template="<|start_header_id|>user<|end_header_id|>",
                response_template="<|start_header_id|>assistant<|end_header_id|>",
                tokenizer=tokenizer
            )

            # define training arguments
            training_args = SFTConfig(
                output_dir=args.ft_model_output_dir,
                num_train_epochs=args.ft_epoch,
                optim=args.ft_optimizer,
                bf16=True,
                per_device_train_batch_size=args.ft_train_batch_size,
                max_seq_length=args.ft_max_seq_len,
                save_strategy="epoch",
                gradient_accumulation_steps=args.ft_gradient_accumulation_steps,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                ddp_find_unused_parameters=False,
                report_to="wandb"
            )

            # define sft trainer
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                tokenizer=tokenizer,
                packing=False,
                data_collator=data_collator,
                peft_config=peft_config
            )

            # train for epochs defined above
            trainer.train()

            # save model checkpoint
            trainer.save_model()
        # preference fine-tuning
        #   input: prompt + positive sample + negative sample
        elif args.ft_type == "pft_dpo":
            # load dataset
            train_data = load_dataset("json", data_files=args.ft_train_data, split="train")
            train_data.shuffle()

            # load pre-trained model
            tokenizer = AutoTokenizer.from_pretrained(
                args.llm_model_id
            )
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2"
            }
            # set use_cache as Fasle for ft
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_id,
                device_map={'':device_string},
                use_cache=False,
                **model_kwargs
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            peft_config = LoraConfig(
                r=16,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            # model = get_peft_model(model, peft_config)
            # model.print_trainable_parameters()

            # define training arguments
            training_args = DPOConfig(
                output_dir=args.ft_model_output_dir,
                num_train_epochs=args.ft_epoch,
                optim=args.ft_optimizer,
                bf16=True,
                per_device_train_batch_size=args.ft_train_batch_size,
                max_length=args.ft_max_seq_len,
                beta=args.ft_dpo_beta,
                save_strategy="epoch",
                gradient_accumulation_steps=args.ft_gradient_accumulation_steps,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                ddp_find_unused_parameters=False,
                report_to="wandb",
                remove_unused_columns=False,
                max_prompt_length=128
            )

            # define dpo trainer
            trainer = DPOTrainer(
                model,
                ref_model=None,
                args=training_args,
                train_dataset=train_data,
                tokenizer=tokenizer,
                peft_config=peft_config
            )

            # train for epochs defined above
            trainer.train()

            # save model checkpoint
            trainer.save_model()
        else:
            print("Error: model tuning mode %s not supported." % args.ft_type)
            sys.exit(1)

def main():
    set_seed(19)

    # parse arguments
    parser = argparse.ArgumentParser(prog='agent.py', \
                        description='This is a python program to load a \
                        large lannguage model as an agent and interact with the environment.')
    parser.add_argument('--mode', type=str, help="Program mode: training/test")
    parser.add_argument('--prompt_type', type=str, \
                        help="Prompt type: no_tool_zeroshot/no_tool_zeroshot_cot")
    parser.add_argument('--tool_usage_type', type=str, \
                        help="Tool Usage type: no_tool/single_step/multi_step")
    parser.add_argument('--llm_model_id', type=str)
    # inference related args
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_output_json_file', type=str)
    parser.add_argument('--test_llm_do_sample', type=bool)
    parser.add_argument('--test_llm_max_new_tokens', type=int)
    parser.add_argument('--test_llm_temperature', type=float)
    parser.add_argument('--test_llm_top_p', type=float)
    # fine-tuning related args
    parser.add_argument('--ft_type', type=str, help="Fine-tuning type: sft_lora/pft_dpo")
    parser.add_argument('--ft_train_data', type=str, help="Location of training data in the json format")
    parser.add_argument('--ft_model_output_dir', type=str, help="Directory to save fine-tuned model")
    parser.add_argument('--ft_max_seq_len', type=int)
    parser.add_argument('--ft_optimizer', type=str)
    parser.add_argument('--ft_epoch', type=int)
    parser.add_argument('--ft_train_batch_size', type=int)
    parser.add_argument('--ft_gradient_accumulation_steps', type=int)
    # dpo related args
    parser.add_argument('--ft_dpo_beta', type=float)
    args = parser.parse_args()
    print(args)

    # init agent
    agent = ToolUseAgent(args)

    # model inference
    if args.mode == "test":
        agent.test(args)
    # model training
    elif args.mode == "training":
        agent.train(args)
    else:
        print("Error: program mode %s not supported." % args.mode)
        sys.exit(1)

if __name__ == "__main__":
    main()

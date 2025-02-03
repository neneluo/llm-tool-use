#!/bin/bash

export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

model="Meta-Llama-3-70B-Instruct"
subset="valid"
prompt_type="three_tools_single_step_zeroshot_cot"
tool_usage_type="single_step"
results_dir="results/triviaqa_gsm8k"
mkdir -p ${results_dir}

for dataset in triviaqa gsm8k popqa nq_open; do
    json_prefix="${dataset}-subset-${subset}"
    result_json="${results_dir}/${model}.${prompt_type}.${tool_usage_type}.${json_prefix}.jsonl"

    python toolusellm/agent.py --mode "test" \
        --prompt_type ${prompt_type} \
        --tool_usage_type ${tool_usage_type} \
        --llm_model_id "meta-llama/${model}" \
        --test_data "data/${json_prefix}.jsonl" \
        --test_output_json_file ${result_json} \
        --test_llm_do_sample True \
        --test_llm_max_new_tokens 512 \
        --test_llm_temperature 0.6 \
        --test_llm_top_p 0.9
    
    sh score.sh ${result_json} ${dataset}
done


prompt_type="no_tool_zeroshot_cot"
tool_usage_type="no_tool"
results_dir="results/triviaqa_gsm8k"

for dataset in triviaqa gsm8k popqa nq_open; do
    json_prefix="${dataset}-subset-${subset}"
    result_json="${results_dir}/${model}.${prompt_type}.${tool_usage_type}.${json_prefix}.jsonl"

    python toolusellm/agent.py --mode "test" \
        --prompt_type ${prompt_type} \
        --tool_usage_type ${tool_usage_type} \
        --llm_model_id "meta-llama/${model}" \
        --test_data "data/${json_prefix}.jsonl" \
        --test_output_json_file ${result_json} \
        --test_llm_do_sample True \
        --test_llm_max_new_tokens 512 \
        --test_llm_temperature 0.6 \
        --test_llm_top_p 0.9
    
    sh score.sh ${result_json} ${dataset}
done

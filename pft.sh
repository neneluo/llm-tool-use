#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

export PYTHONPATH=$(pwd)

subset="valid"
prompt_type="three_tools_single_step_zeroshot_cot"
tool_usage_type="single_step"
results_dir="results/triviaqa_gsm8k"
mkdir -p ${results_dir}

ft_type="pft_dpo"
training_data_prefix="pft_single.triviaqa_gsm8k.tools.three_tools_single_step_zeroshot_cot.single_step.acc"
training_data="training_data/${training_data_prefix}.jsonl"
 
stage="all"
sft_base_model="${results_dir}/Meta-Llama-3-8B-Instruct.sft.triviaqa_gsm8k.tools_wo_tools.three_tools_single_step_zeroshot_cot.single_step.acc"
model_output_dir="${results_dir}/Meta-Llama-3-8B-Instruct.${training_data_prefix}.beta0.5"

if [ "$stage" = "verify" ]; then
    echo $training_data
    echo $model_output_dir
fi

if [ "$stage" = "all" ]; then
    echo "Model training..."
    accelerate launch toolusellm/agent.py --mode "training" \
        --prompt_type ${prompt_type} \
        --tool_usage_type ${tool_usage_type} \
        --llm_model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
        --ft_type ${ft_type} \
        --ft_train_data ${training_data} \
        --ft_model_output_dir ${model_output_dir} \
        --ft_max_seq_len 8192 \
        --ft_optimizer "adamw_torch" \
        --ft_epoch 3 \
        --ft_train_batch_size 1 \
        --ft_dpo_beta 0.5 \
        --ft_gradient_accumulation_steps 16
fi

if [ "$stage" = "all" ]; then
    echo "Model testing..."
    for dataset in triviaqa gsm8k popqa nq_open; do
        json_prefix="${dataset}-subset-${subset}"
        result_json="${model_output_dir}.${prompt_type}-${tool_usage_type}-${json_prefix}.jsonl"
        python toolusellm/agent.py --mode "test" \
            --prompt_type ${prompt_type} \
            --tool_usage_type ${tool_usage_type} \
            --llm_model_id ${model_output_dir} \
            --test_data "data/${json_prefix}.jsonl" \
            --test_output_json_file ${result_json} \
            --test_llm_do_sample True \
            --test_llm_max_new_tokens 512 \
            --test_llm_temperature 0.6 \
            --test_llm_top_p 0.9

        sh score.sh ${result_json} ${dataset}
    done
fi

OUTPUTPATH=output
mkdir -p $OUTPUTPATH
dt=`date '+%Y%m%d_%H%M%S'`
task_name=processed_Interview
GPU=0
args=$@

CUDA_VISIBLE_DEVICES=$GPU python main.py \
--key api_key.txt \
--openai_url https://api.deepseek.com/v1 \
--data_path input/Musk2.txt \
--model deepseek-chat \
--clean_prompt_path structllm/prompt_/clean_prompt.json \
--extract_q_prompt_path structllm/prompt_/extract_q_prompt.json \
--extract_a_prompt_path structllm/prompt_/extract_a_prompt.json \
--summary_prompt_path structllm/prompt_/summary_prompt.json \
--character_path input/character.txt \
--output_result_path output/output_result.txt \
--debug 1 \
--batch_size 10 \
--encoder_model SentenceBERT \
--output_path $OUTPUTPATH/${data_path}/llm-${model}__embedding-${encoder_model}__bs-${batch_size}__${dt} $args \

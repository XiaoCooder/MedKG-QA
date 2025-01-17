OUTPUTPATH=output
mkdir -p $OUTPUTPATH
dt=`date '+%Y%m%d_%H%M%S'`
task_name=processed_Interview
GPU=0

model=deepseek-chat
input=Musk2
encoder_model=SentenceBERT
batch_size=10
args=$@

CUDA_VISIBLE_DEVICES=$GPU python main.py \
--key api_key.txt \
--openai_url https://api.deepseek.com/v1 \
--data_path input/${input}.txt \
--model ${model} \
--clean_prompt_path structllm/prompt_/clean_prompt.json \
--extract_q_prompt_path structllm/prompt_/extract_q_prompt.json \
--extract_a_prompt_path structllm/prompt_/extract_a_prompt.json \
--summary_prompt_path structllm/prompt_/summary_prompt.json \
--character_path input/character.txt \
--output_result_path output/output_result.txt \
--debug 1 \
--batch_size ${batch_size} \
--encoder_model ${encoder_model} \
--output_path $OUTPUTPATH/${input}/llm-${model}__embedding-${encoder_model}__bs-${batch_size}__${dt} $args \

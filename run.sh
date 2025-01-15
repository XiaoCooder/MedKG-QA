OUTPUTPATH=output
mkdir -p $OUTPUTPATH
GPU=0

CUDA_VISIBLE_DEVICES=$GPU python main.py \
--key api_key.txt \
--openai_url https://api.deepseek.com/v1 \
--data_path input/Musk2.txt \
--model deepseek-chat \
--clean_prompt_path structllm/prompt_/clean_prompt.json \
--extract_q_prompt_path structllm/prompt_/clean_prompt.json \
--extract_a_prompt_path structllm/prompt_/clean_prompt.json \
--summary_prompt_path structllm/prompt_/clean_prompt.json \
--character_path input/character.txt \
--output_result_path output/output_result.txt \
--debug 1 \
--batch_size 10 \
--encoder_model SentenceBERT \
--output_path $OUTPUTPATH/result.txt
#cat $OUTPUTPATH/output_detail* > $OUTPUTPATH/all_detail.txt
#cat $OUTPUTPATH/output_result* > $OUTPUTPATH/all_result.txt

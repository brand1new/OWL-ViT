#!/bin/bash


query_info=""
image_dir=""
digital_twins_dir=""
result_output_dir=""
scoring_llm_provider="doubao"
scoring_llm_model="lite"
ranking_llm_provider="deepseek"
ranking_llm_model="v3"
object_retrieval_llm_provider="deepseek"
object_retrieval_llm_model="v3"
log_file=""
gpu_id=0

export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Start llm retrieval"
echo "    query_info: "$query_info
echo "    image_dir: "$image_dir
echo "    digital_twins_dir: "$digital_twins_dir
echo "    result_output_dir: "$result_output_dir
echo "    scoring_llm_provider: "$scoring_llm_provider
echo "    scoring_llm_model: "$scoring_llm_model
echo "    ranking_llm_provider: "$ranking_llm_provider
echo "    ranking_llm_model: "$ranking_llm_model
echo "    object_retrieval_llm_provider: "$object_retrieval_llm_provider
echo "    object_retrieval_llm_model: "$object_retrieval_llm_model
echo "    gpu_id:"$gpu_id
echo "    log_file:"$log_file

python ./retrieval_pipeline/llm_retrieval.py\
 --query_info $query_info\
 --image_dir $image_dir\
 --dt_dir $digital_twins_dir\
 --output_dir $result_output_dir\
 --scoring_llm_provider $scoring_llm_provider\
 --scoring_llm_model $scoring_llm_model\
 --ranking_llm_provider $ranking_llm_provider\
 --ranking_llm_model $ranking_llm_model\
 --object_retrieval_llm_provider $object_retrieval_llm_provider\
 --object_retrieval_llm_model $object_retrieval_llm_model\
 > $log_file

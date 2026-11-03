LLM="Qwen/Qwen3.5-27B"
CUDA_VISIBLE_DEVICES=1,2,3,4 vllm serve ${LLM} \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 96000 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 16 \
    --language-model-only \
    --stream-interval 4 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

    #--max-num-batched-tokens 4096 \



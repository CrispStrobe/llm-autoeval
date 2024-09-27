#!/bin/bash

start=$(date +%s)

# (Existing GPU detection code remains unchanged)

# Install dependencies
apt update
apt install -y screen vim git-lfs
screen

# Install common libraries
pip install -q requests accelerate sentencepiece pytablewriter einops protobuf huggingface_hub==0.21.4
pip install -U transformers

# (Existing Hugging Face login code remains unchanged)

# Set dtype based on environment variable or default to float32
DTYPE=${MODEL_DTYPE:-float32}
echo "Using dtype: $DTYPE"

# Run evaluation
if [ "$BENCHMARK" == "nous" ]; then
    git clone -b add-agieval https://github.com/dmahan93/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .

    benchmark="agieval"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/4] =================="
    python main.py \
        --model hf-causal \
        --model_args pretrained=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=$DTYPE \
        --tasks agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math \
        --device cuda:$cuda_devices \
        --batch_size auto \
        --output_path ./${benchmark}.json

    # (Repeat the change for the other benchmarks in the "nous" section)

elif [ "$BENCHMARK" == "openllm" ]; then
    git clone https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .
    pip install accelerate

    benchmark="arc"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/6] =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL_ID},dtype=$DTYPE,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    # (Repeat the change for the other benchmarks in the "openllm" section)

elif [ "$BENCHMARK" == "lighteval" ]; then
    # (The "lighteval" section remains unchanged as it doesn't specify dtype)

elif [ "$BENCHMARK" == "eq-bench" ]; then
    git clone https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .
    pip install accelerate

    benchmark="eq-bench"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/1] =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL_ID},dtype=$DTYPE,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks eq_bench \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path ./evals/${benchmark}.json

    # (The rest of this section remains unchanged)

else
    echo "Error: Invalid BENCHMARK value. Please set BENCHMARK to 'nous', 'openllm', 'lighteval', or 'eq-bench'."
fi

# (The rest of the script remains unchanged)

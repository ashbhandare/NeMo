
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
unset SLURM_NTASKS


# Mounts
DATA_DIR="/lustre/fsw/bignlp/big_nlp/gpt3/prepare_dataset/the_pile/"
DIR="/NeMo/ptuneing_fp8_scripts/5b"


# Necessary Exports
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/NeMo/:\${PYTHONPATH}

NSYS="nsys profile -s cpu -t nvtx,cuda -o ${DIR}/profile_h100te --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"

${NSYS} torchrun --nproc_per_node=8 /NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning.py \
        --config-path=${DIR} \
        --config-name=5b_squad.yaml 


if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <num_nodes>"
    exit 1
fi

model_name=$1
nnodes=$2

tp=$((8*$nnodes))
rank_id=$RANK

if [ "$rank_id" -eq 0 ]; then
    python3 -m sglang.launch_server \
        --model-path $model_name \
        --tp $tp \
        --dist-init-addr $MASTER_ADDR:$MASTER_PORT \
        --nnodes $nnodes \
        --node-rank 0 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 30000
else
    python3 -m sglang.launch_server \
        --model-path $model_name \
        --tp $tp \
        --dist-init-addr $MASTER_ADDR:$MASTER_PORT \
        --nnodes $nnodes \
        --trust-remote-code \
        --node-rank $rank_id
fi

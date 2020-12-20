DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA=data/
LOG=logs/
MODEL=./logs/model_last.pth.tar
clear

python3 test.py --directory $DIR \
    --data $DATA \
    --log_dir $LOG \
    --local_rank 0 \
    --batch_size 20 \
    --num_workers 4 \
    --log_interval 5 \
    --val_interval 5 \
    --model $MODEL \



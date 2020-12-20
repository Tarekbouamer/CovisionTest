DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA=data/
LOG=logs/
RESUME=./logs/model_last.pth.tar
clear
python3 train.py --directory $DIR \
    --data $DATA \
    --log_dir $LOG \
    --batch_size 20 \
    --num_workers 4 \
    --log_interval 10 \
    --val_interval 5 \
    --local_rank 0 \
    --epochs 60 \
    --learning_rate 1e-3 \
    --weight_decay 1e-1 \
    --refresh False \
    #--resume $RESUME \



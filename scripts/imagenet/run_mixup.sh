DATASET=imagenet
DATA_ROOT='~/generalizationconfusion/datasets/ImageNet/Data/CLS-LOC'
ARCH=resnet50
LR=0.3
LR_SCHEDULE='cosine'
EPOCHS=95
BATCH_SIZE=16
LOSS=sat
ALPHA=0.99
ES=50
NOISE_RATE=$2
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='noisy_train'
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_${NOISE_TYPE}_r${NOISE_RATE}_m${ALPHA}_p${ES}_${LR_SCHEDULE}_mixup$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --sat-alpha ${ALPHA} --sat-es ${ES} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --base-width 64 --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} --is_tpu --mixup --mixup-alpha $3 \
        --mixup-gamma 0.1 --logdir "logs/imagenet"

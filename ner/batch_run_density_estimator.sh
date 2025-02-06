#!/bin/bash -eu
#SBATCH --job-name=nlp_calibration
#SBATCH --cpus-per-task=10
#SBATCH --output=output.%J.log
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00


export BASE_DATA_DIR=/work/wataru-ha/data/ner_data
export BASE_DIR=/work/wataru-ha/knn_softmax/ner
export TODAYDATE=$(date '+%Y-%m-%d-%H-%M-%S')
export MAX_LENGTH=512
export MODEL_NAME_OR_PATH="pass to your trained Baseline (Softmax Response) model path"
export MODEL_TYPE=BaselineDeBERTaV3ForTokenClassification
export DENSITY_ESTIMATOR_TYPE=RealNVP
export BATCH_SIZE=32
export DENSITY_ESTIMATOR_BATCH_SIZE=128
export DATA_NAME=conll2003
export DATA_PATH=$BASE_DATA_DIR/$DATA_NAME
export SEED=1
export USE_OTHER_UNDER_SAMPLING=false
export FEATURE_NORMALIZATION=false
export USE_PCA=false


if "${FEATURE_NORMALIZATION}" ; then
    OUTPUT_DIR="$BASE_DIR/models/$DATA_NAME-$MODEL_TYPE-$DENSITY_ESTIMATOR_TYPE-density-estimation-feature-normalization-f1"
else
    OUTPUT_DIR="$BASE_DIR/models/$DATA_NAME-$MODEL_TYPE-$DENSITY_ESTIMATOR_TYPE-density-estimation-f1"
fi

if "${USE_PCA}" ; then
    OUTPUT_DIR+="-pca"
fi

export NUM_EPOCHS=7
export EMBEDDING_DIM=768
export PCA_EMBEDDING_DIM=256
export EVAL_STEPS=200
export LEARNING_RATE=1e-4  # 5e-5, 1e-4
export RUN_MULTIPLE_SEEDS=false


mkdir -p $OUTPUT_DIR

cd $BASE_DIR

module load singularity

singularity run --nv /work/knn_ue/research-dev.sif /opt/conda/bin/python src/train_density_estimator_from_json.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--data_path $DATA_PATH \
--labels $DATA_PATH/labels.txt \
--output_dir $OUTPUT_DIR \
--model_type $MODEL_TYPE \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--seed $SEED \
--use_fast \
--bf16 \
--learning_rate $LEARNING_RATE \
--eval_steps $EVAL_STEPS \
--density_estimator_type $DENSITY_ESTIMATOR_TYPE \
--embedding_dim $EMBEDDING_DIM \
--density_estimator_batch_size $DENSITY_ESTIMATOR_BATCH_SIZE \
--use_other_under_sampling $USE_OTHER_UNDER_SAMPLING \
--feature_normalization $FEATURE_NORMALIZATION \
--use_pca $USE_PCA \
--pca_embedding_dim $PCA_EMBEDDING_DIM

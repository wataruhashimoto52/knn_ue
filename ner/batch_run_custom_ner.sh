#!/bin/bash -eu
#SBATCH --job-name=nlp_calibration
#SBATCH --cpus-per-task=8
#SBATCH --output=output.%J.log
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00


export BASE_DATA_DIR=/work/knn_ue/data/ner
export BASE_DIR=/work/knn_ue/ner
export TODAYDATE=$(date '+%Y-%m-%d-%H-%M-%S')
export MAX_LENGTH=512
export BERT_MODEL=microsoft/mdeberta-v3-base  # microsoft/mdeberta-v3-base, bert-base-multilingual-cased
export MODEL_TYPE=AutoTuneFaissKNNDeBERTaV3ForTokenClassification
export BATCH_SIZE=32
export DATA_NAME=ontonotes5_bn
export USE_OOD_TEST=true
export DATA_PATH=$BASE_DATA_DIR/$DATA_NAME

export TAU=1.0
export SMOOTHING=0.3   # For LS. [0.01, 0.05, 0.1, 0.2, 0.3]
export STOPPING_CRITERIA=f1
export OPTIMIZE_TYPE=maximize
export NUM_EPOCHS=200
export EVAL_STEPS=200
export LEARNING_RATE=1e-5

export SAVE_MODEL=false
export SEED=7
export RUN_MULTIPLE_SEEDS=true

export NUM_MONTE_CARLO=20    # 10, 20, 50, 100 (default; 20)
export SPECTRAL_NORM_UPPER_BOUND=0.95
export BM_COEFF=1e-3
export BM_PRIOR=1.0

# for Posterior Networks
export PN_DENSITY_TYPE='radial_flow'   # 'planar_flow', 'radial_flow'
export PN_LATENT_DIM=50

# for kNN's
export DENSITY_ESTIMATOR_PATH=models/conll2003-BaselineDeBERTaV3ForTokenClassification-RealNVP-density-estimation-f1
export DENSITY_ESTIMATOR_TYPE=RealNVP
export KNN_TOPK=32
export N_LIST=100
export N_SUBVECTORS=32
export KNN_NPROBE=8
export INDEX_TYPE=FLAT
export KNN_ALPHA=1.0
export KNN_TEMPERATURE=800
export USE_NEIGHBOR_LABELS=false
export USE_DAC_ONLY_LAST_LAYER=false
export USE_FEATURE_NORMALIZATION=false
export USE_OPQ=false
export USE_PCA=false
export USE_RECOMP=false
export TRANSFORM_DIM=128


OUTPUT_DIR="$BASE_DIR/models/$DATA_NAME-$DATA_NAME-$MODEL_TYPE-$STOPPING_CRITERIA-$"

if [[ "$BERT_MODEL" == "microsoft/deberta-v3-base" ]]; then
    OUTPUT_DIR+="-en"
fi

if [[ "$MODEL_TYPE" == *"LabelSmoothing"* ]] ; then
    OUTPUT_DIR+="-smooth$SMOOTHING"
fi

if "${SAVE_MODEL}" ; then
    OUTPUT_DIR+="-saved"
    export RUN_MULTIPLE_SEEDS=false
fi

if "${USE_PCA}" ; then
    export DENSITY_ESTIMATOR_PATH=models/ontonotes5_bn-BaselineDeBERTaV3ForTokenClassification-RealNVP-density-estimation-ece-pca
fi

if [[ "$MODEL_TYPE" == *"DensitySoftmax"* ]] ; then
    export BERT_MODEL='hoge'
    OUTPUT_DIR+="-$DENSITY_ESTIMATOR_TYPE"
fi

if [[ "$DENSITY_ESTIMATOR_PATH" == *"feature-normalization"* ]] ; then
    OUTPUT_DIR+="-feature-normalization"
fi

if  [[ "$MODEL_TYPE" == "AutoTuneFaissKNNDeBERTaV3ForTokenClassification" ]] ; then
    OUTPUT_DIR+="-K$KNN_TOPK-NPROBE$KNN_NPROBE-$INDEX_TYPE"
    if [[ "$INDEX_TYPE" == *"IVF"* ]]; then
        OUTPUT_DIR+="-NLIST$N_LIST"
    fi
    if [[ "$INDEX_TYPE" == *"PQ"* ]]; then
        OUTPUT_DIR+="-NSUBVECTORS$N_SUBVECTORS"
    fi

    if "${USE_NEIGHBOR_LABELS}" ; then
        OUTPUT_DIR+="-use-neighbor-labels"
    fi
fi

if  [[ "$MODEL_TYPE" == "AutoTuneDACDeBERTaV3ForTokenClassification" ]] ; then
    OUTPUT_DIR+="-K$KNN_TOPK-NPROBE$KNN_NPROBE-$INDEX_TYPE"
    if "${USE_DAC_ONLY_LAST_LAYER}" ; then
        OUTPUT_DIR+="-only-lastlayer"
    fi
fi

if  [[ "$MODEL_TYPE" == *"SLPN"* ]] ; then
    OUTPUT_DIR+="-$PN_DENSITY_TYPE-latent$PN_LATENT_DIM"
fi

if  [[ "$MODEL_TYPE" == *"PosteriorNetworks"* ]] ; then
    OUTPUT_DIR+="-$PN_DENSITY_TYPE-latent$PN_LATENT_DIM"
fi


if [[ "$MODEL_TYPE" == *"MCDropout"* ]] ; then
    OUTPUT_DIR+="-NUM_MC$NUM_MONTE_CARLO"
fi


if "${USE_OPQ}" ; then
    OUTPUT_DIR+="-opq-$TRANSFORM_DIM"
fi

if "${USE_PCA}" ; then
    OUTPUT_DIR+="-pca-$TRANSFORM_DIM"
fi

if "${USE_FEATURE_NORMALIZATION}" ; then
    OUTPUT_DIR+="-feature-normalization"
fi

if "${USE_RECOMP}" ; then
    OUTPUT_DIR+="-recomp"
fi


if "${RUN_MULTIPLE_SEEDS}" ; then
    export LOGFILE_NAME=$OUTPUT_DIR/$TODAYDATE-multple-seeds.log
else
    export LOGFILE_NAME=$OUTPUT_DIR/$TODAYDATE-seed-$SEED.log
fi

mkdir -p $OUTPUT_DIR

cd $BASE_DIR

module load singularity

singularity run --nv /work/knn_ue/research-dev.sif /opt/conda/bin/python src/train_from_json.py \
--model_name_or_path $BERT_MODEL \
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
--run_multiple_seeds $RUN_MULTIPLE_SEEDS \
--tau $TAU \
--belief_matching_coeff $BM_COEFF \
--belief_matching_prior $BM_PRIOR \
--num_monte_carlo $NUM_MONTE_CARLO \
--spectral_norm_upper_bound $SPECTRAL_NORM_UPPER_BOUND \
--save_model $SAVE_MODEL \
--stopping_criteria $STOPPING_CRITERIA \
--optimize_type $OPTIMIZE_TYPE \
--use_ood_test $USE_OOD_TEST \
--smoothing $SMOOTHING \
--density_estimator_path $DENSITY_ESTIMATOR_PATH \
--knn_topk $KNN_TOPK \
--knn_alpha $KNN_ALPHA \
--knn_temperature $KNN_TEMPERATURE \
--use_pca $USE_PCA \
--use_opq $USE_OPQ \
--use_feature_normalization $USE_FEATURE_NORMALIZATION \
--use_dac_only_last_layer $USE_DAC_ONLY_LAST_LAYER \
--knn_nprobe $KNN_NPROBE \
--use_recomp $USE_RECOMP \
--transform_dim $TRANSFORM_DIM \
--index_type $INDEX_TYPE \
--n_list $N_LIST \
--n_subvectors $N_SUBVECTORS \
--use_neighbor_labels $USE_NEIGHBOR_LABELS \
--pn_density_type $PN_DENSITY_TYPE \
--pn_latent_dim $PN_LATENT_DIM

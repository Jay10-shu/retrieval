export MASTER_PORT=12450
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export LOCAL_RANK=0

DATA_PATH="/home/mfy/code/PAU-main/data"
SAVE_PATH="log-activity-test"
MODEL_PATH="./log-activity/cluster-results_15_1.1/pytorch_model_15_.bin.6"
python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT\
    main_DLAR.py --do_eval  --num_thread_reader=10 \
    --lr 1e-4 --batch_size=64 --batch_size_val 42 \
    --epochs=10  --n_display=10\
    --init_model ${MODEL_PATH} \
    --data_path ${DATA_PATH}/ActivityNet \
    --features_path ${DATA_PATH}/ActivityNet/videos/\
    --output_dir ${SAVE_PATH} \
    --max_words 32 --max_frames 24 \
    --datatype activity \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --tau 20  --lambda1 1 --lambda2 1 \
    --num_clusters 15 \
    --fuzzy_index 1.1 \
    --loose_type --linear_patch 2d --sim_header seqTransf \



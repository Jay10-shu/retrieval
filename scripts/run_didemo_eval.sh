export MASTER_PORT=2955
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export LOCAL_RANK=0

DATA_PATH="/home/mfy/code/PAU-main/data"
SAVE_PATH="log-didemo-test"
MODEL_PATH="/home/sj/MM-retrieval/log-didemo/pytorch_model_15_.bin.3"
python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT\
    main_DLAR.py --do_eval --num_thread_reader=12 \
    --epochs=5 --batch_size=32 --n_display=10 \
    --eval_epoch=3 \
    --init_model ${MODEL_PATH} \
    --data_path ${DATA_PATH}/DiDeMo/annotation \
    --features_path ${DATA_PATH}/DiDeMo/video \
    --output_dir ${SAVE_PATH} \
    --lr 1e-4 --max_words 32 --max_frames 36 --batch_size_val 103 \
    --datatype didemo \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --tau 20  \
    --num_clusters 15 \
    --fuzzy_index 1.1 \
    --lambda1 1 --lambda2 1 \
    --loose_type --linear_patch 2d --sim_header seqTransf \

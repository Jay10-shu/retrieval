export MASTER_PORT=1220
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export LOCAL_RANK=0

DATA_PATH="/home/mfy/code/PAU-main/data"
SAVE_PATH="log-didemo"
python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT \
    main_DLAR.py --do_train --num_thread_reader=14 \
    --epochs=5 --batch_size=32 --n_display=20 \
    --data_path ${DATA_PATH}/DiDeMo/annotation \
    --features_path ${DATA_PATH}/DiDeMo/video_compressed \
    --output_dir ${SAVE_PATH} \
    --lr 1e-4 --max_words 64 --max_frames 32 --batch_size_val 8 \
    --datatype didemo \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --tau 20 --lambda1 1 --lambda2 1 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --num_clusters 15 \
    --fuzzy_index 1.1 \
    # --extract_feature \



  
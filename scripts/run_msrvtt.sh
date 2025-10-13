export MASTER_PORT=6110
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export LOCAL_RANK=0


DATA_PATH="/home/mfy/code/PAU-main/data"
SAVE_PATH="log-msrvtt1"
python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT \
    main_DLAR1.py --do_train  --num_thread_reader=10 \
    --lr 1e-4 --batch_size=96 --batch_size_val 100 \
    --epochs=5  --n_display=50\
    --train_csv ${DATA_PATH}/msrvtt_data//MSRVTT_train.9k.csv\
    --val_csv ${DATA_PATH}/msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/msrvtt_data/MSRVTT_data.json \
    --features_path ${DATA_PATH}/msrvtt_data/MSRVTT_Videos \
    --output_dir ${SAVE_PATH} \
    --max_words 32 --max_frames 12 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --tau 20  \
    --lambda1 1 --lambda2 1 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --num_clusters 100 \
    --fuzzy_index 1.1

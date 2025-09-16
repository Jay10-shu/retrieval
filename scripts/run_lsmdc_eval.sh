export MASTER_PORT=1152
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export LOCAL_RANK=2

DATA_PATH="/home/mfy/code/PAU-main/data"
SAVE_PATH="log-lsmdc-test"
MODEL_PATH="./log-lsmdc/pytorch_model_15_.bin"
python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT\
    main_DLAR.py --do_eval --num_thread_reader=4 \
    --epochs=5 --batch_size=64 --n_display=10 \
    --data_path ${DATA_PATH}/lsdmc \
    --features_path ${DATA_PATH}/lsdmc \
    --init_model ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 37 \
    --datatype lsmdc \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --tau 20 --lambda1 1 --lambda2 1 \
    --num_clusters 15 \
    --fuzzy_index 1.1 \
 


    
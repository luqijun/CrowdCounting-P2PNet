CUDA_VISIBLE_DEVICES=0 python train.py --data_root /mnt/d/MyDocs/Datasets/ShanghaiTech/ShanghaiTech_P2PNet \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --match_type anchor_points \
    --eval_start 350 \
    --eval_freq 1 \
    --gpu_id 0
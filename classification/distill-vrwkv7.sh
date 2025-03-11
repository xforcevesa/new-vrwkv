python -m torch.distributed.launch \
    --nnodes=1 --node_rank=0 --nproc_per_node=8 \
    --master_addr="127.0.0.1" --master_port=29509 \
    distill_main.py --batch-size 16 --data-path ./imagenet/ \
    --output ./tmp/ --cfg configs/vrwkv7/vrwkv7v_tiny_224.yaml --distill
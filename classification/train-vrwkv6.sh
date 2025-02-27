python -m torch.distributed.launch \
    --nnodes=1 --node_rank=0 --nproc_per_node=8 \
    --master_addr="127.0.0.1" --master_port=29501 \
    main.py --batch-size 32 --data-path ./imagenet/ \
    --output ./tmp/ --cfg configs/vrwkv6/vrwkv6v_tiny_224.yaml
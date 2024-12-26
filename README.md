# Vision RWKV (New Generation)

New structure for `vrwkv6` model is ready in classification, fully based on the `VMamba` project.

While `vrwkv7` is in progress, awaiting for further testing...

We'll also add other models for contrastive experiments.

Please follow the instructions in [the old README](README_OLD.md) to setup the environment.

See classification [here](./classification/readme.md).

## Issue

The wind_rwkv7 CUDA implementation may cause SIGABRT, with `dmesg` log like this:

```dmesg
[660093.553553] NVRM: Xid (PCI:0000:0f:00): 13, pid='<unknown>', name=<unknown>, Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0): Illegal Instruction Parameter
[660093.555065] NVRM: Xid (PCI:0000:0f:00): 13, pid='<unknown>', name=<unknown>, Graphics SM Global Exception on (GPC 0, TPC 0, SM 0): Multiple Warp Errors
[660093.556620] NVRM: Xid (PCI:0000:0f:00): 13, pid='<unknown>', name=<unknown>, Graphics Exception: ESR 0x505730=0x3000b 0x505734=0x24 0x505728=0x1f81fb60 0x50572c=0x1174
[660093.558195] NVRM: Xid (PCI:0000:0f:00): 13, pid='<unknown>', name=<unknown>, Graphics SM Warp Exception on (GPC 0, TPC 0, SM 1): Illegal Instruction Parameter
...
[660094.064010] NVRM: Xid (PCI:0000:0f:00): 13, pid='<unknown>', name=<unknown>, Graphics Exception: ESR 0x57d730=0x2000b 0x57d734=0x24 0x57d728=0x1f81fb60 0x57d72c=0x1174
[660094.065577] NVRM: Xid (PCI:0000:0f:00): 13, pid='<unknown>', name=<unknown>, Graphics SM Warp Exception on (GPC 7, TPC 8, SM 1): Illegal Instruction Parameter
[660094.066934] NVRM: Xid (PCI:0000:0f:00): 13, pid='<unknown>', name=<unknown>, Graphics SM Global Exception on (GPC 7, TPC 8, SM 1): Multiple Warp Errors
[660094.068651] NVRM: Xid (PCI:0000:0f:00): 13, pid='<unknown>', name=<unknown>, Graphics Exception: ESR 0x57d7b0=0x1000b 0x57d7b4=0x24 0x57d7a8=0x1f81fb60 0x57d7ac=0x1174
[660094.073526] NVRM: Xid (PCI:0000:0f:00): 43, pid=1660871, name=python, Ch 0000000a
```

The model code is in [classification/models/vrwkv7.py](classification/models/vrwkv7.py) and the implementation is in [classification/models/cuda_v7/wind_rwkv7.cpp](classification/models/cuda_v7/wind_rwkv7.cpp) and [classification/models/cuda_v7/wind_rwkv7.cu](classification/models/cuda_v7/wind_rwkv7.cu).

You can set the `wind_cuda` flag in the line 65 of `vrwkv7.py` to True and rerun the training to reproduce this issue.

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

### Runtime Environment

Server Fastfetch:

```
                             ....
              .',:clooo:  .:looooo:.
           .;looooooooc  .oooooooooo'          demo@server
        .;looooool:,''.  :ooooooooooc          -----------------
       ;looool;.         'oooooooooo,          OS: Ubuntu jammy 22.04 x86_64
      ;clool'             .cooooooc.  ,,       Host: NF5688-M7-A0-R0-00 (0)
         ...                ......  .:oo,      Kernel: Linux 5.15.0-113-generic
  .;clol:,.                        .loooo'     Uptime: 7 days, 18 hours, 24 mins
 :ooooooooo,                        'ooool     Packages: 1143 (dpkg)
'ooooooooooo.                        loooo.    Shell: bash 5.1.16
'ooooooooool                         coooo.    Display (VGA-1): 1024x768 [External]
 ,loooooooc.                        .loooo.    Terminal: node
   .,;;;'.                          ;ooooc     CPU: 2 x Intel(R) Xeon(R) Platinum 8468V (96) @ 3.80 GHz
       ...                         ,ooool.     GPU 1: ASPEED Technology, Inc. ASPEED Graphics Family
    .cooooc.              ..',,'.  .cooo.      GPU 2: NVIDIA Device 2324 (3D)
      ;ooooo:.           ;oooooooc.  :l.       GPU 3: NVIDIA Device 2324 (3D)
       .coooooc,..      coooooooooo.           GPU 4: NVIDIA Device 2324 (3D)
         .:ooooooolc:. .ooooooooooo'           GPU 5: NVIDIA Device 2324 (3D)
           .':loooooo;  ,oooooooooc            GPU 6: NVIDIA Device 2324 (3D)
               ..';::c'  .;loooo:'             GPU 7: NVIDIA Device 2324 (3D)
                                               GPU 8: NVIDIA Device 2324 (3D)
                                               GPU 9: NVIDIA Device 2324 (3D)
                                               Memory: 105.09 GiB / 1.97 TiB (5%)
                                               Swap: 0 B / 1024.00 MiB (0%)
                                               Disk (/): 406.03 GiB / 1.72 TiB (23%) - ext4
                                               Disk (/data): 622.54 GiB / 6.93 TiB (9%) - ext4
```

NVIDIA GPUs:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H800                    On  |   00000000:0F:00.0 Off |                    0 |
| N/A   44C    P0             89W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H800                    On  |   00000000:34:00.0 Off |                    0 |
| N/A   36C    P0            111W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H800                    On  |   00000000:48:00.0 Off |                    0 |
| N/A   43C    P0             81W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H800                    On  |   00000000:5A:00.0 Off |                    0 |
| N/A   37C    P0            130W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA H800                    On  |   00000000:87:00.0 Off |                    0 |
| N/A   44C    P0            128W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA H800                    On  |   00000000:AE:00.0 Off |                    0 |
| N/A   35C    P0             93W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA H800                    On  |   00000000:C2:00.0 Off |                    0 |
| N/A   44C    P0            140W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA H800                    On  |   00000000:D7:00.0 Off |                    0 |
| N/A   34C    P0            107W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Toolchain version:

```
$ gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

$ cmake --version
cmake version 3.22.1

CMake suite maintained and supported by Kitware (kitware.com/cmake).
$ ninja -v
ninja: error: loading 'build.ninja': No such file or directory
$ ninja --version
1.11.1.git.kitware.jobserver-1
```

Conda environment:

```yml
name: vmamba
channels:
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - bzip2=1.0.8=h5eee18b_6
  - ca-certificates=2024.11.26=h06a4308_0
  - ld_impl_linux-64=2.40=h12ee557_0
  - libffi=3.4.4=h6a678d5_1
  - libgcc-ng=11.2.0=h1234567_1
  - libgomp=11.2.0=h1234567_1
  - libstdcxx-ng=11.2.0=h1234567_1
  - libuuid=1.41.5=h5eee18b_0
  - ncurses=6.4=h6a678d5_0
  - openssl=3.0.15=h5eee18b_0
  - pip=24.2=py310h06a4308_0
  - python=3.10.16=he870216_1
  - readline=8.2=h5eee18b_0
  - setuptools=75.1.0=py310h06a4308_0
  - sqlite=3.45.3=h5eee18b_0
  - tk=8.6.14=h39e8969_0
  - wheel=0.44.0=py310h06a4308_0
  - xz=5.4.6=h5eee18b_1
  - zlib=1.2.13=h5eee18b_1
  - pip:
      - addict==2.4.0
      - chardet==5.2.0
      - click==8.1.8
      - cloudpickle==3.1.0
      - contourpy==1.3.1
      - cycler==0.12.1
      - einops==0.8.0
      - exceptiongroup==1.2.2
      - filelock==3.16.1
      - fonttools==4.55.3
      - fsspec==2024.12.0
      - ftfy==6.3.1
      - fvcore==0.1.5.post20221221
      - importlib-metadata==8.5.0
      - iniconfig==2.0.0
      - iopath==0.1.10
      - jinja2==3.1.5
      - kiwisolver==1.4.8
      - markdown==3.7
      - markdown-it-py==3.0.0
      - markupsafe==3.0.2
      - mat4py==0.6.0
      - matplotlib==3.10.0
      - mdurl==0.1.2
      - mmcv==2.1.0
      - mmdet==3.3.0
      - mmengine==0.10.1
      - mmpretrain==1.2.0
      - mmsegmentation==1.2.2
      - model-index==0.1.11
      - modelindex==0.0.2
      - mpmath==1.3.0
      - networkx==3.4.2
      - ninja==1.11.1.3
      - numpy==2.2.1
      - nvidia-cublas-cu12==12.4.5.8
      - nvidia-cuda-cupti-cu12==12.4.127
      - nvidia-cuda-nvrtc-cu12==12.4.127
      - nvidia-cuda-runtime-cu12==12.4.127
      - nvidia-cudnn-cu12==9.1.0.70
      - nvidia-cufft-cu12==11.2.1.3
      - nvidia-curand-cu12==10.3.5.147
      - nvidia-cusolver-cu12==11.6.1.9
      - nvidia-cusparse-cu12==12.3.1.170
      - nvidia-nccl-cu12==2.21.5
      - nvidia-nvjitlink-cu12==12.4.127
      - nvidia-nvtx-cu12==12.4.127
      - opencv-python==4.10.0.84
      - opencv-python-headless==4.10.0.84
      - ordered-set==4.1.0
      - packaging==24.2
      - pandas==2.2.3
      - pillow==11.0.0
      - platformdirs==4.3.6
      - pluggy==1.5.0
      - portalocker==3.0.0
      - prettytable==3.12.0
      - protobuf==5.29.2
      - pycocotools==2.0.8
      - pygments==2.18.0
      - pyparsing==3.2.0
      - pytest==8.3.4
      - python-dateutil==2.9.0.post0
      - pytz==2024.2
      - pyyaml==6.0.2
      - regex==2024.11.6
      - rich==13.9.4
      - scipy==1.14.1
      - seaborn==0.13.2
      - selective-scan==0.0.2
      - shapely==2.0.6
      - six==1.17.0
      - submitit==1.5.2
      - sympy==1.13.1
      - tabulate==0.9.0
      - tensorboardx==2.6.2.2
      - termcolor==2.5.0
      - terminaltables==3.1.10
      - timm==0.4.12
      - tomli==2.2.1
      - torch==2.5.1
      - torchaudio==2.5.1
      - torchvision==0.20.1
      - tqdm==4.67.1
      - triton==3.1.0
      - typing-extensions==4.12.2
      - tzdata==2024.2
      - wcwidth==0.2.13
      - yacs==0.1.8
      - yapf==0.43.0
      - zipp==3.21.0
```

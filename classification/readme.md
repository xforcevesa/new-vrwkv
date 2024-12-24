## origins

based on https://github.com/microsoft/Swin-Transformer#20240103

`main.py` and `utils/utils_ema.py` is modified from https://github.com/microsoft/Swin-Transformer#20240103, based on https://github.com/facebookresearch/ConvNeXt#20240103

## Run

To run training for `vrwkv6` model, run:

```bash
mkdir tmp
# Downloading your ImageNet dataset
# Ensuring it is in the directory named "imagenet"
bash train.sh
```

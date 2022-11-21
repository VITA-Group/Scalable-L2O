# Generating Subspaces

## Experiments

Examples: CIFAR-100 w. RN-18

```python
CUDA_VISIBLE_DEVICES=2 python train.py --arch resnet18 --save-dir resnet18 --seed 1 --epochs 100 --momentum 0.0 --dataset CIFAR10 --save-every 1 
python generate_subspaces.py --params_start 0 --params_end 60 --n_components 40 --arch resnet18 --datasets CIFAR10 --save-dir resnet18
cp resnet18/0.pt resnet18_CIFAR10_0.pt
```

## Acknowledgement 

We sincerely thank [DLDR](https://github.com/nblt/DLDR) where we extracted part of the codes.
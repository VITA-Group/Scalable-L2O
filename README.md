# Scalable-L2O

Implementation of ECCV 2022 paper: Scalable Learning to Optimize: A Learned Optimizer Can Train Big Models. 

## Environment

We recommend using Anaconda to manage the virtual environment. 

```bash
conda env create -f environment.yaml
conda activate sl2o
```

## Experiments

### CNNs
#### Subspaces

We provide pre-generated subspaces for models in this [link]. Optionally, one can generate the subspaces by themselves. Please refer to the `subspaces` directory for more details. 

#### Meta-Training

ResNet-18 (CIFAR-10)
```bash
python -u src/resnet18_ft_de.py --max_epoch 20 --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10 --meta_train_eval_epoch 2 
```

ResNet-18 (CIFAR-100)
```bash
python -u src/resnet18_ft_de.py --max_epoch 20 --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10 --meta_train_eval_epoch 2 --dataset CIFAR100
```


ResNet8 (CIFAR-10)
```bash
python -u src/resnet8_ft_de.py --max_epoch 20 --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10 --meta_train_eval_epoch 2 
```

ResNet20 (CIFAR-10)
```bash
python -u src/resnet20_ft_de.py --max_epoch 20 --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10 --meta_train_eval_epoch 2
```

#### Meta-Testing

ResNet-8 (CIFAR-10)
```bash
python -u src/resnet8_eval_de.py --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10 --max_epoch 100
```

ResNet-20 (CIFAR-10)
```bash
python -u src/resnet20_eval_de.py --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10 --max_epoch 100
```

### VITs

#### Meta-Training
```bash
python -u src/vit_ft.py --max_epoch 20 --lora_dim 16 --lora_alpha 32 --lora_dropout 0.1 --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 64 --unroll 10 --random_seed 1 --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --meta_train_eval_epoch 2 
```

#### Meta-Testing
```bash
python -u src/vit_ft_eval.py --max_epoch 20 --lora_dim 16 --lora_alpha 32 --lora_dropout 0.1 --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 10 --training_steps 1000 --batch-size 64 --unroll 10 --random_seed 1 --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --random_seed 1 --eval_interval 391 
```
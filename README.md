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

ResNet20 (CIFAR-100)
```bash
python -u src/resnet20_ft_de.py --max_epoch 20 --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10 --meta_train_eval_epoch 2 --dataset CIFAR100
```

#### Meta-Testing

ResNet-8 (CIFAR-10)
```bash
python -u src/resnet8_eval_de.py --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10
```

ResNet-20 (CIFAR-100)
```bash
python -u src/resnet20_eval_de.py --eval_interval 2000 --log_interval 100 --hidden_sz 8 --scale 1e-4 --log_interval 5 --training_steps 1000 --batch-size 128 --unroll 10 --dataset CIFAR100 --max_epoch 100
```
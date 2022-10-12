# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import wandb

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    SchedulerType,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from exp_utils import create_exp_dir, set_seed, AverageMeter, rgetattr, rsetattr, evaluate_cifar10, print_args, hutch_loop, group_product
import time

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")



from PIL import Image
def pil_loader(array):
    return Image.fromarray(array.astype(np.uint8)).convert('RGB')

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_dir", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_dir", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

    parser.add_argument('--unroll_length', type=int, default=5, help='num_unroll')

    parser.add_argument('--training_steps', type=int, default=100, help='training_steps')

    parser.add_argument('--hidden_sz', type=int, default=20, help='hidden_size')

    parser.add_argument('--il_weight', type=float, default=1e-3, help='il')

    parser.add_argument('--cl', action="store_true")

    parser.add_argument('--il', action="store_true")

    parser.add_argument('--stronger-il', action="store_true")

    parser.add_argument('--meta-lr', default=1e-2)

    parser.add_argument('--random_scaling', action="store_true")

    parser.add_argument('--name', type=str)

    parser.add_argument('--use-second-layer', action="store_true")

    parser.add_argument('--hessian', action="store_true")

    parser.add_argument('--scale', default=1e-4, type=float)

    parser.add_argument('--optimizer_checkpoint', default=None, type=str)
    parser.add_argument("--platform", default='single', type=str, help='platform cloud')
    parser.add_argument("--local_rank", default=0, type=int, help='local rank')
    parser.add_argument("--rank", default=0, type=int, help='rank')
    parser.add_argument("--device", default=0, type=int, help='device')
    parser.add_argument("--world_size", default=0, type=int, help='world size')
    parser.add_argument("--random_seed", default=10, type=int, help='random seed')
    parser.add_argument('--meta-optimizer', type=str, default='Adam')
    parser.add_argument('--work_dir', type=str, default="work")
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    parser.add_argument('--late-update', action="store_true")

    parser.add_argument('--hessian-coef', default=1e-3, type=float)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--lora_dim', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)


    args = parser.parse_args()

    return args

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    args = parse_args()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    ds = load_dataset(
        'cifar10',
        None,
        data_files=None,
        cache_dir=None
    )
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    _train_transforms = Compose(
        [
            RandomResizedCrop(args.image_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            normalize,
        ]
    )


    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        img = np.array(example_batch['img'])[0]
        result = {}
        result["pixel_values"] = _train_transforms(pil_loader(img)).unsqueeze(0)
        result["label"] = example_batch['label']
        return result

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        img = np.array(example_batch['img'])[0]
        result = {}
        result["pixel_values"] = _val_transforms(pil_loader(img)).unsqueeze(0)
        result["label"] = example_batch['label']
        return result
    args.train_val_split = None if "validation" in ds.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = ds["train"].train_test_split(args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    labels = ds["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    from networks import RNNOptimizer
    args.name = f"eval_opt_deit_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}"
    args.work_dir = f"./trained_models/eval_opt_deit_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}"
    args.optimizer_checkpoint = f"./trained_models/opt_deit_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}/best.pt"
    opt_net = RNNOptimizer(preproc=True, hidden_sz=args.hidden_sz, use_second_layer=args.use_second_layer)
    opt_net.cuda()
    opt_net.load_state_dict(torch.load(args.optimizer_checkpoint, map_location="cpu")['optimizer_state_dict'])
    opt_net.cuda()
    os.makedirs(args.work_dir, exist_ok=True)
    wandb.init(project=f"l2o_lora", entity="xxchen", name=args.name)
    wandb.config.update({'hidden_sz': args.hidden_sz, 'training_steps': args.training_steps, 'unroll_length': args.unroll_length})
    from configuration_deit import DeiTConfig
    from feature_extraction_deit import DeiTFeatureExtractor
    from modeling_deit import DeiTForImageClassification
    config = DeiTConfig.from_pretrained(
        'facebook/deit-base-distilled-patch16-224',
        #apply_lora=True,
        lora_r=args.lora_dim,
        lora_alpha=args.lora_alpha,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification")

    feature_extractor = DeiTFeatureExtractor.from_pretrained(
        'facebook/deit-base-distilled-patch16-224',
        cache_dir=None,
        use_auth_token=None,
        size=args.image_size,
        image_mean=normalize.mean,
        image_std=normalize.std,
    )

    model = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-224',
        config=config
    )
    ds["train"].set_transform(train_transforms)
    ds["validation"].set_transform(val_transforms)
    

    n_params = 0
    for name, param in model.named_parameters():
        param.requires_grad = ((not ('lora' not in name and (name.startswith('bert') or name.startswith('deberta')))) or 'coef' in name)
        print(name)
        print(param.requires_grad)

    for name, p in model.named_parameters():
        if p.requires_grad:
            n_params += int(np.prod(p.size()))

    print(n_params)
    if args.meta_optimizer == 'Adam':
        meta_optimizer = torch.optim.Adam(opt_net.parameters(), lr=args.meta_lr)
    elif args.meta_optimizer == 'AdamW':
        meta_optimizer = torch.optim.AdamW(opt_net.parameters(), lr=args.meta_lr)
    elif args.meta_optimizer == 'RMSprop':
        meta_optimizer = torch.optim.RMSprop(opt_net.parameters(), lr=args.meta_lr)
    # Preprocessing the datasets

    train_dataset = ds["train"]
    eval_dataset = ds['validation']

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size, num_workers=32,
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size, num_workers=32)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    args.logging = create_exp_dir(args.work_dir)
    # Train!
    total_batch_size = 32

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    completed_steps = 0
    beta1 = 0.9
    beta2 = 0.99

    def to_use(name):
        return ((not ('lora' not in name and (name.startswith('deit')))) or 'coef' in name)
    
    log_start_time = time.time()
    best_metric_value = 0

    mt = {}
    vt = {}
    hidden_states = {}
    cell_states = {}
    model = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-224',
        config=config
    )
    model = model.cuda()
    for name, p in model.named_parameters():
        if to_use(name):
            print(name)
            mt[name] = torch.zeros_like(p, requires_grad=False)
            vt[name] = torch.zeros_like(p, requires_grad=False)
            hidden_states[name] = [torch.zeros(p.nelement(), opt_net.hidden_sz, requires_grad=True).to(args.local_rank)]
            cell_states[name] = [torch.zeros(p.nelement(), opt_net.hidden_sz, requires_grad=True).to(args.local_rank)]
    for epoch in range(args.num_train_epochs):
        all_losses = []
        model.train()
        opt_net.eval()
        meta_optimizer.zero_grad()
        avg_lm_loss = AverageMeter()

        for step, batch in enumerate(train_dataloader):
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            loss = outputs.loss
            
            wandb.log({"meta_train/train_loss": loss}, step=completed_steps)
            
            
            result_params = {}
            completed_steps += 1
            
            loss.backward(retain_graph=True)
            all_losses.append(loss)
            avg_lm_loss.update(loss.item())
            result_h = {}
            result_c = {}
            for (name, p) in model.named_parameters():
                if to_use(name):
                    m, v, h, c = mt[name], vt[name], hidden_states[name], cell_states[name]
                    grad = p.grad.data
                    #print(grad.shape)
                    #print(m.shape)
                    m.mul_(beta1).add_((1 - beta1) * grad)
                    mt[name] = m
                    mt_hat = m / (1-beta1**(completed_steps))
                    v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
                    vt[name] = v
                    vt_hat = v / (1-beta2**(completed_steps))
                    mt_tilde = mt_hat / (torch.sqrt(vt_hat) + 1e-8)
                    gt_tilde = grad / (torch.sqrt(vt_hat) + 1e-8)

                    mt_tilde = mt_tilde.view(-1, 1)
                    gt_tilde = gt_tilde.view(-1, 1)

                    updates, nh, nc = opt_net(
                        torch.cat([mt_tilde, gt_tilde], 1),
                        h, c
                    )

                    result_h[name] = nh
                    result_c[name] = nc


                    result_params[name] = p - updates.view(*p.size()) * args.scale
                    #p.sub_(updates.view(*p.size()).detach() * args.scale)
                    result_params[name].retain_grad()        
            hidden_states = result_h
            cell_states   = result_c

            if True:
                meta_loss = sum(all_losses)
                #meta_optimizer.step()
                #meta_optimizer.zero_grad()
                #meta_scheduler.step()

                all_losses = []

                for name in result_params:
                    param = result_params[name].detach()
                    param.requires_grad_()
                    rsetattr(model, name, param)

                for h in hidden_states:
                    for hi in hidden_states[h]:
                        hi.detach_()
                        hi.requires_grad = True

                for c in cell_states:
                    for ci in cell_states[c]:
                        ci.detach_()
                        ci.requires_grad = True
                    
            if False:
                elapsed = time.time() - log_start_time

                log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches ' \
                                    '| meta loss {:5.2f}'.format(
                                    epoch, completed_steps, step + 1, 
                                    meta_loss) 
                wandb.log({"meta_train/meta_loss": meta_loss}, step=completed_steps)
                print(log_str)
                log_start_time = time.time()
                avg_lm_loss.reset()

        print("EVAL")
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
        eval_metric_value = list(eval_metric.values())[0]
        wandb.log({f"meta_train/{list(eval_metric.keys())[0]}": eval_metric_value}, step=completed_steps)
        model.train()

    '''
    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = eval_dataloader

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")
    '''


if __name__ == "__main__":
    main()

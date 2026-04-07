import os
import argparse
import warnings
import utils2

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='tinyimagenet', help="")
parser.add_argument('--out_num', type=int, default=50, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=224)

# optimization
parser.add_argument('--MAX_EPOCH', type=int, default=150)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--NAME', type=str, default='sgd')
parser.add_argument('--LR', type=float, default=0.002)
parser.add_argument('--LR_SCHEDULER', type=str, default='cosine')
parser.add_argument('--WARMUP_EPOCH', type=int, default=1)
parser.add_argument('--WARMUP_TYPE', type=str, default='constant')
parser.add_argument('--WARMUP_CONS_LR', type=float, default=1e-5)
parser.add_argument('--WEIGHT_DECAY', type=float, default=5e-4)
parser.add_argument('--MOMENTUM', type=float, default=0.9)
parser.add_argument('--SGD_DAMPNING', type=int, default=0)
parser.add_argument('--SGD_NESTEROV', type=utils2.str2bool, default=False)
parser.add_argument('--RMSPROP_ALPHA', type=float, default=0.99)
parser.add_argument('--ADAM_BETA1', type=float, default=0.9)
parser.add_argument('--ADAM_BETA2', type=float, default=0.999)
parser.add_argument('--STAGED_LR', type=utils2.str2bool, default=False)
parser.add_argument('--NEW_LAYERS', type=tuple, default=())
parser.add_argument('--BASE_LR_MULT', type=float, default=0.1)
parser.add_argument('--STEPSIZE', type=tuple, default=(-1,))
parser.add_argument('--GAMMA', type=float, default=0.1)
parser.add_argument('--WARMUP_RECOUNT', type=utils2.str2bool, default=True)
parser.add_argument('--WARMUP_MIN_LR', type=float, default=1e-5)

# Eval
parser.add_argument('--eval', action='store_true')

# model
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")
# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)
# misc
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--split_train_val', default=False, type=utils2.str2bool,
                    help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--use_default_parameters', default=False, type=utils2.str2bool,
                    help='Set to True to use optimized hyper-parameters from paper', metavar='BOOL')
# parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
# parser.add_argument('--gpus', default=[1], type=int, nargs='+',
#                         help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--split_idx', default=4, type=int, help='0-4 OSR splits for each dataset')

# Prompt
parser.add_argument('--ctx_num', type=int, default=16)
parser.add_argument('--ctx_init', type=str, default='')
parser.add_argument('--csc', type=utils2.str2bool, default=False)
parser.add_argument('--ctp', type=str, default='end')
parser.add_argument('--epoch', type=int, default=6)
parser.add_argument('--backbone', type=str, default='ViT-B/32')
parser.add_argument('--prec', type=str, default='fp16')  # fp16, fp32, amp

parser.add_argument('--save_path', type=str, default='./results/tinyimagnet')  # fp16, fp32, amp
parser.add_argument('--lambda1', type=float, default='1')
parser.add_argument('--lambda2', type=float, default='1')
parser.add_argument('--lambda3', type=float, default='1')
parser.add_argument('--alpha1', type=float, default='0.5')


args = parser.parse_args()
device = utils2.set_device(args.gpu)

import clip
import numpy as np
from torch.utils.data import DataLoader
from data.open_set_datasets import get_class_splits, get_datasets

from utils import init_seeds
import pandas as pd
import dspl_model

import torch
import torch.nn as nn
from optim import build_optimizer, build_lr_scheduler
from train import test, test_openset, train_double

from data.augmentations import get_transform
from data.imagenet import get_image_net_datasets
import json


init_seeds(args.seed)
# args.eval = True
# torch.cuda.set_device(args.gpu)
# device = "cuda" if torch.cuda.is_available() else "cpu"

# Save path

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def reshape_transform(tensor, height=7, width=7):
    result = tensor[1:, :, :].reshape(height, width, tensor.size(1), tensor.size(2))
    result = result.permute(3, 2, 0, 1)
    return result


# Prepare dataset
target_classes = []
if args.dataset in ['cifar-10-10', 'cifar-10-100-10', 'cifar-10-100-50', 'tinyimagenet']:

    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                 cifar_plus_n=args.out_num)

    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            open_set_classes=args.open_set_classes, balance_open_set_eval=False,
                            split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                            args=args)

    if args.transform == 'rand-augment':
        if args.rand_aug_m is not None:
            if args.rand_aug_n is not None:
                datasets['train'].transform.transforms[0].m = args.rand_aug_m
                datasets['train'].transform.transforms[0].n = args.rand_aug_n
    target_classes = np.array(datasets['train'].classes)[args.train_classes].tolist()
    # for only tinyimagenet dataset:
    if args.dataset == 'tinyimagenet':
        target_class_tmp = []
        words_dict = {}
        word_txt = '../data/tiny-imagenet-200/words.txt'
        words_name = pd.read_csv(word_txt, names=['indexs', 'name'], header=None, sep="\t")
        for i in words_name.index:
            words = words_name.loc[i].values
            words_dict[words[0]] = words[1]

        for i in target_classes:
            target_class_tmp.append(words_dict[i])
        target_classes = target_class_tmp

elif args.dataset == 'imagenet_1k':

    train_transform, test_transform = get_transform(transform_type=args.transform, image_size=args.image_size,
                                                    args=args)

    datasets = get_image_net_datasets(train_transform=train_transform, test_transform=test_transform, seed=args.seed)

    args.train_classes = [i for i in range(100)]
    target_classes = np.array(datasets['train'].classes)[args.train_classes].tolist()


    # 定义文件路径
    synset_path = '/home/wsco/wrokspace/lhy/ILSVRC2012/synset_words.txt'

    # 读取synset映射为字典
    id_to_classname = {}
    with open(synset_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                synset_id, class_name = parts
                id_to_classname[synset_id] = class_name

    # 转换为真实类名
    real_classnames = [id_to_classname.get(cls_id, f'Unknown: {cls_id}') for cls_id in target_classes]
    target_classes = real_classnames




target_classes_tmp = target_classes.copy()
for tar_cls in target_classes_tmp:
    fake_cls = 'fake ' + tar_cls
    target_classes.append(fake_cls)
# target_classes.append('background')
# args.train_classes.append(-1)
args.target_classes = target_classes
print(target_classes)

# Prepare dataloader

dataloaders = {}
for k, v, in datasets.items():
    shuffle = True if k == 'train' else False
    dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                shuffle=shuffle, sampler=None, num_workers=args.num_workers)

trainloader = dataloaders['train']
testloader = dataloaders['val']
outloader = dataloaders['test_unknown']

# Prepare text
text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in target_classes]).to(device)

model_p, preprocess = clip.load(args.backbone, 'cpu')
model_p = dspl_model.CustomCLIP(args, target_classes, model_p)
model_p = model_p.to(device)

for name, param in model_p.named_parameters():
    if "prompt_learner" or 'multi_attention' in name:
        param.requires_grad_(True)
    else:
        param.requires_grad_(False)

# Optimize
optimizer = build_optimizer(
    [{'params': model_p.prompt_learner.parameters()}, {'params': model_p.multi_attention.parameters()}], args)
# optimizer = build_optimizer(model_p, args)
scheduler = build_lr_scheduler(optimizer, args)

# model_p, optimizer = amp.initialize(model_p, optimizer, opt_level="O2")
# Dataparallel:
device_count = torch.cuda.device_count()
if device_count > 1:
    print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
    model_p = nn.DataParallel(model_p)
    # model_n = nn.DataParallel(model_n)

# Train:
# args.eval = True
best_oscr = 0
if args.eval == False:
    log_data = []
    for epoch in range(args.MAX_EPOCH):
        print('epoch:', epoch)
        loss = train_double(model_p, optimizer, trainloader, epoch, args)
        scheduler.step()
        # eval_result = evaluate(testloader, model_p, device, evaluation)
        if (epoch + 1) % 2 == 0:
            # acc = test(model_p, testloader, args)
            result_auc, result_oscr, result_macrof1, acc = test_openset(model_p, testloader, outloader, args)
            if result_oscr > best_oscr:
                best_oscr = result_oscr
                best_auc = result_auc
                best_epoch = epoch
                state = {
                    'net_p': model_p.state_dict(),
                    'epoch': epoch}
                torch.save(state, os.path.join(args.save_path, 'bestsplit_' + str(args.split_idx) + '.pth'))
            print('best oscr', best_oscr, 'best auc', best_auc)
            log_data.append({
                'epoch': epoch,
                'loss': loss,
                'result_auc': result_auc,
                'result_oscr': result_oscr,
                'result_macrof1': result_macrof1,
                'acc': acc
            })
        else:
            log_data.append({
                'epoch': epoch,
                'loss': loss,
            })
        with open(os.path.join(args.save_path, str(args.split_idx) + 'log.json'), 'w') as f:
            json.dump(log_data, f, indent=4)


        # if (epoch + 1) % 100 == 0:
        #     # test(model_p, testloader, args)
        #     # result_auc, result_oscr = test_openset(model_p, testloader, outloader, args)
        #     print('Saving..')
        #     state = {
        #         'net_p': model_p.state_dict(),
        #         'epoch': epoch}
        #     torch.save(state, os.path.join(args.save_path, str(epoch) + 'split_' + str(args.split_idx) + '.pth'))
        if (epoch + 1) % 150 == 0:
            # test(model_p, testloader, args)
            # result_auc, result_oscr, result_macrof1 = test_openset(model_p, testloader, outloader, args)
            print('Saving..')
            state = {
                'net_p': model_p.state_dict(),
                'epoch': epoch}
            torch.save(state, os.path.join(args.save_path, str(epoch) + 'split_' + str(args.split_idx) + '.pth'))
            print('best epoch:', best_epoch)

else:
    checkpoint_p = torch.load(os.path.join(args.save_path, 'bestsplit_' +  str(args.split_idx) + '.pth'))

    # from collections import OrderedDict
    # state_dict = checkpoint_p['net_p']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k.replace("module.", "")  # remove 'module.'
    #     new_state_dict[name] = v
    # model_p.load_state_dict(new_state_dict)

    model_p.load_state_dict(checkpoint_p['net_p'])
    result_auc, result_oscr, result_macrof1, acc = test_openset(model_p, testloader, outloader, args)
    # test(model_p, testloader, args)
    print(result_auc)
    print(result_oscr)
    print(result_macrof1)




import paddle
import argparse
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss
import loader
import builder
from sklearn.metrics.pairwise import cosine_similarity
from main import compute_features, retrieval_precision_cal, accuracy

import resnet

model_names = sorted(name for name in resnet.__dict__ if name.islower() and
    not name.startswith('__') and callable(resnet.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-A', metavar='DIR Domain A', help='path to domain A dataset')
parser.add_argument('--data-B', metavar='DIR Domain B', help='path to domain B dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to clean model (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--low-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--num-cluster', default='250,500,1000', type=str,
                    help='number of clusters for self entropy loss')
parser.add_argument('--cwcon-filterthresh', default=0.2, type=float,
                    help='the threshold of filter for cluster-wise contrastive loss')
parser.add_argument('--selfentro-temp', default=0.2, type=float,
                    help='the temperature for self-entropy loss')
parser.add_argument('--prec-nums', default='1,5,15', type=str,
                    help='the evaluation metric')



def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        paddle.seed(seed=args.seed)
        warnings.warn(
            'You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.'
            )
    args.num_cluster = args.num_cluster.split(',')
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))
    print("=> creating model '{}'".format(args.arch))
    traindirA = os.path.join(args.data_A)
    traindirB = os.path.join(args.data_B)
    eval_dataset = loader.EvalDataset(traindirA, traindirB)
    eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=args
        .batch_size * 2, shuffle=False, num_workers=args.workers)
    model = builder.UCDIR(resnet.__dict__[args.arch], dim=args.low_dim, K_A
        =eval_dataset.domainA_size, K_B=eval_dataset.domainB_size, m=args.
        moco_m, T=args.temperature, mlp=args.mlp, selfentro_temp=args.
        selfentro_temp, num_cluster=args.num_cluster, cwcon_filterthresh=
        args.cwcon_filterthresh)
    paddle.device.set_device(device='gpu:{}'.format(args.gpu))
    model.set_state_dict(paddle.load(args.model))

    features_A, features_B, targets_A, targets_B = compute_features(
        eval_loader, model, args)
    features_A = features_A.numpy()
    targets_A = targets_A.numpy()
    features_B = features_B.numpy()
    targets_B = targets_B.numpy()
    prec_nums = args.prec_nums.split(',')
    res_A, res_B = retrieval_precision_cal(features_A, targets_A,
        features_B, targets_B, preck=(int(prec_nums[0]), int(prec_nums[
        1]), int(prec_nums[2])))
    
    print("Domain A->B: P@{}: {}; P@{}: {}; P@{}: {}".format(int(prec_nums[0]), res_A[0],
                                                                        int(prec_nums[1]), res_A[1],
                                                                        int(prec_nums[2]), res_A[2]))
    print("Domain B->A: P@{}: {}; P@{}: {}; P@{}: {}".format(int(prec_nums[0]), res_B[0],
                                                                        int(prec_nums[1]), res_B[1],
                                                                        int(prec_nums[2]), res_B[2]))


if __name__ == '__main__':
    main()

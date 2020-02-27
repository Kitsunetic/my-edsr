import argparse


parser = argparse.ArgumentParser(description='my edsr tester for raw images')

# normalization
parser.add_argument('--black_lv', type=int, default=512)
parser.add_argument('--white_lv', type=int, default=16383)
parser.add_argument('--patch_size', type=int, default=512)

# train
parser.add_argument('--num_epochs', type=int, default=50)

# dataset
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--shuffle', action='store_true', default=1)
parser.add_argument('--num_workers', type=int, default=4)

# file system
parser.add_argument('--dataset_path', type=str, default='../datasets/SRRAW196.v2')
parser.add_argument('--result_path', type=str, default='../result')

# model
parser.add_argument('--num_resblock', type=int, default=16)
parser.add_argument('--num_channels', type=int, default=32)

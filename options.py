import argparse


parser = argparse.ArgumentParser(description='my edsr tester for raw images')

# raw image normalization
parser.add_argument('--black_lv', type=int, default=512)
parser.add_argument('--white_lv', type=int, default=16383)

# image preprocessing
parser.add_argument('--patch_size', type=int, default=512)

# train
parser.add_argument('--num_epochs', type=int, default=50)

# dataset
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--shuffle', action='store_true', default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--validation_split', type=float, default=0.05)

# file system
parser.add_argument('--dataset_path', type=str, default='../../datasets/SRRAW48')
parser.add_argument('--result_path', type=str, default='./results')

# model
parser.add_argument('--num_resblock', type=int, default=16) # baseline mode size
parser.add_argument('--num_channels', type=int, default=64) # baseline mode size
parser.add_argument('--in_channels', type=int, default=4)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--res_scale', type=int, default=1)
parser.add_argument('--scale', type=int, default=2)

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', type = str, default = './trained_models/RobotLocomotion-v0/design-4/')

args = parser.parse_args()

cmd = 'python train/train.py --rule-sequence 0, 14, 2, 11, 4, 16, 6, 4, 19, 10, 20, 18, 11, 9, 10, 4, 5, 11, 19, 5 --save-dir {}'.format(args.save_dir)

os.system(cmd)



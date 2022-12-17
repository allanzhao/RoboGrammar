import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', type = str, default = './trained_models/RobotLocomotion-v0/design-1/')

args = parser.parse_args()

cmd = 'python train/train.py --rule-sequence 0, 6, 20, 12, 2, 7, 18, 20, 10, 4, 20, 10, 11, 5, 10, 4, 10, 5, 19, 5 --save-dir {}'.format(args.save_dir)

os.system(cmd)



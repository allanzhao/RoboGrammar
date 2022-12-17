import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', type = str, default = './trained_models/RobotLocomotion-v0/design-3/')

args = parser.parse_args()

cmd = 'python train/train.py --rule-sequence 0, 14, 6, 18, 4, 2, 18, 4, 18, 16, 11, 4, 20, 10, 4, 17, 5, 9, 10, 10, 11, 5, 10 --save-dir {}'.format(args.save_dir)

os.system(cmd)



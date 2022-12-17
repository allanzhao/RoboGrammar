import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', type = str, default = './trained_models/RobotLocomotion-v0/design-2/')

args = parser.parse_args()

cmd = 'python train/train.py --rule-sequence 0, 6, 20, 14, 2, 18, 10, 4, 21, 5, 9, 4, 19, 10, 11, 11, 4, 18, 10, 4, 10, 17, 5 --save-dir {}'.format(args.save_dir)

os.system(cmd)



import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type = str, required = True)

args = parser.parse_args()

cmd = 'python train/play.py --rule-sequence 0, 14, 2, 11, 4, 16, 6, 4, 19, 10, 20, 18, 11, 9, 10, 4, 5, 11, 19, 5 --grammar-file ../../data/designs/grammar_jan21.dot --model-path {}'.format(args.model_path)

os.system(cmd)



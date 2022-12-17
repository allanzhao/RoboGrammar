import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type = str, required = True)

args = parser.parse_args()

cmd = 'python train/play.py --rule-sequence 0, 6, 20, 14, 2, 18, 10, 4, 21, 5, 9, 4, 19, 10, 11, 11, 4, 18, 10, 4, 10, 17, 5 --grammar-file ../../data/designs/grammar_jan21.dot --model-path {}'.format(args.model_path)

os.system(cmd)



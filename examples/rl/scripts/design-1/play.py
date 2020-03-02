import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type = str, required = True)

args = parser.parse_args()

cmd = 'python train/play.py --rule-sequence 0, 6, 20, 12, 2, 7, 18, 20, 10, 4, 20, 10, 11, 5, 10, 4, 10, 5, 19, 5 --grammar-file ../../data/designs/grammar_jan21.dot --model-path {}'.format(args.model_path)

os.system(cmd)



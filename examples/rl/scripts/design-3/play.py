import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type = str, required = True)

args = parser.parse_args()

cmd = 'python train/play.py --rule-sequence 0, 14, 6, 18, 4, 2, 18, 4, 18, 16, 11, 4, 20, 10, 4, 17, 5, 9, 10, 10, 11, 5, 10 --grammar-file ../../data/designs/grammar_jan21.dot --model-path {}'.format(args.model_path)

os.system(cmd)



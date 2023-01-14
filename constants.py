import multiprocessing as mp
import numpy as np

GRAMMAR_FILEPATH = "./data/designs/grammar_apr30_asym.dot"
RULE_SEQUENCE = "0, 7, 1, 12, 1, 13, 10, 6, 2, 4, 16, 3, 16, 4, 19, 17, 9, 2, 4, 5, 5, 4, 5, 16, 16, 4, 17, 18, 9, 14, 5, 8, 9, 8, 9, 9, 8, 8".split(" ")
ACTION_FILTER_COEFS = [0.10422766377112629, 0.3239870556899027, 0.3658903830367387, 0.3239870556899027, 0.10422766377112629]
CHANNELS = 16
DT = 16 / 240
HORIZON = 40
NUM_THREADS = mp.cpu_count() - 1
SEED = np.random.randint(10000000)
SAVE_ACTION_SEQUENCE = False
OPTIMIZE = False
INPUT_ACTION_SEQUENCE = None # "output/nice.npy"

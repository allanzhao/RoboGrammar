import os
from time import sleep

import pandas as pd


if __name__ == "__main__":
    command = "python3 examples/design_search/viewer.py FlatTerrainTask data/designs/grammar_apr30_asym.dot {} -j 8 -o --opt_seed {} --save_video_file {} --no_view"
    
    numbers_and_designs_and_seeds = [
        (292, [0, 7, 1, 12, 1, 13, 10, 6, 2, 4, 16, 3, 16, 4, 19, 17, 9, 2, 4, 5, 5, 4, 5, 16, 16, 4, 17, 18, 9, 14, 5, 8, 9, 8, 9, 9, 8, 8], 100000000),
        (165, [0, 10, 10, 1, 2, 19, 16, 6, 8, 4, 17, 7, 4, 14, 9, 9, 5, 2, 16, 5, 14, 9, 4, 18, 8, 5, 5, 9, 9], 2935603211),
        (370, [0, 2, 19, 18, 1, 7, 4, 2, 12, 8, 6, 5, 17, 19, 17, 9, 5, 4, 18, 5, 4, 16, 8, 4, 4, 15, 19, 8, 5, 8, 8, 8, 8, 8], 2795305861),
        (287, [0, 12, 13, 2, 7, 18, 16, 5, 9, 4, 16, 4, 16, 8, 1, 12, 9, 3, 5, 1, 10, 6, 2, 14, 5, 18, 8, 8, 4, 8, 8, 18, 5], 1680518472),
        (195, [0, 12, 2, 17, 11, 9, 9, 18, 5, 1, 5, 7, 13, 2, 16, 17, 6, 4, 19, 8, 4, 14, 4, 19, 5, 8, 9, 5, 8, 8], 1392964597),
        (261, [0, 1, 7, 1, 10, 12, 1, 12, 11, 6, 3, 3, 3, 2, 14, 16, 8, 8, 4, 15, 5, 5, 9], 262357606),
        (162, [0, 7, 1, 2, 15, 9, 6, 4, 12, 18, 5, 14, 5, 8, 8, 3], 3964608891),
        (6, [0, 12, 1, 7, 2, 16, 13, 9, 19, 9, 2, 9, 12, 9, 16, 5, 17, 5, 5, 5, 6], 1112038970),
        (166, [0, 10, 2, 19, 17, 5, 7, 8, 10, 5, 8, 6], 1680603714),
        (423, [0, 12, 10, 1, 11, 6, 7, 3, 2, 4, 17, 14, 4, 16, 19, 9, 9, 8, 4, 5, 5, 18, 9, 8], 3533555167),
        (42, [0, 11, 13, 6, 2, 14, 4, 17, 17, 5, 5, 8, 9, 7, 9], 3134603515)
    ]
    
    for number, design, seed in numbers_and_designs_and_seeds:
        filepath = os.path.join("output", "yellow_subset_sideways_v2", "optim_" + str(number) + ".mp4")
        # print(i, row["rule_seq"], row["reward"])
        os.system(command.format(str(design)[1:-1], seed, filepath))
        sleep(0.1)
    
    # for i in range(0, 64):
    #     filepath = "output/videos_292/" + str(i) + "_optim.mp4"
    #     os.system(command.format(filepath))
    #     sleep(0.1)
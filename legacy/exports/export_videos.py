import os
from time import sleep

import pandas as pd


if __name__ == "__main__":
    command = "python3 examples/design_search/viewer.py FlatTerrainTask data/designs/grammar_apr30_asym.dot {} -j 8 -o --opt_seed {} --save_video_file {}"
    
    designs = pd.read_csv("/home/biobot/BiobotGrammar/trained_models/FlatTerrainTask/07-07-2022-17-56-16/designs.csv")
    designs = designs.sort_values("reward", ascending=False).head(15)
    
    for i, row in designs.iterrows():
        filepath = os.path.join("output", "optim_" + str(i) + ".mp4")
        # print(i, row["rule_seq"], row["reward"])
        os.system(command.format(row["rule_seq"][1:-1], row["opt_seed"], filepath))
        sleep(0.1)
    
    # for i in range(0, 64):
    #     filepath = "output/videos_292/" + str(i) + "_optim.mp4"
    #     os.system(command.format(filepath))
    #     sleep(0.1)
import argparse
import os

import pandas as pd
import json
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs_file", type=str, required=True, help="The designs file used for exporting .obj files")
    parser.add_argument("--output_path", type=str, default="output", help="Where to save the files.")
    parser.add_argument("--save_obj", action="store_true", help="Whether to save the obj file")
    parser.add_argument("--save_render", action="store_true", help="Whether to save the render")
    args = parser.parse_args()
    
    execute_string = "/home/biobot/miniconda3/envs/roboGrammar/bin/python examples/design_search/viewer.py -j 8 -o -l 1 --save_obj_dir \"{}\" --no_view FlatTerrainTask data/designs/grammar_apr30_asym.dot {}"
    
    designs = pd.read_csv(args.designs_file)

    for index, data in designs.iterrows():
        rule_seq = data.rule_seq[1:-1]
        
        output_path = os.path.join(args.output_path, str(index).zfill(5) + "_" + rule_seq)
        os.makedirs(output_path, exist_ok=True)

        print(datetime.now(), "-", index, rule_seq)
        os.system(execute_string.format(output_path, rule_seq))
        
        if args.save_render:
            params_json_path = "pytorch3d-renderer/params.json"
            params = dict()
            params["image_size"] = 1024
            params["camera_dist"] = 1.1
            params["elevation"] = [30]
            params["azim_angle"] = [30]
            params["obj_filename"] = os.path.abspath(os.path.expanduser(os.path.expandvars(os.path.join(output_path, "robot_0000.obj"))))
            params["z_coord"] = 0
            params["out_folder"] = os.path.join("..", output_path)
            
            json.dump(params, open(params_json_path, "w"), indent=4)
            os.system("cd pytorch3d-renderer; /home/biobot/miniconda3/envs/roboGrammar/bin/python render.py")

        if not args.save_obj:
            os.system(f"rm \"{output_path}/robot_0000.obj\"")
            
        os.system(f"rm \"{output_path}/robot.mtl\"")
        os.system(f"rm \"{output_path}/terrain.mtl\"")
        os.system(f"rm \"{output_path}/terrain.obj\"")
    
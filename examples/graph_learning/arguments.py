import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--task', type = str, default = 'FlatTerrainTask', help = 'Task (Python class name')
    parser.add_argument("--grammar-file", type = str, default = '../../data/designs/grammar_jan21.dot', help="Grammar file (.dot)")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("-j", "--mpc-num-processes", type=int, required=True,
                        help="Number of threads for mpc")
    parser.add_argument("-i", "--num-iterations", type=int, required=True,
                        help="Number of iterations")
    parser.add_argument('--lr', type=float, required=True, help='learning rate for dqn')
    parser.add_argument("-d", "--depth", type=int, required=True,
                        help="Maximum tree depth")
    parser.add_argument('--eps-start', type = float, default = 1.0, help = 'starting eps for e-greedy')
    parser.add_argument('--eps-end', type = float, default = 0.0, help = 'ending eps for e-greedy')
    parser.add_argument('--batch-size', type = int, default = 32, help = 'batch size for dqn optimizer')
    parser.add_argument("--save-dir", type=str, default='',
                        help="Log directory")
    parser.add_argument("-f", "--log-file", type=str,
                        help="Existing log file, for resuming a previous run")
    parser.add_argument('--render-interval', type=int)
    
    return parser
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--task', type=str, default='FlatTerrainTask', help='Task (Python class name')
    parser.add_argument("--grammar-file", type=str, default='../../data/designs/grammar_jan21.dot',
                        help="Grammar file (.dot)")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed")
    parser.add_argument("-j", "--mpc-num-processes", type=int, required=True, help="Number of threads for mpc")
    parser.add_argument("-i", "--num-iterations", type=int, required=True, help="Number of iterations")
    parser.add_argument('--lr', type=float, required=True, help='learning rate for dqn')
    parser.add_argument("-d", "--depth", type=int, default=25, help="Maximum tree depth")
    parser.add_argument('--eps-start', type=float, default=1.0, help='starting eps for e-greedy')
    parser.add_argument('--eps-end', type=float, default=0.0, help='ending eps for e-greedy')
    parser.add_argument('--eps-decay', type=float, default=0.5, help='decay factor of eps')
    parser.add_argument('--eps-schedule', type=str, default='exp-decay', help='eps schedule [eps-decay, linear-decay]')
    parser.add_argument('--eps-sample-start', type=float, default=1.0,
                        help='starting eps for sampling multiple designs')
    parser.add_argument('--eps-sample-end', type=float, default=0.0, help='ending eps for sampling multiple designs')
    parser.add_argument('--eps-sample-decay', type=float, default=0.5,
                        help='decay factor of eps for sampling multiple designs')
    parser.add_argument('--eps-sample-schedule', type=str, default='exp-decay',
                        help='eps schedule for sampling multiple designs [eps-decay, linear-decay]')
    parser.add_argument('--num-samples', type=int, default=1, help='number of samples each epoch')
    parser.add_argument('--opt-iter', type=int, default=25, help='number of iterations for optimizer')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for optimizer')
    parser.add_argument('--states-pool-capacity', type=int, default=10000000, help='the maximum size of states pool')
    parser.add_argument('--max-trials', type=int, default=100,
                        help='the max number of trials before determining invalid')
    parser.add_argument('--max-nodes', type=int, default=40,
                        help='the default max nodes of the graph (used to initialize the GNN)')
    parser.add_argument('--no-noise', default=False, action='store_true', help='if remove noise from simulation')
    parser.add_argument('--num-eval', default=1, type=int, help='number of evaluation for each design')

    # parameters for multi-objs case
    parser.add_argument('--task1', type=str, default='FlatTerrainTask', help='Task1 (Python class name')
    parser.add_argument('--task2', type=str, default='FrozenLakeTask', help='Task2 (Python class name')
    parser.add_argument('--weight1', type=float, default=0.5, help='scalarization weight for task1')
    parser.add_argument('--weight2', type=float, default=0.5, help='scalarization weight for task2')

    parser.add_argument('--load-V-path', type=str, default=None, help='the path to load pretrained value function')
    parser.add_argument('--load-Vhat-path', type=str, default=None,
                        help='the path to load pretrained value_target table')
    parser.add_argument('--load-designs-path', type=str, default=None, help='the path to load explored designs')
    parser.add_argument("--save-dir", type=str, default='', help="Log directory")
    parser.add_argument("-f", "--log-file", type=str, help="Existing log file, for resuming a previous run")
    parser.add_argument('--render-interval', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--eval-interval', type=int, default=1000)

    parser.add_argument('--test', default=False, action='store_true')

    # custom
    parser.add_argument('--freq-update', type=int, default=20, help="[DQL] Update Frequency target network")
    parser.add_argument('--batch-norm', type=bool, default=20, help="[DQL] GNN - Batch Normalization")
    parser.add_argument('--layer-size', type=int, default=20, help="[DQL] GNN - Layer Size")
    parser.add_argument('--store-cache', type=bool, default=20, help="[DQL] Store on cache simulated results")

    return parser

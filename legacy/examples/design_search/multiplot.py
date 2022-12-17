import argparse
import ast
from design_search import make_graph, build_normalized_robot
import glob
import itertools
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import pyrobotdesign as rd
import seaborn as sns
import sys
import tasks

def is_negative_objective(obj_name):
    return obj_name == 'ServoCount'

def plot_iterations(df, ind_rewards, estimator=None, **kwargs):
    df['algorithm'].replace({'hs': "GHS", 'mcts': "MCTS"}, inplace=True)
    df['reward_max'] = df.groupby(['task', 'algorithm', 'trial'])['reward'].cummax()

    fig, ax = plt.subplots()
    if ind_rewards:
        sns.scatterplot(x='iteration', y='reward', hue='algorithm', data=df,
                        ax=ax, alpha=0.2, legend=False)
    units = None if estimator else 'trial'
    sns.lineplot(x='iteration', y='reward_max', hue='algorithm', units=units,
                 data=df, ax=ax, estimator=estimator, ci=100)
    tasks = df['task'].unique()
    title = tasks[0] if len(tasks) == 1 else None
    ax.set(title=title, xlabel='iteration', ylabel='reward')
    fig.tight_layout()
    plt.show()

def plot_pareto(df, **kwargs):
    tasks = df['task'].unique()
    if len(tasks) < 2:
        raise ValueError("At least two tasks are required to plot Pareto sets")

    # Keep only the highest reward for each (task, hash) combination
    df = df.loc[df.groupby(['task', 'hash'])['reward'].idxmax()]

    for task0, task1 in itertools.product(tasks, tasks):
        if task0 >= task1:
            continue

        df_task0 = df[df['task'] == task0]
        df_task1 = df[df['task'] == task1]
        df_merged = df_task0.merge(df_task1, on='hash', validate='one_to_one')

        df_merged.sort_values('reward_x',
                              ascending=is_negative_objective(task0),
                              inplace=True)
        if is_negative_objective(task1):
            df_pareto = df_merged[df_merged['reward_y'] < df_merged['reward_y'].shift(1, fill_value=float('inf')).cummin()]
        else:
            df_pareto = df_merged[df_merged['reward_y'] > df_merged['reward_y'].shift(1, fill_value=float('-inf')).cummax()]

        fig, ax = plt.subplots()
        sns.scatterplot(x='reward_x', y='reward_y', data=df_merged, alpha=0.2)
        sns.scatterplot(x='reward_x', y='reward_y', data=df_pareto)
        ax.set(xlabel=task0, ylabel=task1)

        for _, row in df_pareto.iterrows():
            print(row['rule_seq_x'])

    fig.tight_layout()
    plt.show()

def main():
    sns.set_context('paper')

    parser = argparse.ArgumentParser(
        description="Create plots using multiple log directories.")
    parser.add_argument('log_dir', type=str, nargs='+',
                        help="Log directory containing meta.json")
    parser.add_argument('-t', '--task', type=str, nargs='+',
                        help="Task to include in plots")
    parser.add_argument('-a', '--algorithm', type=str, nargs='+',
                        help="Task to include in plots")
    parser.add_argument('-i', '--iterations', type=int,
                        help="Maximum number of iterations to show")
    parser.add_argument('--servo_count', action='store_true',
                        help="Include servo count as an objective")
    parser.add_argument('--ind_rewards', action='store_true',
                        help="Include individual rewards in iterations plot")
    parser.add_argument('--estimator', type=str,
                        help="Estimator for aggregating multiple trials")
    subparsers = parser.add_subparsers(help='Plot type')
    parser_iterations = subparsers.add_parser('iterations')
    parser_iterations.set_defaults(func=plot_iterations)
    parser_pareto = subparsers.add_parser('pareto')
    parser_pareto.set_defaults(func=plot_pareto)

    args = parser.parse_args()

    # Store every log file's contents into one big pandas dataframe
    df = pd.DataFrame()

    for log_dir in args.log_dir:
        try:
            with open(os.path.join(log_dir, 'meta.json'), 'r') as json_file:
                metadata = json.load(json_file)
        except FileNotFoundError:
            print("Directory '{}' does not contain metadata file, skipping".format(log_dir),
                  file=sys.stderr)
            continue

        # Load the .csv data
        csv_file_names = glob.glob(os.path.join(log_dir, '*.csv'))
        if len(csv_file_names) == 0:
            print("Directory '{}' does not contain any .csv files, skipping".format(log_dir), file=sys.stderr)
            continue

        for trial_num, csv_file_name in enumerate(csv_file_names):
            try:
                log_df = pd.read_csv(csv_file_name)
            except FileNotFoundError:
                print("File '{}' does not exist, skipping".format(csv_file_name), file=sys.stderr)
                continue

            if 'iteration' not in log_df.columns:
                log_df['iteration'] = log_df.index

            log_df.rename(columns={'result': 'reward'}, inplace=True)

            if 'task' not in log_df.columns:
                log_df['task'] = metadata.get('task')

            if 'grammar' not in log_df.columns:
                log_df['grammar'] = metadata.get(
                    'grammar', 'data/designs/grammar_apr30.dot')

            if 'algorithm' not in log_df.columns:
                log_df['algorithm'] = metadata.get('algorithm')

            log_df['trial'] = trial_num

            df = df.append(log_df, ignore_index=True, sort=True)

    # Filter data based on arguments
    if args.iterations:
        df = df[df['iteration'] < args.iterations]
    if args.task:
        df = df[df['task'].isin(args.task)]
    if args.algorithm:
        df = df[df['algorithm'].isin(args.algorithm)]

    try:
        # Expecting only one grammar
        grammar_file, = df['grammar'].unique()
    except ValueError:
        print("All runs must use the same grammar", file=sys.stderr)
        raise

    graphs = rd.load_graphs(grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    # Compute a graph hash, servo count for each rule_seq
    rule_seq_hashes = {}
    rule_seq_servo_counts = {}
    for rule_seq_str in df['rule_seq'].unique():
        rule_seq = ast.literal_eval(rule_seq_str)
        graph = make_graph(rules, rule_seq)
        robot = build_normalized_robot(graph)

        rule_seq_hashes[rule_seq_str] = hash(graph)

        servo_count = 0
        for link in robot.links:
            if link.joint_type == rd.JointType.HINGE:
                # Only hinge joints have servos
                servo_count += 1
        rule_seq_servo_counts[rule_seq_str] = servo_count

    if args.servo_count:
        servo_count_df = pd.DataFrame({'rule_seq': df['rule_seq'].unique()})
        servo_count_df['task'] = 'ServoCount'
        servo_count_df['reward'] = \
            servo_count_df['rule_seq'].map(rule_seq_servo_counts)
        df = df.append(servo_count_df, ignore_index=True, sort=True)

    df['hash'] = df['rule_seq'].map(rule_seq_hashes)

    args.func(df, ind_rewards=args.ind_rewards, estimator=args.estimator)

if __name__ == '__main__':
  main()

import argparse
import ast
from design_search import make_graph, build_normalized_robot
import glob
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

def plot_iterations(ax, df):
    df['reward_max'] = df.groupby(['task', 'algorithm'])['reward'].cummax()

    sns.scatterplot(x='iteration', y='reward', hue='algorithm', data=df,
                    ax=ax, alpha=0.2)
    sns.lineplot(x='iteration', y='reward_max', hue='algorithm', data=df, ax=ax,
                 legend=False)
    ax.set(xlabel='iteration', ylabel='reward')

def plot_pareto(ax, df):
    tasks = df['task'].unique()
    if len(tasks) != 2:
        raise ValueError("Only two tasks are supported")

    # Keep only the highest reward for each (task, hash) combination
    df = df.loc[df.groupby(['task', 'hash'])['reward'].idxmax()]

    df_task0 = df[df['task'] == tasks[0]]
    df_task1 = df[df['task'] == tasks[1]]
    df = df_task0.merge(df_task1, on='hash', validate='one_to_one')

    df.sort_values('reward_x', ascending=is_negative_objective(tasks[0]),
                   inplace=True)
    if is_negative_objective(tasks[1]):
        pareto_df = df[df['reward_y'] < df['reward_y'].shift(1, fill_value=float('inf')).cummin()]
    else:
        pareto_df = df[df['reward_y'] > df['reward_y'].shift(1, fill_value=float('-inf')).cummax()]

    sns.scatterplot(x='reward_x', y='reward_y', data=df, alpha=0.2)
    sns.scatterplot(x='reward_x', y='reward_y', data=pareto_df)
    ax.set(xlabel=tasks[0], ylabel=tasks[1])

    for _, row in pareto_df.iterrows():
        print(row['rule_seq_x'])

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
        if len(csv_file_names) != 1:
            print("Directory '{}' does not contain exactly one .csv file, skipping", file=sys.stderr)
        log_df = pd.read_csv(csv_file_names[0])

        if 'iteration' not in log_df.columns:
            log_df['iteration'] = log_df.index

        log_df.rename(columns={'result': 'reward'}, inplace=True)

        log_df['task'] = metadata.get('task')
        log_df['grammar'] = metadata.get('grammar',
                                         'data/designs/grammar_apr30.dot')
        log_df['algorithm'] = metadata.get('algorithm')

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

    fig, ax = plt.subplots()
    args.func(ax, df)
    plt.show()

if __name__ == '__main__':
  main()

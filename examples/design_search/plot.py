import argparse
import ast
from design_search import make_graph, build_normalized_robot
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import pandas as pd
import pyrobotdesign as rd
from results import get_robot_image
import seaborn as sns
import tasks

def main():
  parser = argparse.ArgumentParser(description="Plot results of a search run.")
  parser.add_argument("task", type=str, help="Task (Python class name)")
  parser.add_argument("grammar_file", type=str, help="Grammar file (.dot)")
  parser.add_argument('log_file', type=str, help="Log file (.csv)")
  parser.add_argument('--max_iters', type=int,
                      help="Maximum number of iterations to show")
  subparsers = parser.add_subparsers(dest='subcommand',
                                     help="Plotting subcommand")

  iter_scatter_parser = subparsers.add_parser(
      'iter_scatter', help="Scatter plot of results vs. iteration")
  iter_scatter_parser.add_argument('--image_count', type=int, default=0,
                                   help='Number of design images to show')
  iter_scatter_parser.add_argument('--spacing', type=int, default=1000, help='Minimum spacing between chosen designs, in iterations')

  args = parser.parse_args()

  task_class = getattr(tasks, args.task)
  task = task_class()
  graphs = rd.load_graphs(args.grammar_file)
  rules = [rd.create_rule_from_graph(g) for g in graphs]
  log_df = pd.read_csv(args.log_file)
  if args.max_iters:
    log_df = log_df[log_df['iteration'] < args.max_iters]

  if args.subcommand == 'iter_scatter':
    if args.image_count > 0:
      grid = plt.GridSpec(2, args.image_count, wspace=0, hspace=0,
                          height_ratios=(0.2, 0.8))
      scatter_ax = plt.subplot(grid[1,:])

      # Select the top `image_count` designs, spaced at least `spacing` apart
      best_indices = []
      log_df_remaining = log_df
      for i in range(args.image_count):
        best_idx = log_df_remaining['result'].idxmax()
        best_indices.append(best_idx)
        log_df_remaining = log_df_remaining[
            abs(log_df_remaining.index - best_idx) > args.spacing]
      best_indices.sort()
      print(best_indices)

      for j, best_idx in enumerate(best_indices):
        rule_seq = ast.literal_eval(log_df['rule_seq'][best_idx])
        graph = make_graph(rules, rule_seq)
        robot = build_normalized_robot(graph)
        image = get_robot_image(robot, task)
        image_ax = plt.subplot(grid[0, j])
        plt.axis('off')
        plt.imshow(image, origin='lower')
        patch = ConnectionPatch(xyA=(0.5 * image.shape[1], 0.0),
                                xyB=(log_df['iteration'][best_idx],
                                     log_df['result'][best_idx]),
                                coordsA='data', coordsB='data', 
                                axesA=image_ax, axesB=scatter_ax,
                                color='darkgray')
        image_ax.add_artist(patch)
    else:
      fig, scatter_ax = plt.subplots()

    sns.scatterplot(x='iteration', y='result', data=log_df, ci=None, marker='.',
                    linewidth=0, ax=scatter_ax)
    plt.tight_layout()
    plt.savefig('iter_scatter.pdf')

if __name__ == '__main__':
  main()

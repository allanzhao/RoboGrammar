import argparse
import ast
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns

grammar_names = OrderedDict([
    ('grammar_apr30', 'standard'),
    ('grammar_apr30_simple', 'simple'),
    ('grammar_apr30_asym', 'asymmetric')
])

def moving_average_filter(x, window_size):
    sums = np.cumsum(x) - np.cumsum(np.pad(x, window_size, 'constant'))[:len(x)]
    lens = np.minimum(np.r_[:len(x)] + 1, window_size)
    return sums / lens

# Modified version of a seaborn function
def my_aggregate(self, vals, grouper, units=None):
    """Compute an estimate and confidence interval using grouper."""
    func = self.estimator
    ci = self.ci
    n_boot = self.n_boot
    seed = self.seed if hasattr(self, "seed") else None

    # Define a "null" CI for when we only have one value
    null_ci = pd.Series(index=["low", "high"], dtype=np.float)

    # Function to bootstrap in the context of a pandas group by
    def bootstrapped_cis(vals):

        if len(vals) <= 1:
            return null_ci

        boots = sns.algorithms.bootstrap(vals, func=func, n_boot=n_boot, seed=seed)
        cis = sns.utils.ci(boots, ci)
        return pd.Series(cis, ["low", "high"])

    # Group and get the aggregation estimate
    grouped = vals.groupby(grouper, sort=self.sort)
    est = grouped.agg(func)

    # Exit early if we don't want a confidence interval
    if ci is None:
        return est.index, est, None

    # Compute the error bar extents
    if callable(ci):
        cis = pd.DataFrame(ci(vals),
                           index=est.index,
                           columns=["low", "high"]).stack()
    elif ci == "sd":
        sd = grouped.std()
        cis = pd.DataFrame(np.c_[est - sd, est + sd],
                           index=est.index,
                           columns=["low", "high"]).stack()
    else:
        cis = grouped.apply(bootstrapped_cis)

    # Unpack the CIs into "wide" format for plotting
    if cis.notnull().any():
        cis = cis.unstack().reindex(est.index)
    else:
        cis = None

    return est.index, est, cis

def range_ci(vals):
    return np.c_[vals.min(level=0).to_numpy(), vals.max(level=0).to_numpy()]

def main():
    window_size = 100

    sns.set_context('paper')
    plt.rc('font', family='serif', size=15)

    # Patch sns._LinePlotter to allow custom confidence intervals
    sns.relational._LinePlotter.aggregate = my_aggregate

    parser = argparse.ArgumentParser()
    parser.add_argument('log_dirs', type=str, nargs='+')
    args = parser.parse_args()

    trial_args_parser = argparse.ArgumentParser()
    trial_args_parser.add_argument('--grammar-file', type=str, required=True)
    trial_args_parser.add_argument('--seed', '-s', type=int, default=0)

    log_dfs = []

    for log_dir in args.log_dirs:
        with open(path.join(log_dir, 'args.txt')) as args_file:
            trial_args, _ = trial_args_parser.parse_known_args(ast.literal_eval(args_file.read()))

        with open(path.join(log_dir, 'log.txt')) as log_file:
            rows = []

            for line in log_file:
                pairs = [list(map(str.strip, p.split('='))) for p in line.split(',')]
                row = dict((pair[0], float(pair[1])) for pair in pairs)
                rows.append(row)

            log_df = pd.DataFrame(rows)
            log_df['grammar'] = grammar_names[path.splitext(path.basename(trial_args.grammar_file))[0]]
            log_df['seed'] = trial_args.seed
            log_df['epoch'] = log_df.index
            log_df['prediction_error'] = np.abs(log_df['predicted_reward'] - log_df['reward'])
            log_df['loss_smoothed'] = moving_average_filter(log_df['loss'], window_size)
            log_df['prediction_error_smoothed'] = moving_average_filter(log_df['prediction_error'], 100)
            log_df['best_reward'] = np.maximum.accumulate(log_df['reward'])
            log_dfs.append(log_df)

    def grammar_ordering(log_df):
        return list(grammar_names.values()).index(log_df['grammar'][0])

    df = pd.concat(sorted(log_dfs, key=grammar_ordering))

    fig, ax = plt.subplots(1, 3, figsize=(8, 3))

    sns.lineplot(x='epoch', y='loss', hue='grammar', data=df, ax=ax[0], ci=range_ci)
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('loss')
    ax[0].set_ylim(bottom=0.0, top=1.0)

    sns.lineplot(x='epoch', y='prediction_error_smoothed', hue='grammar', data=df, ax=ax[1], ci=range_ci, legend=False)
    ax[1].set_title('Average Prediction Error')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('error')

    #sns.scatterplot(x='epoch', y='reward', hue='grammar', data=df, ax=ax[2], s=5, alpha=0.3, legend=False)
    sns.lineplot(x='epoch', y='best_reward', hue='grammar', data=df, ax=ax[2], ci=range_ci, legend=False)
    ax[2].set_title('Best Reward')
    ax[2].set_xlabel('iteration')
    ax[2].set_ylabel('reward')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

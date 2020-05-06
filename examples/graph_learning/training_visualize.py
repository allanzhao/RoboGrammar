import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse(data_str):
    data_items = data_str.replace(' ', '').split(',')
    data = dict()
    for item in data_items:
        key, value = item.split('=')
        data[key] = float(value)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', type = str, required = True)
    parser.add_argument('--log-path-compare', type = str, default = None)

    args = parser.parse_args()
    log_path = args.log_path

    fp = open(log_path, 'r')
    datalines = fp.readlines()
    fp.close()

    print('total {} lines'.format(len(datalines)))

    eps, loss, epoch_reward, avg_reward, avg_window_reward, best_reward, prediction_error, avg_window_pred_error = [], [], [], [], [], [], [], []
    for dataline in datalines:
        data = parse(dataline)
        eps.append(data['eps'])
        loss.append(data['loss'])
        epoch_reward.append(data['reward'])
        avg_reward.append(data['avg_reward'])
        prediction_error.append((data['predicted_reward'] - data['reward']) ** 2)
        if len(best_reward) == 0:
            best_reward.append(data['reward'])
        else:
            best_reward.append(max(best_reward[-1], data['reward']))

    loss_smooth = []
    for i in range(len(loss)):
        his_len = min(i + 1, 30)
        loss_sum = np.sum(loss[i - his_len + 1:i + 1])
        loss_smooth.append(loss_sum / his_len)

    for i in range(len(eps)):
        his_len = min(i + 1, 10000)
        reward_sum = np.sum(epoch_reward[i - his_len + 1:i + 1])
        avg_reward[i] = reward_sum / his_len

    window_size = 100
    for i in range(len(eps)):
        his_len = min(i + 1, window_size)
        reward_sum = np.sum(epoch_reward[i - his_len + 1:i + 1])
        avg_window_reward.append(reward_sum / his_len)
        pred_error_sum = np.sum(prediction_error[i - his_len + 1:i + 1])
        avg_window_pred_error.append(pred_error_sum / his_len)
    
    fig, ax = plt.subplots(1, 5, figsize = (25, 5))

    epoch = list(range(0, len(eps)))
    
    # plot eps
    ax[0].plot(epoch, eps, c = 'tab:blue')
    ax[0].set_title('eps')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('eps')

    # plot loss
    ax[1].plot(epoch, loss, c = 'tab:red')
    ax[1].set_title('Training Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('training loss')

    # plot avg_window_pred_error
    ax[2].plot(epoch, avg_window_pred_error, c = 'tab:green', zorder = 10)
    ax[2].set_title('Avg Pred Err over {} epochs'.format(window_size))
    ax[2].set_xlabel('epoch')

    # plot avg_window_reward
    ax[3].plot(epoch, avg_window_reward, c = 'tab:green', zorder = 10)
    ax[3].set_title('Avg Reward over {} epochs'.format(window_size))
    ax[3].set_xlabel('epoch')
    
    # plot epoch reward and best reward
    ax[4].scatter(epoch, epoch_reward, c = 'tab:blue', s = 5, alpha = 0.2, zorder = 10)
    ax[4].plot(epoch, best_reward, c = 'tab:green')
    ax[4].set_title('Epoch Reward & Best Reward')
    ax[4].set_xlabel('epoch')

    # for compare
    if args.log_path_compare is not None:
        log_path_compare = args.log_path_compare

        fp = open(log_path_compare, 'r')
        datalines = fp.readlines()
        fp.close()

        print('total {} lines'.format(len(datalines)))

        eps, loss, epoch_reward, avg_reward, avg_window_reward, best_reward, prediction_error, avg_window_pred_error = [], [], [], [], [], [], [], []
        for dataline in datalines:
            data = parse(dataline)
            eps.append(data['eps'])
            loss.append(data['loss'])
            epoch_reward.append(data['reward'])
            avg_reward.append(data['avg_reward'])
            prediction_error.append((data['predicted_reward'] - data['reward']) ** 2)
            if len(best_reward) == 0:
                best_reward.append(data['reward'])
            else:
                best_reward.append(max(best_reward[-1], data['reward']))

        loss_smooth = []
        for i in range(len(loss)):
            his_len = min(i + 1, 30)
            loss_sum = np.sum(loss[i - his_len + 1:i + 1])
            loss_smooth.append(loss_sum / his_len)

        for i in range(len(eps)):
            his_len = min(i + 1, 10000)
            reward_sum = np.sum(epoch_reward[i - his_len + 1:i + 1])
            avg_reward[i] = reward_sum / his_len

        for i in range(len(eps)):
            his_len = min(i + 1, window_size)
            reward_sum = np.sum(epoch_reward[i - his_len + 1:i + 1])
            avg_window_reward.append(reward_sum / his_len)
            pred_error_sum = np.sum(prediction_error[i - his_len + 1:i + 1])
            avg_window_pred_error.append(pred_error_sum / his_len)
            
        epoch = list(range(0, len(eps)))

        ax[2].plot(epoch, avg_window_pred_error, c = 'tab:red')
        ax[3].plot(epoch, avg_window_reward, c = 'tab:red')
        ax[4].plot(epoch, best_reward, '--', c = 'tab:red')
        ax[4].scatter(epoch, epoch_reward, c = 'tab:purple', s = 5, alpha = 0.2)
    
    # ax[2].set_ylim(ax[3].get_ylim())

    plt.show()



import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


def visualize_path(domain, paths, pred_traj, correct):
    fig, ax = plt.subplots()
    ax.imshow(domain.T, cmap="Greys")
    ax.plot(paths[:, 0], paths[:, 1], c='b',
            label='Optimal Path')
    ax.plot(pred_traj[:, 0], pred_traj[:, 1],
            '-X', c='r', label='Predicted Path')
    ax.plot(paths[0, 0], paths[0, 1], '-o', label='Start')
    ax.plot(paths[-1, 0], paths[-1, 1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small')   # The legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)        # The legend line width
    fig.savefig(
        '../image/path/path{}.pdf'.format(int(correct)))
    plt.close()


def visualize_reward(reward_image, correct):
    fig, ax = plt.subplots()
    reward_image = (reward_image - reward_image.min()) / (reward_image.max() - reward_image.min())
    reward_map = ax.imshow(reward_image.T, cmap=cm.cool)
    fig.colorbar(reward_map, ax=ax)
    fig.savefig('../image/reward/reward{}.pdf'.format(int(correct)))
    plt.close()


def visualize_v_value(v_value_image, correct):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter('../video/v_value{}.mp4'.format(
        int(correct)), fourcc, 20.0, (560, 480))

    for i in range(v_value_image.shape[0]):
        fig, ax = plt.subplots()
        v_value_map = ax.imshow(
            v_value_image[i].T, cmap=cm.cool)
        fig.colorbar(v_value_map, ax=ax)
        fig.savefig('../image/v_value/v_value{}_{}.png'.format(int(correct), i))
        plt.close()

        img = cv2.imread(
            '../image/v_value/v_value{}_{}.png'.format(int(correct), i))
        img = cv2.resize(img, (560, 480))
        video.write(img)
    video.release()

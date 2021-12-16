import matplotlib.pyplot as plt
from IPython import display
import os

plt.ion()

def plot(scores, mean_scores, mini_mean_scores, title):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('{}'.format(title))
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(mini_mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(mini_mean_scores)-1, mini_mean_scores[-1], str(mini_mean_scores[-1]))
    plt.show(block=False)

    folder_path = './{}'.format(title)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig('{}/{}-{} result.png'.format(folder_path,title, len(scores)))

    f = open('{}/report.txt'.format(folder_path), 'w')
    f.write('scores:\n{}\nmean scores:\n{}\nmini mean scores:\n{}'.format(scores, mean_scores, mini_mean_scores))
    f.close()
    # plt.pause(.1)

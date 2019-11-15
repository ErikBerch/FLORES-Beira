"""


"""

import seaborn as sns
import matplotlib as mpl

sns.set_style('whitegrid')
mpl.rcParams['figure.figsize'] = (12, 8)


def save_fig(fig, dir, name):
    """save a high res and a low res version of the figure in the specified
    directory, using the label i.

    Parameters
    ----------
    fig : a Figure instance
    dir : str
          the directory where figures are to be saved
    name : str

    """

#    fig.savefig('{}/fig{}_lowres.png'.format(dir, name), dpi=75,
#                bbox_inches='tight', format='png')
    fig.savefig('{}/fig{}_highres.png'.format(dir, name), dpi=300,
                bbox_inches='tight', format='png')


def change_fontsize(fig, fs=14):
    """Change fontsize of figure items to specified size"""
    # TODO:: add legend and general text items

    for ax in fig.axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs)

        try:
            parasites = ax.parasites
        except AttributeError:
            pass
        else:
            for parisite in parasites:
                for axis in parisite.axis.values():
                    axis.major_ticklabels.set_fontsize(fs)
                    axis.label.set_fontsize(fs)

            for axis in ax.axis.values():
                axis.major_ticklabels.set_fontsize(fs)
                axis.label.set_fontsize(fs)

        if ax.legend_ is not None:
            for entry in ax.legend_.get_texts():
                entry.set_fontsize(fs)


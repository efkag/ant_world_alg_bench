import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
sns.set_context("paper", font_scale=1)

directory = '2022-03-17'
fig_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(fig_save_path, 'results.csv'), index_col=False)
# data['trial_fail_count'] = data['trial_fail_count'].apply(literal_eval)

figsize = (5, 3)
fig, ax = plt.subplots(figsize=figsize)
sns.barplot(x="window", y="trial_fail_count", data=data, ax=ax, estimator=sum, capsize=.2)
window_labels = ['Adaptive SMW', 'PM', 'Fixed 15']
ax.set_xticklabels(window_labels)
plt.tight_layout()
path = os.path.join(fig_save_path, 'failed trials.png')
fig.savefig(path)
plt.show()


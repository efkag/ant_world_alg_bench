import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source2 import load_route_naw, plot_route

sns.set_context("paper", font_scale=1)


fig_save_path = '/home/efkag/Desktop/route'
# data = pd.read_csv('combined-results.csv')
data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 4
fig_save_path = fig_save_path + str(route_id)
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = 15
matcher = 'corr'
edge = '(220, 240)'
res = '(180, 50)'
threshold = 30
figsize = (4, 4)

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
traj['window_log'] = literal_eval(traj['window_log'])

route = load_route_naw(path, route_id=route_id)
plot_route(route, traj, size=(6, 6), save=False, path=fig_save_path)

plt.plot(range(len(traj['abs_index_diff'])), traj['abs_index_diff'])
plt.show()

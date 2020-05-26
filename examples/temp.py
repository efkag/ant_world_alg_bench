import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Load the example tips dataset
tips = sns.load_dataset("tips")
x = np.random.choice(5, 400)
y = np.random.rand(400)
hue = np.random.choice(2, 400)


sns.violinplot(x=x, y=y, hue=hue, split=True, inner="quart")
plt.show()

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="day", y="total_bill", hue="smoker",
               split=True, inner="quart",
               palette={"Yes": "y", "No": "b"},
               data=tips)
plt.show()
sns.despine(left=True)
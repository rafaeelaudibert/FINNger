import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# Accuracy obtained when running 100 epochs
ACCURACY = [
    17, 35, 43, 52, 54, 64, 65, 70, 78, 67, 70, 76, 76, 80, 78, 83, 81, 81, 86, 83, 81, 86, 84, 83, 84,
    89, 87, 90, 89, 89, 89, 87, 90, 91, 90, 91, 89, 87, 94, 90, 94, 93, 92, 91, 92, 92, 91, 91, 92, 93,
    92, 93, 94, 95, 92, 94, 91, 93, 94, 90, 92, 95, 95, 97, 95, 93, 94, 95, 93, 94, 94, 97, 96, 95, 96,
    96, 96, 96, 96, 98, 96, 96, 96, 95, 96, 94, 96, 96, 95, 95, 95, 96, 94, 96, 96, 97, 96, 96, 97, 96, 95
]
polynomial_parameters = np.polyfit(range(len(ACCURACY)), ACCURACY, 10)
poly_fit = np.poly1d(polynomial_parameters)

sns.set_palette("husl")
sns.lineplot(x=range(len(ACCURACY)), y=[
             poly_fit(x) for x in range(len(ACCURACY))])
sns.lineplot(x=range(len(ACCURACY)), y=ACCURACY, lw=0.8)
plt.title("Accuracy over epochs")
plt.show()

# Correlation plots
data = defaultdict(list)
with open("data/input.txt") as f:
    for line in tqdm(f.readlines()):
        expected, result = line.split(" ")
        data[int(expected)].append(int(result))

corr = [[0] * 6 for _ in range(6)]
for opt in data:
    input = data[opt]

    total = len(input)
    counter = Counter(input)
    for i in range(6):
        corr[opt][i] = counter[i] / total
    print(counter, opt)

f, ax = plt.subplots(figsize=(10, 8))
sns.set_theme()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot=True)
plt.title("Correlation for NN")
plt.show()

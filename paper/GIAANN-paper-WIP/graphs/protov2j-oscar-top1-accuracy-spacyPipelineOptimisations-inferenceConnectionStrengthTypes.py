import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

x = np.array([0, 134750, 5484495, 27167113, 54306502, 81375876])
y1 = np.array([0, 0.074, 0.192, 0.237, 0.253, 0.262])
y2 = np.array([1.0, 1.0, 0.992, 0.976, 0.961, 0.949])

l1, = plt.plot(x, y2, color='red', label='train-set eval (c seg=3, f seg=5).\ninference seed=8, segment timings=exact, \nactivations=bool, connection strengths=bool.')
l2, = plt.plot(x, y1, color='blue', label='test-set eval (c seg=3, f seg=5).\ninference seed=8, segment timings=none, \nactivations=bool, connection strengths=float.')

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(10_000_000))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:g}M"))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Four minor ticks per 10M interval
plt.yticks(np.arange(0, 1.0+0.1, 0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

plt.xlabel("number of o200k_base tokens")
plt.ylabel("OSCAR-2201 train/test-set accuracy (Top-1)")
plt.title("GIAANN Proto 2j with subword tokeniser")

plt.legend(handles=[l1, l2])

plt.savefig("protov2j-oscar-top1-accuracy-spacyPipelineOptimisations-inferenceConnectionStrengthTypes.pdf", format="pdf", bbox_inches="tight")
plt.show()

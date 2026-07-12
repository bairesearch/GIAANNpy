import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

x = np.array([0, 134750, 5484495, 27167113, 54306502, 81375876])
y1 = np.array([1.0, 1.0, 0.992, 0.976, 0.961, 0.949])

l1, = plt.plot(x, y1, color='red', label='train c seg=3, f seg=5.\ninference seed=8, segment timings=exact, \nactivations=bool, connection strengths=bool.')

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(10_000_000))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:g}M"))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Four minor ticks per 10M interval
plt.yticks(np.arange(0.5, 1.0+0.1, 0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

plt.xlabel("number of o200k_base tokens")
plt.ylabel("OSCAR-2201 training-set accuracy (Top-1)")
plt.title("GIAANN Proto 2j with subword tokeniser")

plt.legend(handles=[l1], loc='center right')

plt.savefig("protov2j-oscar-train-top1-accuracy-spacyPipelineOptimisations-inferenceConnectionStrengthTypes.pdf", format="pdf", bbox_inches="tight")
plt.show()

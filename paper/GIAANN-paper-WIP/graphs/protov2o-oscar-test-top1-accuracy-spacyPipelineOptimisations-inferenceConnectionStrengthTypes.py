import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

x = np.array([0, 328518, 13238046, 66168928])
y1 = np.array([0, 0.097, 0.189, 0.217])

l1, = plt.plot(x, y1, color='blue', label='train c seg=3, f seg=5.\ninference seed=8.')

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(10_000_000))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:g}M"))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Four minor ticks per 10M interval
plt.yticks(np.arange(0, 0.5+0.1, 0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

plt.xlabel("number of o200k_base tokens")
plt.ylabel("OSCAR-2201 test-set accuracy (Top-1)")
plt.title("GIAANN Proto 2o with subword tokeniser")

plt.legend(handles=[l1])

plt.savefig("protov2o-oscar-test-top1-accuracy-spacyPipelineOptimisations-inferenceConnectionStrengthTypes.pdf", format="pdf", bbox_inches="tight")
plt.show()

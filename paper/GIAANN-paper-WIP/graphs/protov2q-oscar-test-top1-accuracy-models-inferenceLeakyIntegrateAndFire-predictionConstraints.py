import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

x4 = np.array([0, 3778117, 37478755, 150184342, 300788029, 601010669])
y4 = np.array([0, 0.1719163840689126, 0.27905393697863808, 0.33018453427065025, 0.38758075694942945, 0.4093758508873982])
x5 = np.array([0, 3778117, 37478755, 150184342, 300788029, 601010669])
y5 = np.array([0, 0.21351085422906507, 0.306882596103864, 0.348994, 0.40220054951855246, 0.42044666947201664])

x6 = np.array([0, 3356675, 33596809, 133951027, 268177316, 537020923])
y6 = np.array([0, 0.14418154878020287, 0.2946896313548088, 0.365966238629818, 0.3884571637511253, 0.4074458841085434])

l4, = plt.plot(x4, y4, color='blue', label='test-set GIAANN.\ntrain c seg=4, f seg=4. inference seed=8.')	#direct connectivity enforced
l5, = plt.plot(x5, y5, color='darkblue', label='test-set GIAANN direct connectivity relaxed.\ntrain c seg=4, f seg=4. inference seed=8.')
l6, = plt.plot(x6, y6, color='cyan', label='test-set RoBERTa.\n6-layers, 512-token context window.')

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(100_000_000))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:g}M"))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Four minor ticks per 10M interval
plt.yticks(np.arange(0, 1.0+0.1, 0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

plt.xlabel("number of GIAANN o200k_base or RoBERTa byte-level BPE tokens")
plt.ylabel("OSCAR-2201 test-set accuracy (Top-1)")
plt.title("GIAANN Proto 2q with subword tokeniser")

plt.legend(handles=[l4, l5, l6], loc='center right', bbox_to_anchor=(1.0, 0.65))

plt.savefig("protov2q-oscar-test-top1-accuracy-models-inferenceLeakyIntegrateAndFire-predictionConstraints.pdf", format="pdf", bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

x1 = np.array([1.0, 3778117, 37478755, 150184342])
y1 = np.array([1.0, 0.9825611406409129, 0.9658393671786716, 0.947672261052749])
#x2 = np.array([1.0, 3778117, 37478755, 150184342])
#y2 = np.array([1.0, 0.9825750917284002, 0.9658505280486613, 0.9476806317052414])

x3 = np.array([0, 3356675, 33596809, 133951027])
y3 = np.array([0, 0.1499752982020378, 0.29901477644443514, 0.36907796862125397])

l1, = plt.plot(x1, y1, color='red', label='training-set GIAANN.\ntrain c seg=4, f seg=4. inference seed=8.')	#direct connectivity enforced
#l2, = plt.plot(x2, y2, color='darkred', label='training-set GIAANN direct connectivity relaxed.\ntrain c seg=4, f seg=4.\ninference seed=8.')
l3, = plt.plot(x3, y3, color='magenta', label='training-set RoBERTa.\n6-layers, 512-token context window.')

x4 = np.array([0, 3778117, 37478755, 150184342])
y4 = np.array([0, 0.16826839773261715, 0.2694559271269091, 0.3117682368375455])
#x5 = np.array([0, 3778117, 37478755, 150184342])
#y5 = np.array([0, 0.21241243595138493, 0.2970432436446447, 0.3299400975271665])

x6 = np.array([0, 3356675, 33596809, 133951027])
y6 = np.array([0, 0.14418154878020287, 0.2946896313548088, 0.365966238629818])

l4, = plt.plot(x4, y4, color='blue', label='test-set GIAANN.\ntrain c seg=4, f seg=4. inference seed=8.')	#direct connectivity enforced
#l5, = plt.plot(x5, y5, color='darkblue', label='test-set GIAANN direct connectivity relaxed.\ntrain c seg=4, f seg=4.\ninference seed=8.')
l6, = plt.plot(x6, y6, color='cyan', label='test-set RoBERTa.\n6-layers, 512-token context window.')

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(20_000_000))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:g}M"))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Four minor ticks per 10M interval
plt.yticks(np.arange(0, 1.0+0.1, 0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

plt.xlabel("number of GIAANN o200k_base or RoBERTa byte-level BPE tokens")
plt.ylabel("OSCAR-2201 train/test-set accuracy (Top-1)")
plt.title("GIAANN Proto 2o with subword tokeniser")

plt.legend(handles=[l1, l3, l4, l6], loc='center right', bbox_to_anchor=(1.0, 0.6))

plt.savefig("protov2o-oscar-top1-accuracy-models-inferenceLeakyIntegrateAndFire-predictionConstraints.pdf", format="pdf", bbox_inches="tight")
plt.show()

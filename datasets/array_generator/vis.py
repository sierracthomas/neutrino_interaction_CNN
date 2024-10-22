import numpy as np
import matplotlib.pyplot as plt

test = np.load("nuenc_h5.0000000_0.npy")
number = 0
beam = np.linspace(-65, 65, 65)
drift = np.linspace(-65, 65, 65)
vertical = np.linspace(-65, 65, 65)
for i in test:
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    ax1.pcolormesh(beam, drift, i[1], cmap='gray_r')#, vmax = 0.0001)
    ax3.pcolormesh(beam, vertical, i[2], cmap='gray_r')#, vmax = 0.0001)
    ax4.pcolormesh(drift, vertical, i[0], cmap='gray_r')#, vmax = 0.0001)

    ax4.text(50, -50, r"$\circledast \^z$", fontsize=12)
    ax4.invert_xaxis()
    ax3.set_xlabel("Beam")
    ax3.set_ylabel("Vertical")
    ax1.set_ylabel("Drift")
    ax4.set_xlabel("Drift")
    for ax in fig.get_axes():
        ax.label_outer()


    plt.savefig(f"images_nc/{number}.png", transparent = False)
    number +=1
    plt.close()
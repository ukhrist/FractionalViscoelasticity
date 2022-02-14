

from config import *

tikz_folder = config['outputfolder']


"""
==================================================================================================================
Load data
==================================================================================================================
"""

(tip, EnergyElastic, EnergyKinetic, EnergyViscous, mode_abs, sol_abs, theta) = load_data(config['inputfolder']+"target_model")

time_steps = np.linspace(0, config['FinalTime'], config['nTimeSteps']+1)[1:]


"""
==================================================================================================================
FIGURES
==================================================================================================================
"""

import tikzplotlib
import matplotlib
# plt.style.use("ggplot")
plt.style.use("bmh")
font = {
    # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)


figure_settings = {
    'figsize'   :   (10,6),
}

plot_settings = {
    'markersize'   :   2,
}

legend_settings = {
    'loc'             :   'center left',
    'bbox_to_anchor'  :   (1.1, 0.5),
}

tikz_settings = {
    'axis_width'  :   '\\textwidth',
}

with torch.no_grad():

    """
    ==================================================================================================================
    Figure 1: Observations
    ==================================================================================================================
    """
    nmodes = mode_abs.shape[1]
    labels = [f"Mode {i+1}" for i in range(nmodes)]
    labels.append("Solution (scaled)")

    plt.figure('Norm of modes and solution', **figure_settings)
    plt.plot(time_steps, np.sqrt(mode_abs), "-", **plot_settings)
    plt.plot(time_steps, np.sqrt(sol_abs)/15, "--",  color="grey", **plot_settings)

    plt.legend(labels)
    plt.ylabel(r"Norm")
    plt.xlabel(r"$t$")
    tikzplotlib.save(tikz_folder+"plt_modes_norm.tex", **tikz_settings)
    plt.show()

    
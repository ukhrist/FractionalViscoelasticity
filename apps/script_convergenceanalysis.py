import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
from config import *
from scipy.optimize import curve_fit

plt.style.use("bmh")
font = {'size'   : 12}
matplotlib.rc('font', **font)

figure_settings = {'figsize' : (10,6)}
plot_settings = {'markersize' : 2}
legend_settings = {'loc' : 'center left', 'bbox_to_anchor' : (1.1, 0.5)}
tikz_settings = {'axis_width' : '0.45*160mm', 'axis_height' : '0.45*160mm', 'standalone': True}

exclude_loading = False

#create figure to display convergence for different alpha in one plot
fig_all = plt.figure(1, figsize=(6,6))
ax_all = fig_all.add_subplot()

#get correct color cycle for current theme
style_colors = matplotlib.rcParams['axes.prop_cycle']
style_colors = [color for color in style_colors]

#loop over all alpha, corresponding folders and files have to exist
for idx, alpha in enumerate([0., 0.25, 0.5, 0.75, 1.]):

    print("#"*80)
    print(f"Alpha = {alpha}")

    dir = config['outputfolder']
    dir_plot = dir + "convergence/plots/"
    dir += "convergence/alpha" + str(alpha)
    tikz_folder = dir_plot

    data = {}
    data_full = {}
    numsteps = []
    numsteps_full = []

    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            tmp_num = int(float((filename.split("_")[-1]).rstrip(".txt")))//5*4
            tmp_num_full = int(float((filename.split("_")[-1]).rstrip(".txt")))
        except:
            continue

        numsteps.append(tmp_num)
        numsteps_full.append(tmp_num_full)
        tmp_data_full = np.insert(np.loadtxt(filepath), 0, 0.)
        tmp_data = np.copy(tmp_data_full[len(tmp_data_full)//5:])
        data.update({tmp_num : tmp_data})
        data_full.update({tmp_num_full : tmp_data_full})

    if not exclude_loading:
        numsteps = numsteps_full
        data = data_full

    print("All solutions loaded.")

    numsteps = sorted(numsteps)
    numsteps_full = sorted(numsteps_full)
    print(numsteps)
    print(numsteps_full)

    # temporary to exclude finest discretization from analysis
    numsteps.pop()
    numsteps_full.pop()

    reference_steps = numsteps.pop()
    reference = data[reference_steps]

    reference_steps_full = numsteps_full.pop()
    reference = data[reference_steps]
    reference_full = data_full[reference_steps_full]

    dt = 1/np.array(numsteps)

    #Plot solutions
    t = np.linspace(0, config['FinalTime'], len(reference_full))
    plotskip = len(reference_full)//500
    fig = plt.figure('Solutions', **figure_settings)
    plt.plot(t[::plotskip], reference_full[::plotskip], **plot_settings)
    plt.xlabel("Time [s]")
    plt.ylabel("Tip displacement [arb. unit]")
    #plt.title(f"Alpha = {alpha}")
    #plt.savefig(dir_plot+f"Solution_{alpha}.pdf", bbox_inches="tight")
    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(dir_plot+f"plt_convergence_solution_{alpha}.tex", **tikz_settings)
    plt.close(fig)

    error2 = []

    for i, numstep in enumerate(numsteps):
        skip = reference_steps//numstep
        error = 0
        for j in range(numstep):
            u_ref_tip = reference[j*skip]
            u_tip = data[numstep][j]
            error += (u_ref_tip-u_tip)**2
        error2.append(np.sqrt(dt[i]*error))

    order = np.log(error2[-2]/error2[-1])/np.log(dt[-2]/dt[-1])
    print("Order: ", order)

    def f(dt, coeff, order):
        return np.log(dt)*order + coeff

    param, param_cov = curve_fit(f, dt[:], np.log(np.array(error2)[:]))
    fit_error = np.exp(f(dt, param[0], param[1]))
    print(param)

    fig = plt.figure('Solutions', figsize=(6,6))
    plt.plot(dt, error2, "o--", label=f"Data", zorder=10, **plot_settings)
    plt.plot(dt, dt**2/dt[0]**2*error2[0], label=f"Analytical order", c="tab:blue", linestyle="-", zorder=9, **plot_settings)
    #plt.plot(dt, fit_error, label=f"Fit ({param[1]:{0}.{3}})", c="grey", linestyle="-", zorder=9, **plot_settings)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("$dt$")
    plt.ylabel("$\mathcal{E}_{tip}(dt)$")
    #plt.title(f"Alpha = {alpha}")
    plt.legend()
    #plt.savefig(dir_plot+f"Convergence_{alpha}.pdf", bbox_inches="tight")
    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(dir_plot+f"plt_convergence_{alpha}.tex", **tikz_settings)
    plt.close(fig)
    print("Plots for specific alpha created.")

    ax_all.plot(dt, error2, "o--", color=style_colors[idx]['color'], label=f"$\\alpha = {alpha}$", zorder=10, **plot_settings)
    ax_all.plot(dt, dt**2/dt[0]**2*error2[0], color=style_colors[idx]['color'], label=f"_Analytical order", linestyle="-", zorder=9, **plot_settings)

print("#"*80)

plt.figure(1)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("$dt$")
plt.ylabel("$\mathcal{E}_{tip}(dt)$")
#plt.title(f"Alpha= {alpha}")
plt.legend()

#plt.savefig(dir_plot+f"Convergence.pdf", bbox_inches="tight")
tikzplotlib.clean_figure(fig_all)
tikz_settings = {'axis_width' : '0.7*160mm', 'axis_height' : '0.7*160mm', 'standalone': True}
tikzplotlib.save(dir_plot+f"plt_convergence.tex", **tikz_settings)
plt.show()
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
tikz_settings = {'axis_width' : '\\textwidth', 'standalone': True}

# correct or zeroed modes in initial condition
correct = False

config['FinalTime']  = 4

for alpha in [0., 0.5, 0.75, 1.]:

    print("#"*80)
    print(f"Alpha = {alpha}")

    dir = config['outputfolder']
    dir_plot = dir + "convergence/plots/"
    tikz_folder = dir_plot
    if correct:
        dir += f"convergence/correctIC/alpha{alpha}/"
    else:
        dir += f"convergence/wrongIC/alpha{alpha}/"

    sol = []
    nsteps  = []

    data = {}
    numsteps = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            tmp_num = int(float((filename.split("_")[-1]).rstrip(".txt")))
        except:
            if filename.split("_")[-1] == "ic.txt":
                print("Initial condition loaded")
                initial_cond = np.loadtxt(filepath)
            continue

        numsteps.append(tmp_num)
        tmp_data = np.insert(np.loadtxt(filepath), 0, 0.)
        data.update({tmp_num : tmp_data})

    for value in data.values():
        value[0] = initial_cond[-1]
    print("All solutions loaded.")
    numsteps = sorted(numsteps)
    print(numsteps)
    reference_steps = numsteps.pop()
    reference = data[reference_steps]

    dt = 1/np.array(numsteps)

    #Plot solutions
    t = np.linspace(0, config['FinalTime'], len(reference))
    plotskip = len(reference)//500
    fig = plt.figure('Solutions', **figure_settings)
    plt.plot(t[::plotskip]+1., reference[::plotskip], **plot_settings)
    plt.plot(np.linspace(0, 1, len(initial_cond)) ,initial_cond, **plot_settings, color="k")
    plt.xlabel("Time [s]")
    plt.ylabel("Tip displacement [arb. unit]")
    plt.title(f"Alpha= {alpha}")
    plt.savefig(dir_plot+f"Solution_{alpha}_{'correct' if correct else 'wrong'}.pdf", bbox_inches="tight")
    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(dir_plot+f"plt_convergence_solution_{alpha}_{'correct' if correct else 'wrong'}.tex", **tikz_settings)
    plt.close()

    #error1 = []
    #
    #for numstep in numsteps:
    #    error1.append(np.abs(data[numstep][-1] - reference[-1]))
    #
    #order = np.log(error1[-2]/error1[-1])/np.log(dt[-2]/dt[-1])
    #print("Order: ", order)
    #
    #plt.plot(dt, error1, "o-")
    ##plt.title(f"Convergence  -  Alpha= {alpha}  -  Order= {ord:{0}.{3}}")
    #plt.yscale("log")
    #plt.xscale("log")
    #plt.xlabel("$dt$")
    #plt.ylabel("$\mathcal{E}_\infty(dt)$")
    #plt.savefig(dir_plot+f"Convergence_{alpha}.pdf", bbox_inches="tight")
    #plt.show()


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

    #param, param_cov = curve_fit(f, dt[3:], np.log(np.array(error2)[3:]))
    #fit_error = np.exp(f(dt, param[0], param[1]))
    #print(param)

    fig = plt.figure('Solutions', figsize=(6,6))
    plt.plot(dt, error2, "o--", label="Data", zorder=10, **plot_settings)
    plt.plot(dt, dt**2/dt[0]**2*error2[0], label=f"Analytical order", c="tab:blue", linestyle="-", zorder=9, **plot_settings)
    #plt.plot(dt, fit_error, label=f"Fit ({param[1]:{0}.{3}})", c="grey", linestyle="-", zorder=9, **plot_settings)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("$dt$")
    plt.ylabel("$\mathcal{E}_{tip}(dt)$")
    plt.title(f"Alpha= {alpha}")
    plt.legend()
    plt.savefig(dir_plot+f"Convergence_{alpha}_{'correct' if correct else 'wrong'}.pdf", bbox_inches="tight")
    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(dir_plot+f"plt_convergence_{alpha}_{'correct' if correct else 'wrong'}.tex", **tikz_settings)
    plt.close()

    print("All plots created.")
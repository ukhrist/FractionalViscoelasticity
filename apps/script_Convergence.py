import numpy as np
import sys
import os
from config import *
import time
from datetime import datetime

# smaller mesh for faster execution
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 20, 4, 2)
config['mesh'] = mesh
config['inverse'] = True

# continuous loading
magnitude = 1.
tmax      = 4/5
tzero     = 1.
loading = Expression(("0", "0", "0"), degree=0)

# get input values
alpha = float(sys.argv[1])
index = int(sys.argv[2])
maxindex = int(sys.argv[3])
if sys.argv[2] in ['True', '1', '1.']:
    correct = True
else:
    correct = False

timestring = datetime.strftime(datetime.now(), "%Y%m%d%H%M")

# infmode boolean from config
infmode = config.get('infmode', False)

# compute sum of exponentials approximation for fixed alpha
RA = RationalApproximation(alpha=alpha)
parameters = list(RA.c) + list(RA.d)
if infmode==True: parameters.append(RA.c_inf)
kernel  = SumOfExponentialsKernel(parameters=parameters)
kernels = [kernel]

path = config['outputfolder']
path = path+f"convergence/"
if correct:
    path += "correctIC/"
else:
    path += "wrongIC/"
path = path+f"alpha{alpha}/"
config['viscosity'] = True

u, v, a, history, modes, F_old = load_data(path+f"initialcondition")

n_steps_list = 2**np.arange(0, maxindex)*1e2
n_steps_list = np.append(n_steps_list, n_steps_list[-1]*10)
n_steps = n_steps_list[index]
config['nTimeSteps'] = int(n_steps*4)
config['FinalTime']  = 4

print(f"START: dt={1/n_steps} started")

Model = ViscoelasticityProblem(**config, kernels=kernels)

# set initial condition
Model.kernels[0].modes = modes
Model.kernels[0].F_old = F_old

if not correct:
    Model.kernels[0].modes = 0
    Model.kernels[0].F_old = 0 
    Model.history = history

Model.u = u
Model.v = v
Model.a = a

Model.forward_solve(loading=loading)
obs = Model.observations
data = obs.numpy()

np.savetxt(path+f"tipdisplacement_{timestring}_{alpha}_{n_steps}.txt", data)

print(f"END: dt={1/n_steps} finished")
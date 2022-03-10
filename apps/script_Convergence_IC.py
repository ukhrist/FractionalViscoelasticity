import numpy as np
import sys
import os
from config import *
from datetime import datetime
from src.data_manager_convergence import save_data, load_data

# smaller mesh for faster execution
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 20, 4, 2)
config['mesh'] = mesh

# continuous loading
magnitude = 1.
tmax      = 4/5
tzero     = 1.
loading = Expression(("0", "t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0)

# get input values
alpha = float(sys.argv[1])

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
path = path+f"convergence/alpha{alpha}/"
os.makedirs(path, exist_ok=True)

config['viscosity']  = True
config['FinalTime']  = 1
config['nTimeSteps'] = 8000

print()
print(f"Computing initial condition started")
print()

Model = ViscoelasticityProblem(**config, kernels=kernels)
Model.forward_solve(loading=loading)
obs = Model.observations
data = obs.numpy()
np.savetxt(path+f"tipdisplacement_{timestring}_{alpha}_ic.txt", data)
save_data(path+f"initialcondition", Model)

print()
print(f"Computing initial condition finished")
print()
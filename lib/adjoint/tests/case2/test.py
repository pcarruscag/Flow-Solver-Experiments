#!/usr/bin/env python 
import os
import math
import shutil
import subprocess

os.chdir(os.environ["TESTDIR"])

# clean old files
try: os.remove("pass.inf")
except: pass
try: os.remove("objectiveGradient.txt")
except: pass
try: os.remove("initialConditions.dat")
except: pass
try: os.remove("initialConditionsB.dat")
except: pass

# set initial conditions and run
subprocess.call("unzip initialConditions.zip",shell=True)
subprocess.call("../../../../bin/Release/console -adj ./ 4 0 > log.txt",shell=True)

# compare results
fid = open("referenceGradient.txt")
ref = fid.readlines()
fid.close()

fid = open("objectiveGradient.txt")
cur = fid.readlines()
fid.close()

# factors to normalize differences
fval = [float(ref[0].split()[0]), float(ref[0].split()[1])]
norm = 0.0
for line in ref[2:]:
    for i in range(2):
        val = float(line.split()[i])
        norm += (val/fval[i])**2
norm = math.sqrt(0.5*norm/(len(ref)-2))

# compare
maxDiff = 0.0
for a,b in zip(ref[2:],cur[2:]):
    for i in range(2):
        maxDiff = max(maxDiff,abs(float(a.split()[i])-float(b.split()[i]))/fval[i]/norm)

if maxDiff < 0.025:
    fid = open("pass.inf","w")
    fid.close()

print(maxDiff)


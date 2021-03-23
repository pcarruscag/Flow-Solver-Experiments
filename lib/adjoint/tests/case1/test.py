#!/usr/bin/env python 
import os
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
shutil.copy("iniFlow.dat","initialConditions.dat")
shutil.copy("iniAdj.dat","initialConditionsB.dat")
subprocess.call("../../../../bin/Release/console -adj ./ 4 0 > log.txt",shell=True)

# compare results
fid = open("referenceGradient.txt")
ref = fid.readlines()
fid.close()

fid = open("objectiveGradient.txt")
cur = fid.readlines()
fid.close()

maxDiff = -1.0
for a,b in zip(ref,cur):
    try: maxDiff = max(maxDiff,abs(float(a)/float(b)-1.0))
    except: pass

if maxDiff > -1.0 and maxDiff < 0.1:
    fid = open("pass.inf","w")
    fid.close()

print(maxDiff)

import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pickle
import multiprocessing as mp

from particle import Particle
from magneticEnvironment import MagneticEnvironment, StableMagneticEnvironment
from motionSimulation import Simulator


if __name__ == '__main__':
    simulator = Simulator(
        deltaT=1e-6,
        maxIter=int(1e11)
    )
    envWithFM = MagneticEnvironment.initFromCSV(brPath='./BrDistributionWithFM.csv', bzPath='./BzDistributionWithFM.csv', plotXYScale=0.5)
    # envLow = MagneticEnvironment.initFromEnvWithScale(env=env, scale=1e-2)
    envWithoutFM = MagneticEnvironment.initFromCSV(brPath='./BrDistributionWithoutFM.csv', bzPath='./BzDistributionWithoutFM.csv', plotXYScale=0.5)

    simulator.runUnderHighLowEnvRandomParticles(envHigh=envWithFM, envLow=envWithoutFM, samples=100, shouldPlot=False)

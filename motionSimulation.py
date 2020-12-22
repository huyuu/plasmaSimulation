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


class Simulator():
    def __init__(self, deltaT, maxIter):
        self.deltaT = deltaT
        self.maxIter = maxIter


    def run(self, particles, envs):
        assert len(particles) == len(envs)
        with mp.Pool(processes=len(particles)) as pool:
            pool.starmap(self.simulate, zip(particles, envs))


    def simulate(self, particle, magneticEnvironment):
        # get move self.particle
        particle.reset()
        particle.moveUntilStop(magneticEnvironment=magneticEnvironment, deltaT=self.deltaT, maxIter=self.maxIter)
        # plot in animation
        # self.plotAnimation3d(particle=particle, magneticEnvironment=magneticEnvironment)
        self.plotTrajectory3d(particle=particle, magneticEnvironment=magneticEnvironment)


    def plotAnimation3d(self, particle, magneticEnvironment):
        fig = pl.figure()
        ax = p3.Axes3D(fig)
        # https://matplotlib.org/gallery/animation/simple_3danim.html
        def __plot(i):
            pl.cla()
            ax.set_xlim(-magneticEnvironment.plotCubeX0, magneticEnvironment.plotCubeX0)
            ax.set_ylim(-magneticEnvironment.plotCubeY0, magneticEnvironment.plotCubeY0)
            ax.set_zlim(-magneticEnvironment.plotCubeZ0, magneticEnvironment.plotCubeZ0)
            ax.set_xlabel('x [m]', fontsize=16)
            ax.set_ylabel('y [m]', fontsize=16)
            ax.set_zlabel('z [m]', fontsize=16)
            ax.tick_params(labelsize=12)
            ax.scatter3D(particle.spaceTimeSeries[i, 0], particle.spaceTimeSeries[i, 1], particle.spaceTimeSeries[i, 2], label='t = {:.3f} [ms]'.format(particle.spaceTimeSeries[i, 3]*1e3))
            ax.legend(fontsize=12)
        # https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html#matplotlib.animation.FuncAnimation
        ani = animation.FuncAnimation(fig, __plot, interval=1, frames=range(particle.spaceTimeSeries.shape[0]), repeat=False)
        pl.show()


    def plotTrajectory3d(self, particle, magneticEnvironment):
        fig = pl.figure()
        ax = pl.axes(projection='3d')
        ax.set_xlim(-magneticEnvironment.plotCubeX0, magneticEnvironment.plotCubeX0)
        ax.set_ylim(-magneticEnvironment.plotCubeY0, magneticEnvironment.plotCubeY0)
        ax.set_zlim(-magneticEnvironment.plotCubeZ0, magneticEnvironment.plotCubeZ0)
        ax.set_xlabel('x [m]', fontsize=16)
        ax.set_ylabel('y [m]', fontsize=16)
        ax.set_zlabel('z [m]', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.scatter3D(particle.spaceTimeSeries[:, 0].ravel(), particle.spaceTimeSeries[:, 1].ravel(), particle.spaceTimeSeries[:, 2].ravel(), c=particle.spaceTimeSeries[:, 3].ravel(), cmap='Blues')
        pl.show()


if __name__ == '__main__':
    simulator = Simulator(
        deltaT=1e-8,
        maxIter=int(1e14)
    )
    simulator.run(
        particles=[
            Particle(mass=1e-6, q=1.0, x0=nu.array([0, 0, 1.0]), v0=nu.array([1e-2, 1e-2, -100]), a0=nu.zeros(3)),
            Particle(mass=1e-6, q=1.0, x0=nu.array([0, 0, 1.0]), v0=nu.array([1e-2, 1e-2, -100]), a0=nu.zeros(3))
        ],
        envs=[
            StableMagneticEnvironment(br0=0, bz0=1.0, plotCubeX0=1.0, plotCubeY0=1.0, plotCubeZ0=1.0),
            StableMagneticEnvironment(br0=0, bz0=10e-3, plotCubeX0=1.0, plotCubeY0=1.0, plotCubeZ0=1.0)
        ]
    )

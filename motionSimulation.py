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


    def runUnderCertainPairs(self, particles, envs):
        assert len(particles) == len(envs)
        with mp.Pool(processes=min(len(particles), mp.cpu_count()-1)) as pool:
            pool.starmap(self.simulate, zip(particles, envs))
        for particle, env in zip(particles, envs):
            self.plotTrajectory3d(particle, env)


    def runUnderSameEnvRandomParticles(self, env, samples):
        particles = []
        for _ in range(samples):
            # https://note.nkmk.me/python-numpy-random/
            x0_x = env.plotCubeX0 * (2*nu.random.rand()-1)
            x0_y = env.plotCubeY0 * (2*nu.random.rand()-1)
            x0_z = env.plotCubeZ0
            v0_x = env.plotCubeX0/1e2 * (2*nu.random.rand()-1)
            v0_y = env.plotCubeY0/1e2 * (2*nu.random.rand()-1)
            v0_z = env.plotCubeZ0 * (-1)*nu.random.rand()
            particle = Particle(mass=1e-6, q=1.0, x0=nu.array([x0_x, x0_y, x0_z]), v0=nu.array([v0_x, v0_y, v0_z]), a0=nu.zeros(3))
            particles.append(particle)
        with mp.Pool(processes=min(len(particles), mp.cpu_count()-1)) as pool:
            pool.starmap(self.simulate, [ (particle, env) for particle in particles ])
        self.plotTrajectories3d(particles, env)


    def runUnderHighLowEnvRandomParticles(self, envHigh, envLow, samples):
        particlesHigh = []
        particlesLow = []
        for _ in range(samples):
            # https://note.nkmk.me/python-numpy-random/
            x0_x = envHigh.plotCubeX0 * (2*nu.random.rand()-1)
            x0_y = envHigh.plotCubeY0 * (2*nu.random.rand()-1)
            x0_z = envHigh.plotCubeZ0
            v0_x = envHigh.plotCubeX0/1e2 * (2*nu.random.rand()-1)
            v0_y = envHigh.plotCubeY0/1e2 * (2*nu.random.rand()-1)
            # v0_z = envHigh.plotCubeZ0 * (-1)*(nu.random.rand()+0.5)
            particlesHigh.append(Particle(mass=1e-3, q=1.0, x0=nu.array([x0_x, x0_y, x0_z]), v0=nu.array([v0_x, v0_y, v0_z]), a0=nu.zeros(3)))
            particlesLow.append(Particle(mass=1e-3, q=1.0, x0=nu.array([x0_x, x0_y, x0_z]), v0=nu.array([v0_x, v0_y, v0_z]), a0=nu.zeros(3)))
        # simulate
        zipped = [ (particleHigh, envHigh) for particleHigh in particlesHigh ]
        # extend for Low
        zipped.extend([ (particleLow, envLow) for particleLow in particlesLow ])
        with mp.Pool(processes=min(len(particlesHigh)*2, mp.cpu_count()-1)) as pool:
            _ = pool.starmap(self.simulate, zipped)
        self.saveTrajectories(particlesHigh, envHigh, path='./particlesUnderEnvHigh.pickle')
        self.saveTrajectories(particlesLow, envLow, path='./particlesUnderEnvLow.pickle')
        self.plotTrajectories3d(particlesHigh, envHigh)
        self.plotTrajectories3d(particlesLow, envLow)


    def simulate(self, particle, magneticEnvironment):
        # get move self.particle
        particle.reset()
        spaceTime = particle.moveUntilStop(magneticEnvironment=magneticEnvironment, deltaT=self.deltaT, maxIter=self.maxIter)
        print(spaceTime)
        return spaceTime
        # plot in animation
        # self.plotAnimation3d(particle=particle, magneticEnvironment=magneticEnvironment)
        # self.plotTrajectory3d(particle=particle, magneticEnvironment=magneticEnvironment)


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
        # plot quiver 3d
        _xs, _ys, _zs, bs_x, bs_y, bs_z, length, arrow_length_ratio = magneticEnvironment.bFieldQuiverProperties
        # https://matplotlib.org/3.1.0/gallery/mplot3d/quiver3d.html#d-quiver-plot
        # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#quiver
        ax.quiver(_xs, _ys, _zs, bs_x, bs_y, bs_z, length=length, arrow_length_ratio=arrow_length_ratio)
        ax.scatter3D(particle.spaceTimeSeries[:, 0].ravel(), particle.spaceTimeSeries[:, 1].ravel(), particle.spaceTimeSeries[:, 2].ravel(), c=particle.spaceTimeSeries[:, 3].ravel(), cmap='Blues')
        pl.show()


    def plotTrajectories3d(self, particles, magneticEnvironment):
        fig = pl.figure()
        ax = pl.axes(projection='3d')
        ax.set_xlim(-magneticEnvironment.plotCubeX0, magneticEnvironment.plotCubeX0)
        ax.set_ylim(-magneticEnvironment.plotCubeY0, magneticEnvironment.plotCubeY0)
        ax.set_zlim(-magneticEnvironment.plotCubeZ0, magneticEnvironment.plotCubeZ0)
        ax.set_xlabel('x [m]', fontsize=16)
        ax.set_ylabel('y [m]', fontsize=16)
        ax.set_zlabel('z [m]', fontsize=16)
        ax.tick_params(labelsize=12)
        # plot quiver 3d
        _xs, _ys, _zs, bs_x, bs_y, bs_z, length, arrow_length_ratio = magneticEnvironment.bFieldQuiverProperties
        # https://matplotlib.org/3.1.0/gallery/mplot3d/quiver3d.html#d-quiver-plot
        # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#quiver
        ax.quiver(_xs, _ys, _zs, bs_x, bs_y, bs_z, length=length, arrow_length_ratio=arrow_length_ratio)
        # plot trajectories
        for particle in particles:
            ax.scatter3D(particle.spaceTimeSeries[:, 0].ravel(), particle.spaceTimeSeries[:, 1].ravel(), particle.spaceTimeSeries[:, 2].ravel(), c=particle.spaceTimeSeries[:, 3].ravel(), cmap='viridis')
        pl.show()


    def saveTrajectories(self, particles, env, path):
        saveDict = {
            'particles': particles,
            'env': env
        }
        with open(path, 'wb') as file:
            pickle.dump(saveDict, file)



if __name__ == '__main__':
    simulator = Simulator(
        deltaT=1e-5,
        maxIter=int(1e10)
    )
    env = MagneticEnvironment.initFromCSV(brPath='./BrDistributionOuter.csv', bzPath='./BzDistributionOuter.csv')
    # simulator.runUnderSameEnvRandomParticles(env=env, samples=10)

    envLow = MagneticEnvironment.initFromEnvWithScale(env=env, scale=1e-2)
    simulator.runUnderHighLowEnvRandomParticles(envHigh=env, envLow=envLow, samples=10)

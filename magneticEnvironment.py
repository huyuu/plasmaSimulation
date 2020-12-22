import numpy as nu
import pandas as pd
from numpy import sqrt, arctan2, abs


class MagneticEnvironment():
    def __init__(self, brDistribution, bzDistribution, plotCubeX0, plotCubeY0, plotCubeZ0):
        self.brDistribution = brDistribution
        self.bzDistribution = bzDistribution
        self.plotCubeX0 = plotCubeX0
        self.plotCubeY0 = plotCubeY0
        self.plotCubeZ0 = plotCubeZ0


    @classmethod
    def initFromCSV(cls, brPath, bzPath):
        brDistribution = pd.read_csv(brPath)
        bzDistribution = pd.read_csv(bzPath)
        return cls(brDistribution=brDistribution, bzDistribution=bzDistribution)


    def bAt(self, position):
        theta = 0
        # return the interpolated B(x, y, z)
        return b


    def isPointInsidePlotCube(self, point):
        return abs(point[0]) <= abs(self.plotCubeX0) and abs(point[1]) <= abs(self.plotCubeY0) and abs(point[2]) <= abs(self.plotCubeZ0)



class StableMagneticEnvironment(MagneticEnvironment):
    def __init__(self, br0, bz0, plotCubeX0, plotCubeY0, plotCubeZ0):
        rs = nu.linspace(-plotCubeX0, plotCubeX0, 10).reshape(1, -1)
        zs = nu.linspace(-plotCubeY0, plotCubeY0, 10).reshape(-1, 1)
        brDistribution = rs * zs * br0
        bzDistribution = rs * zs * bz0
        super(StableMagneticEnvironment, self).__init__(brDistribution=brDistribution, bzDistribution=bzDistribution, plotCubeX0=plotCubeX0, plotCubeY0=plotCubeY0, plotCubeZ0=plotCubeZ0)
        self.br0 = br0
        self.bz0 = bz0


    def bAt(self, position):
        if self.br0 == 0:
            return nu.array([0, 0, self.bz0])
        else:
            theta = arctan2(y/x)
            bx = self.br0 * cos(theta)
            by = self.br0 * sin(theta)
            return nu.array([bx, by, self.bz0])

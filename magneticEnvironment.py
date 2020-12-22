import numpy as nu
import pandas as pd
from scipy import interpolate
from numpy import sqrt, arctan2, abs, sin, cos, tan


class MagneticEnvironment(object):
    def __init__(self, brDistribution, bzDistribution, plotCubeX0, plotCubeY0, plotCubeZ0):
        self.brDistribution = brDistribution
        self.bzDistribution = bzDistribution
        self.plotCubeX0 = plotCubeX0
        self.plotCubeY0 = plotCubeY0
        self.plotCubeZ0 = plotCubeZ0
        self.__interpolatedBrFunc = interpolate.interp2d(
            x=self.brDistribution.index.values.ravel(),
            y=self.brDistribution.columns.values.ravel(),
            z=self.brDistribution.values,
            kind='linear'
        )
        self.__interpolatedBzFunc = interpolate.interp2d(
            x=self.bzDistribution.index.values.ravel(),
            y=self.bzDistribution.columns.values.ravel(),
            z=self.bzDistribution.values,
            kind='linear'
        )


    @classmethod
    def initFromCSV(cls, brPath, bzPath):
        # get br
        brDistribution = pd.read_csv(brPath, skiprows=8)
        brDistribution.columns = ['r', 'z', 'B']
        brDistribution = brDistribution.pivot(index='r', columns='z', values='B')
        # get plot
        plotCubeX0 = brDistribution.index.values.ravel().max()
        plotCubeY0 = brDistribution.index.values.ravel().max()
        plotCubeZ0 = brDistribution.columns.values.ravel().max()
        # get bz
        bzDistribution = pd.read_csv(brPath, skiprows=8)
        bzDistribution.columns = ['r', 'z', 'B']
        bzDistribution = bzDistribution.pivot(index='r', columns='z', values='B')
        # return MagneticEnvironment
        return cls(brDistribution=brDistribution, bzDistribution=bzDistribution, plotCubeX0=plotCubeX0, plotCubeY0=plotCubeY0, plotCubeZ0=plotCubeZ0)


    def bAt(self, position):
        r = sqrt(position[0]**2 + position[1]**2)
        # https://note.nkmk.me/python-numpy-sin-con-tan/
        theta = arctan2(position[1], position[0])
        z = position[2]
        # #
        # almostSameRPointIndices = nu.where(abs(self.brDistribution.index.values.ravel() - r) <= 1e-6)[0]
        # assert len(almostSameRPointIndices) <= 2
        # almostSameZPointIndices = nu.where(abs(self.bzDistribution.index.values.ravel() - z) <= 1e-6)[0]
        # assert len(almostSameZPointIndices) <= 2
        # # if r and z are on grid point, return the distribution of it
        # if len(almostSameRPointIndices) != 0 and len(almostSameZPointIndices) != 0:
        #     rIndex = almostSameRPointIndices[0]
        #     zIndex = almostSameZPointIndices[0]
        #     br = self.brDistribution.iloc[rIndex, zIndex]
        #     bz = self.bzDistribution.iloc[rIndex, zIndex]
        # # if only r is on grid
        # elif len(almostSameRPointIndices) != 0:
        #     rIndex = almostSameRPointIndices[0]
        #     zLeftIndex = nu.where(self.brDistribution.columns.values.ravel() < z)[0][-1]
        #     distance2zLeft = z - self.brDistribution.columns.values.ravel()[zLeftIndex]
        #     zRightIndex = nu.where(self.brDistribution.columns.values.ravel() > z)[0][0]
        #     distance2zRight = self.brDistribution.columns.values.ravel()[zRightIndex] - z
        #     # get br from interpolating br@zLeft and br@zRight
        #     br_zLeft = self.brDistribution.iloc[rIndex, zLeftIndex]
        #     br_zRight = self.brDistribution.iloc[rIndex, zRightIndex]
        #     br = (distance2zRight*br_zLeft + distance2zLeft*br_zRight) / (distance2zLeft+distance2zRight)
        #     # get bz from interpolating bz@zLeft and bz@zRight
        #     bz_zLeft = self.bzDistribution.iloc[rIndex, zLeftIndex]
        #     bz_zRight = self.bzDistribution.iloc[rIndex, zRightIndex]
        #     bz = (distance2zRight*bz_zLeft + distance2zLeft*bz_zRight) / (distance2zLeft+distance2zRight)
        # # if only z is on the grid
        # elif len(almostSameZPointIndices) != 0:
        #     zIndex = almostSameZPointIndices[0]
        #     rLeftIndex = nu.where(self.brDistribution.index.values.ravel() < r)[0][-1]
        #     distance2rLeft = r - self.brDistribution.index.values.ravel()[rLeftIndex]
        #     rRightIndex = nu.where(self.brDistribution.index.values.ravel() > r)[0][0]
        #     distance2rRight = self.brDistribution.index.values.ravel()[rRightIndex] - r
        #     # get br from interpolating br@rLeft and br@rRight
        #     br_rLeft = self.brDistribution.iloc[rLeftIndex, zIndex]
        #     br_rRight = self.brDistribution.iloc[rRightIndex, zIndex]
        #     br = (distance2rRight*br_rLeft + distance2rLeft*br_rRight) / (distance2rLeft+distance2rRight)
        #     # get bz from interpolating bz@zLeft and bz@zRight
        #     bz_rLeft = self.bzDistribution.iloc[rLeftIndex, zIndex]
        #     bz_rRight = self.bzDistribution.iloc[rRightIndex, zIndex]
        #     bz = (distance2rRight*bz_rLeft + distance2rLeft*bz_rRight) / (distance2rLeft+distance2rRight)
        # # if none if them are on grid
        # else:
        #     # rLeft and rRight
        #     rLeftIndex = nu.where(self.brDistribution.index.values.ravel() < r)[0][-1]
        #     distance2rLeft = r - self.brDistribution.index.values.ravel()[rLeftIndex]
        #     rRightIndex = nu.where(self.brDistribution.index.values.ravel() > r)[0][0]
        #     distance2rRight = self.brDistribution.index.values.ravel()[rRightIndex] - r
        #     # zLeft and zRight
        #     zLeftIndex = nu.where(self.brDistribution.columns.values.ravel() < z)[0][-1]
        #     distance2zLeft = z - self.brDistribution.columns.values.ravel()[zLeftIndex]
        #     zRightIndex = nu.where(self.brDistribution.columns.values.ravel() > z)[0][0]
        #     distance2zRight = self.brDistribution.columns.values.ravel()[zRightIndex] - z
        #     # get br
        #     br_rLeft_zLeft = self.brDistribution.iloc[rLeftIndex, zLeftIndex]
        #     br_rLeft_zRight = self.brDistribution.iloc[rLeftIndex, zRightIndex]
        #     br_rRight_zLeft = self.brDistribution.iloc[rRightIndex, zLeftIndex]
        #     br_rRight_zRight = self.brDistribution.iloc[rRightIndex, zRightIndex]
        #     br =

        br = self.__interpolatedBrFunc(r, z)[0]
        bz = self.__interpolatedBzFunc(r, z)[0]
        bx = br * cos(theta)
        by = br * sin(theta)
        return nu.array([bx, by, bz])


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
            theta = arctan2(position[1], position[0])
            bx = self.br0 * cos(theta)
            by = self.br0 * sin(theta)
            return nu.array([bx, by, self.bz0])

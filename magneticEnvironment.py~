import numpy as nu
import pandas as pd
from scipy import interpolate
from numpy import sqrt, arctan2, abs, sin, cos, tan, pi


class MagneticEnvironment(object):
    def __init__(self, brDistribution, bzDistribution, plotCubeX0, plotCubeY0, plotCubeZ0):
        self.brDistribution = brDistribution
        self.bzDistribution = bzDistribution
        self.plotCubeX0 = plotCubeX0
        self.plotCubeY0 = plotCubeY0
        self.plotCubeZ0 = plotCubeZ0
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html#scipy.interpolate.interp2d
        rs = brDistribution.index.values.ravel()
        zs = brDistribution.columns.values.ravel()
        # _rs, _zs = nu.meshgrid(rs, zs, indexing='ij')
        self.__interpolatedBrFunc = interpolate.RegularGridInterpolator(
            points=(rs, zs),
            values=self.brDistribution.values,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        self.__interpolatedBzFunc = interpolate.RegularGridInterpolator(
            points=(rs, zs),
            values=self.bzDistribution.values,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        # create b field distribution plots
        bFieldPlotPoints = 15
        xs=nu.linspace(-plotCubeX0, plotCubeX0, bFieldPlotPoints)
        ys=nu.linspace(-plotCubeY0, plotCubeY0, bFieldPlotPoints)
        zs=nu.linspace(-plotCubeZ0, plotCubeZ0, bFieldPlotPoints)
        _xs, _ys, _zs, bs_x, bs_y, bs_z = self.bsAt(xs=xs, ys=ys, zs=zs)
        length = 0.03
        arrow_length_ratio = 0.8
        self.bFieldQuiverProperties = (_xs, _ys, _zs, bs_x, bs_y, bs_z, length, arrow_length_ratio)


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
        bzDistribution = pd.read_csv(bzPath, skiprows=8)
        bzDistribution.columns = ['r', 'z', 'B']
        bzDistribution = bzDistribution.pivot(index='r', columns='z', values='B')
        # return MagneticEnvironment
        return cls(brDistribution=brDistribution, bzDistribution=bzDistribution, plotCubeX0=plotCubeX0, plotCubeY0=plotCubeY0, plotCubeZ0=plotCubeZ0)


    @classmethod
    def initFromEnvWithScale(cls, env, scale):
        newEnv = cls(
            brDistribution=env.brDistribution*scale,
            bzDistribution=env.bzDistribution*scale,
            plotCubeX0=env.plotCubeX0,
            plotCubeY0=env.plotCubeY0,
            plotCubeZ0=env.plotCubeZ0
        )
        (_xs, _ys, _zs, bs_x, bs_y, bs_z, length, arrow_length_ratio) = newEnv.bFieldQuiverProperties
        newEnv.bFieldQuiverProperties = (_xs, _ys, _zs, bs_x, bs_y, bs_z, 2.0, arrow_length_ratio)
        return newEnv


    def bAt(self, position):
        r = sqrt(position[0]**2 + position[1]**2)
        # https://note.nkmk.me/python-numpy-sin-con-tan/
        theta = arctan2(position[1], position[0])
        z = position[2]
        br = self.__interpolatedBrFunc(nu.array([[r, z]]))[0]
        bz = self.__interpolatedBzFunc(nu.array([[r, z]]))[0]
        bx = br * cos(theta)
        by = br * sin(theta)
        return nu.array([bx, by, bz])


    def bsAt(self, xs, ys, zs):
        samplesPerAxis = xs.shape[0]
        bs_x = nu.zeros((samplesPerAxis, samplesPerAxis, samplesPerAxis))
        bs_y = nu.zeros((samplesPerAxis, samplesPerAxis, samplesPerAxis))
        bs_z = nu.zeros((samplesPerAxis, samplesPerAxis, samplesPerAxis))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    bx, by, bz = self.bAt(position=nu.array([x, y, z]))
                    bs_x[i, j, k] = bx
                    bs_y[i, j, k] = by
                    bs_z[i, j, k] = bz
        _xs, _ys, _zs = nu.meshgrid(xs, ys, zs, indexing='ij')
        return (_xs, _ys, _zs, bs_x, bs_y, bs_z)


    def isPointInsidePlotCube(self, point):
        return abs(point[0]) <= abs(self.plotCubeX0) and abs(point[1]) <= abs(self.plotCubeY0) and abs(point[2]) <= abs(self.plotCubeZ0)


    def __getstate__(self):
        return {
            'brDistribution': self.brDistribution,
            'bzDistribution': self.bzDistribution,
            'plotCubeX0': self.plotCubeX0,
            'plotCubeY0': self.plotCubeY0,
            'plotCubeZ0': self.plotCubeZ0,
            'interpolatedBrFunc': self.__interpolatedBrFunc,
            'interpolatedBzFunc': self.__interpolatedBzFunc,
            'bFieldQuiverProperties': self.bFieldQuiverProperties
        }


    def __setstate__(self, state):
        self.brDistribution = state['brDistribution']
        self.bzDistribution = state['bzDistribution']
        self.plotCubeX0 = state['plotCubeX0']
        self.plotCubeY0 = state['plotCubeY0']
        self.plotCubeZ0 = state['plotCubeZ0']
        self.__interpolatedBrFunc = state['interpolatedBrFunc']
        self.__interpolatedBzFunc = state['interpolatedBzFunc']
        self.bFieldQuiverProperties = state['bFieldQuiverProperties']



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

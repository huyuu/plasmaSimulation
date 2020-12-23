import numpy as nu
import pandas as pd


class Particle():
    # x, v, a are vectors
    def __init__(self, mass, q, x0, v0, a0):
        self.mass = mass
        self.q = q
        self.x0 = x0
        self.x = x0
        self.v0 = v0
        self.v = v0
        self.a0 = a0
        self.a = a0
        self.t = 0
        self.spaceTimeSeries = nu.array([self.x[0], self.x[1], self.x[2], self.t]).reshape(1, -1)


    def reset(self):
        self.x = self.x0
        self.v = self.v0
        self.a = self.a0
        self.t = 0
        self.spaceTimeSeries = nu.array([self.x[0], self.x[1], self.x[2], self.t]).reshape(1, -1)


    def moveOneStep(self, force, deltaT):
        # update x from a
        self.t += deltaT
        self.a = force / self.mass
        self.v += self.a * deltaT
        self.x += self.v * deltaT
        # return new position
        return nu.array([self.x[0], self.x[1], self.x[2], self.t])


    def moveUntilStop(self, magneticEnvironment, deltaT, maxIter):
        step = 1
        while magneticEnvironment.isPointInsidePlotCube(self.x) and step <= maxIter:
            # calculate force from F = qvB. Mark that force is an array of 3 components.
            force = self.q * nu.cross(self.v, magneticEnvironment.bAt(position=self.x))
            # update x
            newSpaceTime = self.moveOneStep(force=force, deltaT=deltaT)
            # save to positionSeries
            self.spaceTimeSeries = nu.concatenate([self.spaceTimeSeries, newSpaceTime.reshape(1, -1)])
            step += 1
        return self.spaceTimeSeries.copy()

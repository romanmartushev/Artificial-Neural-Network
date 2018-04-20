import numpy as np
from random import *
from math import *

class InputNode(object):
    def __init__(self):
        self.input = 0.0

    def setInput(self,input):
        self.input = input

    def getOutput(self):
        return self.input

class BiasNode(object):

    def getOutput(self,max,min):
        return (1.0 - min)/(max - min)

class HiddenNode(object):
    def __init__(self):
        self.output = 0.0
        self.input = 0.0
        self.delta = 0.0

    def setInput(self,input):
        self.input = input

    def getInput(self):
        return self.input

    def getOutput(self):
        return self.output

    def setDelta(self,x):
        self.delta = x * self.derivative()

    def getDelta(self):
        return self.delta

    def sigmoid(self,x):
        self.input = x
        self.output = 1/(1 + np.exp(-x))

    def derivative(self):
        return (1/(1 + np.exp(-self.output))) * (1-1/(1 + np.exp(-self.output)))

class OutputNode(object):
    def __init__(self):
        self.input = 0.0
        self.output = 0.0
        self.delta = 0.0

    def setInput(self,input):
        self.input = input

    def getInput(self):
        return self.input

    def getOutput(self):
        return self.output

    def setDelta(self,x,max,min):
        error = ((x * (max - min) + min)*2) - ((self.output * (max - min) + min)*2)
        self.delta = -error * self.derivative()

    def getDelta(self):
        return self.delta

    def sigmoid(self,x):
        self.input = x
        self.output = 1/(1 + np.exp(-x))

    def derivative(self):
        return (1/(1 + np.exp(-self.output))) * (1- 1/(1 + np.exp(-self.output)))

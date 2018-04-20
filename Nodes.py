import numpy as np
from random import *
from math import *

class InputNode(object):
    def __init__(self):
        self.input = 0.0

    def setInput(self,input):
        self.input = input

    def getInput(self):
        return self.input

    def getOutput(self):
        return self.input

class BiasNode(object):
    def __init__(self):
        self.output = 1.0

    def getOutput(self):
        return self.output

    def getInput(self):
        return self.output

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

    def sigmoid(self):
        self.output = 1/(1 + np.exp(-self.input))

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

    def setDelta(self,x):
        self.delta = (self.output - 2*x) * self.derivative()

    def getDelta(self):
        return self.delta

    def sigmoid(self):
        self.output = 1/(1 + np.exp(-self.input))

    def derivative(self):
        return (1/(1 + np.exp(-self.output))) * (1-1/(1 + np.exp(-self.output)))

#   INPUT       HIDDEN      OUTPUT
#
#               |----|
#               | h1 |
#               |----|
#
#   |----|      |----|
# --| x1 |      | h2 |
#   |----|      |----|      |-----|
#                           | out |----
#   |----|      |----|      |-----|
# --| x2 |      | h3 |
#   |----|      |----|
#
#       |----|      |----|
#       | b1 |      | b2 |
#       |----|      |----|
#
#

# WEIGHTS OF NETWORK
#           h1      h2      h3
#   x1      0.32    0.65    0.77
#   x2      0.41    0.54    0.79
#   b1      0.19    0.38    0.55
#
#           out
#   h1      0.64
#   h2      0.53
#   h3      0.22
#   b2      0.34
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class NNetwork(object):
    LEARNING_RATE = 0.2
    MOMENTUM_RATE = 0.3 # Near Optimum Values 7 * 8

    def __init__(self):
        self.input = 2
        self.hidNode = 3
        self.output = 1
        self.hidLayer = 1
        # self.hid = np.empty([2, self.hidNode], dtype=float)
        # self.out = np.empty([2, self.output], dtype=float)
        self.hiddenLayerWithBias = np.empty(4,dtype=float)
        self.previousDeltaW1 = np.zeros((3,3),dtype=float)
        self.previousDeltaW2 = np.zeros((4,1),dtype=float)

        self.weights = np.empty([self.hidLayer + 1], dtype=np.matrix)
        self.weights[0] = np.array([[0.32, 0.65, 0.77], [0.41, 0.54, 0.79], [0.19 ,0.38 ,0.55]],dtype=float)
        self.weights[1] = np.array([[0.64] ,[0.53] ,[0.22] ,[0.34]], dtype=float)
        # self.weights[0] = np.random.rand(3,3)
        # self.weights[1] = np.random.rand(4,1)
        # print(self.weights)

    def sigmoid(self, x, de=False):
        if (de == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def forward(self, _inputs):
        # print(self.hiddenLayerWithBias)
        self.netHidden = np.dot(_inputs, self.weights[0])
        self.outHidden = self.sigmoid(self.netHidden)
        # print(self.outHidden)
        self.hiddenLayerWithBias = np.append(self.outHidden, 1)
        # print(self.hiddenLayerWithBias)
        self.netOut = np.dot(self.hiddenLayerWithBias, self.weights[1])
        self.out = self.sigmoid(self.netOut)
        # print(self.out)
        return self.out

    def backprop(self, _inputs, _outputKnown, _outFound):
        _inputs = np.array(_inputs)
        self.errorRate = _outputKnown - _outFound
        self.outGrad = self.errorRate * self.sigmoid(_outFound, True)
        print("")

        self.hidErrorRate = self.outGrad.dot(self.weights[1][:-1].T)
        self.hidGrad = self.hidErrorRate * self.sigmoid(self.outHidden,True)



        self.weights[0] += ((_inputs.reshape(3,1).dot(self.hidGrad.reshape(1,3))) * self.LEARNING_RATE) + (self.MOMENTUM_RATE * self.previousDeltaW1)
        self.previousDeltaW1 = ((_inputs.reshape(3,1).dot(self.hidGrad.reshape(1,3))) * self.LEARNING_RATE) + (self.MOMENTUM_RATE * self.previousDeltaW1)
        self.weights[1] += ((self.hiddenLayerWithBias.reshape(4,1).dot(self.outGrad).reshape(1,4).T) * self.LEARNING_RATE) + (self.MOMENTUM_RATE * self.previousDeltaW2)
        self.previousDeltaW2 = ((self.hiddenLayerWithBias.reshape(4,1).dot(self.outGrad).reshape(1,4).T) * self.LEARNING_RATE) + (self.MOMENTUM_RATE * self.previousDeltaW2)
        

    def train(self,_inputs,_outputs):
        _outFound = self.forward(_inputs)
        self.backprop(_inputs,_outputs,_outFound)

    def printSummary(self):
        print("Weights")
        print(self.weights[0])
        print(self.weights[1])


EPOCH = 10000
TRAINING_SIZE = 70
NORM_DIV = 10
# CREATING DATA ARRAY
data = np.zeros((100, 3), dtype=int)

for x in range(1, 11):
    for y in range(1, 11):
        data[(x - 1) + (y - 1) * 10][0] = y
        data[(x - 1) + (y - 1) * 10][1] = x
        data[(x - 1) + (y - 1) * 10][2] = x * y

# CREATING DATA ARRAY NORMALIZED
dataNor = np.zeros((100, 3), dtype=float)

for x in range(1, 11):
    for y in range(1, 11):
        dataNor[(x - 1) + (y - 1) * 10][0] = y / NORM_DIV
        dataNor[(x - 1) + (y - 1) * 10][1] = x / NORM_DIV
        dataNor[(x - 1) + (y - 1) * 10][2] = (x / NORM_DIV) * (y / NORM_DIV)

np.random.shuffle(dataNor)

NN = NNetwork()

dataBiased = np.full((100, 3), 1, dtype=float)
dataBiased[:, :-1] = dataNor.T[:2].T
outputs = dataNor[:,2]
inputTrain = dataBiased[:TRAINING_SIZE]
inputTest = dataBiased[TRAINING_SIZE:]
outputTrain = outputs[:TRAINING_SIZE]
outputTest = outputs[TRAINING_SIZE:]

np.random.shuffle(inputTrain)

for e in range(EPOCH):
    for x in range(TRAINING_SIZE):
        NN.train(inputTrain[x],outputTrain[x])


print("Enter 2 Multiplication Values")
inputKeyb = [float(i) for i in input().split()]
while inputKeyb[0] != 0 and inputKeyb[1] != 0:
    print(inputKeyb[0],inputKeyb[1])
    print((NN.forward([inputKeyb[0]/NORM_DIV , inputKeyb[1]/NORM_DIV,1]))*NORM_DIV*NORM_DIV)
    print("Enter 2 Multiplication Values")
    inputKeyb = [float(i) for i in input().split()]

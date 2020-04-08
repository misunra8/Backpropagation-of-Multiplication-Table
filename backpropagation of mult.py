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
        # print(self.errorRate)
        self.outGrad = self.errorRate * self.sigmoid(_outFound, True)
        # print(self.outGrad)
        print("")

        self.hidErrorRate = self.outGrad.dot(self.weights[1][:-1].T)
        print(self.hidErrorRate)
        self.hidGrad = self.hidErrorRate * self.sigmoid(self.outHidden,True)


        # print(self.hiddenLayerWithBias.reshape(4,1))
        # #
        # print('')
        # print(self.weights[1])
        # print(self.hiddenLayerWithBias.reshape(4,1))
        # print((self.hiddenLayerWithBias.reshape(4,1).dot(self.outGrad).reshape(1,4)).T)


        self.weights[0] += ((_inputs.reshape(3,1).dot(self.hidGrad.reshape(1,3))) * self.LEARNING_RATE) + (self.MOMENTUM_RATE * self.previousDeltaW1)
        # print(self.weights[0])
        self.previousDeltaW1 = ((_inputs.reshape(3,1).dot(self.hidGrad.reshape(1,3))) * self.LEARNING_RATE) + (self.MOMENTUM_RATE * self.previousDeltaW1)
        # print(self.previousDeltaW1)
        self.weights[1] += ((self.hiddenLayerWithBias.reshape(4,1).dot(self.outGrad).reshape(1,4).T) * self.LEARNING_RATE) + (self.MOMENTUM_RATE * self.previousDeltaW2)
        # print(self.weights[1])
        self.previousDeltaW2 = ((self.hiddenLayerWithBias.reshape(4,1).dot(self.outGrad).reshape(1,4).T) * self.LEARNING_RATE) + (self.MOMENTUM_RATE * self.previousDeltaW2)
        #
        # print(self.previousDeltaW2)
        # print('')
        # print(self.weights[0])
        # print('')
        # print(self.weights[1])
        # # print(self.weights[0][0][0])
        # print(self.errorRate)

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

# print(NN.forward([0.2,0.3,1]))
# NN.backprop([0.2,0.3,1],0.06,NN.forward([0.2,0.3,1]))
for e in range(EPOCH):
    for x in range(TRAINING_SIZE):
        NN.train(inputTrain[x],outputTrain[x])

# # NN.printSummary()
# z=np.zeros(30,dtype=float)
# a=np.zeros(30,dtype=int)
# for x in range(100 - TRAINING_SIZE):
#     z[x] = np.round(NN.forward(inputTest[x]),2)*100
#     a[x] = outputTest[x]*100
#     # print(np.round(NN.forward(inputTest[x]),2)*100,outputTest[x]*100)
#
# for e in range(4*EPOCH):
#     for x in range(TRAINING_SIZE):
#         NN.train(inputTrain[x],outputTrain[x])
#
# # NN.printSummary()
# z1=np.zeros(30,dtype=float)
# for x in range(100 - TRAINING_SIZE):
#     z1[x] = np.round(NN.forward(inputTest[x]),2)*100
#
# for e in range(5*EPOCH):
#     for x in range(TRAINING_SIZE):
#         NN.train(inputTrain[x],outputTrain[x])
#
# # NN.printSummary()
# z2=np.zeros(30,dtype=float)
# for x in range(100 - TRAINING_SIZE):
#     z2[x] = np.round(NN.forward(inputTest[x]),2)*100

print("Enter 2 Multiplication Values")
inputKeyb = [float(i) for i in input().split()]
while inputKeyb[0] != 0 and inputKeyb[1] != 0:
    print(inputKeyb[0],inputKeyb[1])
    print((NN.forward([inputKeyb[0]/NORM_DIV , inputKeyb[1]/NORM_DIV,1]))*NORM_DIV*NORM_DIV)
    print("Enter 2 Multiplication Values")
    inputKeyb = [float(i) for i in input().split()]
#
#
# gs = gridspec.GridSpec(2,2)
# plt.figure()
#
#
# ax00 =plt.subplot(gs[0,0])
# plt.plot(np.sort(a), np.sort(a),label="Expected Data",linewidth=2,color='black')
# plt.plot(np.sort(a[:10]), np.sort(z[:10]),label="1 EPOCH",linewidth=2,color='green')
# plt.plot(np.sort(a[:10]), np.sort(z1[:10]),label="5 EPOCH",linewidth=2,color='blue')
# plt.plot(np.sort(a[:10]), np.sort(z2[:10]),label="10 EPOCH",linewidth=2,color='red')
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
# plt.legend(loc=4)
# plt.title('Between 70-80')
#
# ax01 = plt.subplot(gs[0,1])
# plt.plot(np.sort(a), np.sort(a),label="Expected Data",linewidth=2,color='black')
# plt.plot(np.sort(a[10:20]), np.sort(z[10:20]),label="1 EPOCH",linewidth=2,color='green')
# plt.plot(np.sort(a[10:20]), np.sort(z1[10:20]),label="5 EPOCH",linewidth=2,color='blue')
# plt.plot(np.sort(a[10:20]), np.sort(z2[10:20]),label="10 EPOCH",linewidth=2,color='red')
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
# plt.legend(loc=4)
# plt.title('Between 80-90')
#
# ax10 = plt.subplot(gs[1,0])
# plt.plot(np.sort(a), np.sort(a),label="Expected Data",linewidth=2,color='black')
# plt.plot(np.sort(a[20:30]), np.sort(z[20:30]),label="1 EPOCH",linewidth=2,color='green')
# plt.plot(np.sort(a[20:30]), np.sort(z1[20:30]),label="5 EPOCH",linewidth=2,color='blue')
# plt.plot(np.sort(a[20:30]), np.sort(z2[20:30]),label="10 EPOCH",linewidth=2,color='red')
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
# plt.legend(loc=4)
# plt.title('Between 90-100')
#
# ax11 = plt.subplot(gs[1,1])
# plt.plot(np.sort(a), np.sort(a),label="Expected Data",linewidth=2,color='black')
# plt.plot(np.sort(a), np.sort(z),label="1 EPOCH",linewidth=2,color='green')
# plt.plot(np.sort(a), np.sort(z1),label="5 EPOCH",linewidth=2,color='blue')
# plt.plot(np.sort(a), np.sort(z2),label="10 EPOCH",linewidth=2,color='red')
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
# plt.legend(loc=4)
#
# # giving a title to my graph
# plt.title('Between 70-100 - Sorted')
#
# # function to show the plot
# plt.show()
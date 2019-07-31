import numpy as np

#X = mammal (weight, is_carnivore)
#Y = mammal population
xA = ([350, 1], [600, 1], [1200, 0], [800, 0], [19, 1], [40, 0], [2300, 0])
yA = ([20], [22], [500], [750], [3000], [9000])
#xA = ([350, 1], [600, 1], [1200, 0], [800, 0])
#yA = ([20], [22], [500])
np.set_printoptions(suppress=True)
xAll = np.array(xA, dtype=float)
yAll = np.array(yA, dtype=float)

#scale units
xmax = np.amax(xAll, axis=0)
ymax = np.amax(yAll, axis=0)
xAll = xAll/xmax
yAll = yAll/ymax

#split data
X = np.split(xAll, [6])[0]
xPredicted = np.split(xAll, [6])[1]


class Neural_Network:
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1-s)

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted * xmax[0]) )
        print ("Output: \n" + str(self.forward(xPredicted)* ymax))

NN = Neural_Network()
for i in range(100000):
    NN.train(X, yAll)
print ("Input: \n" + str(X*xmax))
print ("Actual Output: \n" + str(yAll*ymax))
print ("Predicted Output: \n" + str(NN.forward(X)*ymax))
print ("Loss: \n" + str(np.mean(np.square(yAll - NN.forward(X)))))
print ("\n")
NN.predict()

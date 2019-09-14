#Artificial Neural Network predicting the scaled sound pressure level (in decibels) of an airfoil based on
#1. Frequency, in Hertzs. 
#2. Angle of attack, in degrees. 
#3. Chord length, in meters. 
#4. Free-stream velocity, in meters per second. 
#5. Suction side displacement thickness, in meters.

#from NASA data set
#found at Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.



from math import ceil

import pandas as pd
import numpy as np

import data_parse
from data_parse import get_sample

np.set_printoptions(suppress=True)
print("~~~ANN given details of an airfoil, finds the sound pressure level~~~")

trainX, trainY, testX, testY, maxX, maxY = data_parse.main()

class Neural_Network:
    def __init__(self):
        self.inputSize = 5
        self.outputSize = 1
        self.hiddenSize1 = 8
        self.hiddenSize2 = 8

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize1)
        self.W2 = np.random.randn(self.hiddenSize1, self.hiddenSize2)
        self.W3 = np.random.randn(self.hiddenSize2, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        self.z4 = self.sigmoid(self.z3)
        self.z5 = np.dot(self.z4, self.W3)
        o = self.sigmoid(self.z5)
        return o

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1-s)

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        self.z4_error = self.o_delta.dot(self.W3.T)
        self.z4_delta = self.z4_error*self.sigmoidPrime(self.z4)
        self.z2_error = self.z4_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.z4_delta)
        self.W3 += self.z4.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        print ("\n\nPredicted test data based on trained weights: ")
        a = testY*maxY
        b = np.around(self.forward(testX)*maxY, 3)
        c = np.column_stack((a, b))
        print("Actual Output vs. Predicted Output: (Db) \n" + str(c))
        print("Loss: \n" + str(np.mean(np.square(testY-self.forward(testX)))))
        #print ("Actual Output: \n" + str(testY*maxY))
        #print ("Predicted Output: \n" + str(np.around(self.forward(testX)*maxY, 3)))

NN = Neural_Network()
l = trainX.shape[0]
n = 32
print("Training...")
for j in range(5000):
    index = 0
    for i in range(ceil(l/n)):
        sampleX = get_sample(trainX, index, n)[0]
        sampleY, index = get_sample(trainY, index, n)
        NN.train(sampleX, sampleY)
print("Done!")
print("Input: Frequency (Hz), Angle (Degrees), Chord Length (m), Wind Velocity (m/s), Thickness (m) \n " + str(trainX*maxX))
print ("Actual Output: (Db) \n" + str(trainY*maxY))
print ("Predicted Output: (Db) \n" + str(np.around(NN.forward(trainX)*maxY, 3)))
print ("Loss: \n" + str(np.mean(np.square(trainY - NN.forward(trainX)))))
print ("\n")
NN.predict()

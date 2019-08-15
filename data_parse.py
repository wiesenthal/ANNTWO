import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions( linewidth=100)
def parse_data():
    df = pd.read_csv('airfoil_self_noise.csv')

    #shuffling the values randomly
    df = df.sample(frac=1).reset_index(drop=True)
    return(df.to_numpy())

def scale_data(data):
    max = np.amax(data, axis = 0)
    return (data / max), max

def split_train_test(data, fraction_testing):
    #returns training data and testing data
    length = data.shape[0]
    fraction_training = 1  -fraction_testing
    split_index = int(length * fraction_training)
    train, test = np.split(data, [split_index])
    return (train, test)

def split_input_output(data):
    #returns arrays of input data and the expected outputs
    inp, out = np.hsplit(data, [5])
    return (inp, out)

def get_sample(data, start=0, n=32):
    #returns the sample, and the new index
    l = data.shape[0]
    if l - start > 32:
        subset = data[start:start+n]
    else:
        subset = data[start:l]
    return(subset, start + n)

def main():
    df = parse_data()
    df, max = scale_data(df)
    maxX, maxY = split_input_output(max)
    train_data, test_data = split_train_test(df, .1)
    train_X, train_Y = split_input_output(train_data)
    test_X, test_Y = split_input_output(test_data)
    return (train_X, train_Y, test_X, test_Y, maxX, maxY)
main()
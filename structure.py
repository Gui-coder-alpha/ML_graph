import numpy as np
import algoritm

Features = np.array([[1],
                     [2],
                     [7]])
Target = Features**3

Learning_rate = 0.01
Epochs = 1000

Activate_function = algoritm.Universal_ML()

def Machine_learning(Features, Target, Learning_rate, Epochs):
    Features_ReLU = Activate_function.Foward_function(Features)
    print(Features_ReLU)
    for i in range(Epochs):
        oi = 1

printing = Machine_learning(Features, Target, Learning_rate, Epochs)
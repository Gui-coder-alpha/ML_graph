import numpy as np
import algoritm

Features = np.array([[1],
                     [2],
                     [7]])
Target = Features**3

Learning_rate = 0.01
Epochs = 1

Activate_function = algoritm.Universal_ML()

def Machine_learning(Features, Target, Learning_rate, Epochs):
    Features_ReLU, Features_Final = Activate_function.Foward_function(Features)
    #print(Features_Final)
    #print("//////////////")
    Features_backward_principal = Activate_function.Backward_function(Features_Final, Target, Learning_rate, Features_ReLU)
    #print(Features_backward_principal)
    for i in range(Epochs):
        oi = 1

printing = Machine_learning(Features, Target, Learning_rate, Epochs)
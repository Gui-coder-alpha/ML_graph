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
    for i in range(Epochs):
        Features_ReLU, Features_Final = Activate_function.Foward_function(Features)
        Features_backward_principal = Activate_function.Backward_function(Features_Final, Target, Features, Features_ReLU)
        

        Cost_value = Activate_function.Cost_function(Features_Final, Target)
        print(Features_backward_principal)
        
printing = Machine_learning(Features, Target, Learning_rate, Epochs)
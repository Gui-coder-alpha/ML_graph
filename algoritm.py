import numpy as np

class Universal_ML:
    def __init__(self):
        self.weights_1 = np.random.randn(1,5)
        self.weights_2 = np.random.randn(5,1)

        self.bias_1 = np.ones((1,5))
        self.bias_2 = np.ones((1,1))

    def activation(self, Features):
        return np.maximum(0, Features)
    
    def sim(self):
        return "test"

    def Foward_function(self, Features):
        Features_foward_1 = (Features @ self.weights_1) + self.bias_1
        Features_Z = self.activation(Features_foward_1)

        Features_transformed = (Features_Z @ self.weights_2) + self.bias_2
        return Features_Z, Features_transformed

    def Backward_function(self, Features_transformed, Target, Learning_rate, Features_with_activation):
        Features_foward_final = (1/Learning_rate) * (Features_transformed - Target)
        print(Features_foward_final)
        Weights_foward_final = np.transpose(Features_with_activation) @ Features_foward_final
        print(Weights_foward_final)

        Features_foward_2 = (Features_foward_final @ np.transpose(Weights_foward_final)) * Features_with_activation
        print("bbbbbbbbbbbbbbbbb")
        print(Features_foward_2)
        return Features_foward_2
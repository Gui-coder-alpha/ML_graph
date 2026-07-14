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
        m = Features_transformed.shape[0]
        Features_foward_final = (1/m) * (Features_transformed - Target)
        Weights_foward_final = np.transpose(Features_with_activation) @ Features_foward_final
        Bias_gradient_final = np.sum(Features_foward_final, axis=0, keepdims=True)

        Relu_derivative = (Features_with_activation > 0).astype(float)
        Features_foward_2 = (Features_foward_final @ np.transpose(Weights_foward_final)) * Relu_derivative
        Bias_gradient_2 = np.sum(Features_foward_2, axis=0, keepdims=True)

        return Features_foward_2
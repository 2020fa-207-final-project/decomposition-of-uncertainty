import numpy as np

class bnn_lv:
    def __init__(self, architecture, out_covariance=None, random = None, weights = None):
        '''

        '''
        self.layers = {'input_n' : architecture['input_n'],
                       'output_n' : architecture['output_n'],
                       'hidden_layers_n' : len(architecture['hidden_layers']),
                       'hidden_layers_shape' : architecture['hidden_layers'],
                       'biases' : architecture['biases'],
                       'activation' : architecture['activation']}   
        self.D = self._calculate_network_size()
        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))
        self.out_cov = out_covariance 
        
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            assert type(weights) == np.ndarray, "Error: Weights must be a numpy array"
            assert len(weights.shape) == 2, "Error: Weights must be specified as a 2D numpy array"
            assert weights.shape[1] == self.D, f"Error: Weight dimensions must match the shape specified in the architecture (num_weights={self.D})"
            self.weights = weights

    def _calculate_network_size(self):
        '''
        Calculate the number of weights required to represent the specific architecture
        '''
        D = 0
        for i in range(self.layers['hidden_layers_n']):
            # First layer so use input nodes
            if i == 0:
                D += (self.layers['input_n'] * self.layers['hidden_layers_shape'][i])
            else:
                D += self.layers['hidden_layers_shape'][i-1] * self.layers['hidden_layers_shape'][i]
            if self.layers['biases'][i]:
                D += self.layers['hidden_layers_shape'][i]
        # Final output layer
        D += self.layers['hidden_layers_shape'][-1] * self.layers['output_n']
        if self.layers['biases'][-1]:
                D += self.layers['output_n']

        return D

    def set_weights(self, weights):
        '''
        Manually set the weights of the Neural Network
        '''
        assert type(weights) == np.ndarray, "Error: Weights must be a numpy array"
        assert len(weights.shape) == 2, "Error: Weights must be specified as a 2D numpy array"
        assert weights.shape[1] == self.D, f"Error: Weight dimensions must match the shape specified in the architecture (num_weights={self.D})"
        self.weights = weights
        return

    @staticmethod
    def relu(x):
        return np.max(np.zeros(shape=x.shape), x)

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def additive_noise(cov):
        return np.random.normal(0, np.sqrt(cov))

    def fit(self, X, y):
        pass

    def forward(self, X):
        '''
        Perform a forward pass through the network from input to output
        '''
        # Check X dimensions
        if len(X.shape) == 2:
            assert X.shape[0] == D_in
            X = X.reshape((1, D_in, -1))
        else:
            assert X.shape[1] == D_in

        weights = self.weights.T
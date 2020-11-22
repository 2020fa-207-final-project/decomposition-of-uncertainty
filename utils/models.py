import numpy as np

class neural_network:
    def __init__(self, architecture, random = None, weights = None):
        '''

        '''
         # Layer Assertions
        assert len(architecture['biases']) == len(architecture['activations']), "Error: Mismatch in layer dimensions - biases vs activations"
        assert (len(architecture['biases'])-1) == len(architecture['hidden_layers']), "Error: Mismatch in layer dimensions - must be 1 fewer hidden layer than biases/activations"
        for bias in architecture['biases']:
            assert bias == bool or bias == 1 or bias == 0, "Error: biases must be bool or int[0,1]"
        for act in architecture['activations']:
            assert act in ['relu', 'linear'], "Error: Only 'relu' and 'linear' activations have been implemented"

        # Combine output and hidden layers together to match N weight layers
        all_layers = architecture['hidden_layers'].copy()
        all_layers.append(architecture['output_n'])

        self.layers = {'input_n' : architecture['input_n'],
                       'output_n' : architecture['output_n'],
                       'hidden_layers_shape' : architecture['hidden_layers'],
                       'all_layers_shape' : all_layers,
                       'biases' : architecture['biases'],
                       'activations' : architecture['activations']}   
        self.D, self._layers_D = self._calculate_network_size()
        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))
                
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            assert type(weights) == np.ndarray, "Error: Weights must be a numpy array"
            assert len(weights.shape) == 2, "Error: Weights must be specified as a 2D numpy array"
            assert weights.shape[1] == self.D, f"Error: Weight dimensions must match the shape specified in the architecture ((1,{self.D}))"
            self.weights = weights

    def _calculate_network_size(self):
        '''
        Calculate the number of weights required to represent the specific architecture
        '''
        D = 0
        layers_D = []
        for i in range(len(self.layers['hidden_layers_shape'])):
            # First layer so use input nodes
            if i == 0:
                D += (self.layers['input_n'] * self.layers['hidden_layers_shape'][i])
            else:
                D += self.layers['hidden_layers_shape'][i-1] * self.layers['hidden_layers_shape'][i]
            if self.layers['biases'][i]:
                D += self.layers['hidden_layers_shape'][i]
            layers_D.append(D - int(np.sum(layers_D)))
        # Final output layer
        D += self.layers['hidden_layers_shape'][-1] * self.layers['output_n']
        if self.layers['biases'][-1]:
                D += self.layers['output_n']
        layers_D.append(D - int(np.sum(layers_D)))

        return D, layers_D

    def set_weights(self, weights):
        '''
        Manually set the weights of the Neural Network
        '''
        assert type(weights) == np.ndarray, "Error: Weights must be a numpy array"
        assert len(weights.shape) == 2, "Error: Weights must be specified as a 2D numpy array"
        assert weights.shape[1] == self.D, f"Error: Weight dimensions must match the shape specified in the architecture (num_weights={self.D})"
        self.weights = weights
        return

    def get_weights(self):
        '''
        Simple wrapper to return weights
        '''
        return self.weights

    @staticmethod
    def relu(x):
        return np.maximum(np.zeros(shape=x.shape), x)

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
            assert X.shape[0] == self.layers['input_n'], f"Error: if 2D, X[0] must match input dimension ({self.layers['input_n']})"
            X = X.reshape((1, self.layers['input_n'], -1))
        elif len(X.shape) == 3:
            assert X.shape[1] == self.layers['input_n'], f"Error: If 3D, X[1] must match input dimension ({self.layers['input_n']})"
        else:
            raise ValueError("Error: Data input shape must be either 2D or 3D")

        # Copy data to values to iterate through the network
        values_in = X.copy()

        weights = self.weights.T

        for i, _ in enumerate(self._layers_D):
            
            if i == 0:
                layer_weights = weights[ 0 : self._layers_D[i] ]
            else:
                layer_weights = weights[ np.sum(self._layers_D[0:i]) : (np.sum(self._layers_D[0:i])+self._layers_D[i]) ]

            # If Biases are enabled for this layer split weights from biases, else set biases to zero
            if self.layers['biases'][i]:
                W = layer_weights[ 0 : -self.layers['all_layers_shape'][i] ]
                b = layer_weights[ -self.layers['all_layers_shape'][i] : ]
            else:
                W = layer_weights
                b = np.zeros(self.layers['all_layers_shape'][i])

            # Shape and calculate the pre activation output
            W = W.T.reshape((-1, self.layers['all_layers_shape'][i], values_in.shape[1]))
            b = b.T.reshape((-1, self.layers['all_layers_shape'][i], 1))
            
            pre_activation = np.matmul(W, values_in) + b
            
            if self.layers['activations'][i] == 'relu':
                outputs = self.relu(pre_activation)
            elif self.layers['activations'][i] == 'linear':
                outputs = self.identity(pre_activation)
            else:
                raise ValueError(f"Error: Unexpected activation type - {self.layers['activations'][i]}")

            # Update for next iteration
            values_in = outputs

        return outputs


class BNN_LV(neural_network):
    def __init__(self, architecture, random = None, weights = None):
        # Add a noise input.
        architecture = architecture.copy()
        architecture['input_n'] += 1
        # Get standard deviation of noise:
        self.gamma = architecture['gamma']
        # Build a neural_network:
        super().__init__(architecture, random=random, weights=weights)
    
    def add_input_noise(self,X):
        Z = self.random.randn(*X.shape[:-1],1) * self.gamma
        return np.append(X,Z, axis=-1)
    
    def add_output_noise(self,Y):
        e = self.random.randn(*Y.shape) * self.gamma
        return Y + e
        
    def forward(self, X):
        X_ = self.add_input_noise(X)
        Y_ = super().forward(X_)
        Y = self.add_output_noise(Y_)
        return Y

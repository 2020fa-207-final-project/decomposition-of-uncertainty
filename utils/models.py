import numpy as np

class BNN:
    """
    A nerual network with M inputs and K outputs with D weights.
    The weights are represented as an S-by-D matrix,
    where D is the total number of weights in the network
    and S is typically 1 but can be an abitrary number of models
    (e.g. when doing a forward pass on a family of models 
    sampled from a posterion during Bayesian modeling).
    """

    def __init__(self, architecture, seed = None, weights = None):
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
        self._D, self._layers_D = self._calculate_network_size()
        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))
        
        self.seed = seed
        self.random = np.random.RandomState(seed)

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            assert type(weights) == np.ndarray, "Error: Weights must be a numpy array"
            assert len(weights.shape) == 2, "Error: Weights must be specified as a 2D numpy array"
            assert weights.shape[1] == self.D, f"Error: Weight dimensions must match the shape specified in the architecture ((1,{self.D}))"
            self.weights = weights

    @property
    def M(self):
        """ Input dimensions. """
        return self.layers['input_n']

    @property
    def K(self):
        """ Output dimensions. """
        return self.layers['output_n']

    @property
    def D(self):
        """ Total number of weights. """
        return self._D

    @property
    def S(self):
        """ Number of different models for which the weights are stored. """
        return self.weights.shape[0]

    @property
    def H(self):
        """ Number of hidden layers (not counting input or output). """
        return len(self.layers['hidden_layers_shape'])

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
        if len(self.layers['hidden_layers_shape'])>0:
            D += self.layers['hidden_layers_shape'][-1] * self.layers['output_n']
        else:  # Special case if there are no hidden layers:
            D += self.layers['input_n'] * self.layers['output_n']
        if self.layers['biases'][-1]:
            D += self.layers['output_n']
        layers_D.append(D - int(np.sum(layers_D)))

        return D, layers_D

    def stack_weights(self, W_layers):
        """
        Takes a list of (S-by-)IN-by-OUT tensors
        storing the weights for each layer in a model
        (or in each model from a set of S models)
        and returns an S-by-D matrix of the weights
        for the whole network (where S is 1 if there is only 1 model).
        IN and OUT are the number of input and output nodes
        for the given layer.
        """
        
        # Store results:
        W = []

        # Check numer of layers:
        n_layers = len(self.layers['all_layers_shape'])
        assert len(W_layers)==n_layers, f"Received {len(W_layers)} layers but expected {n_layers}."
        
        # Get dimensions:
        S = W_layers[0].shape[0] if len(W_layers[0].shape)==3 else 1
        
        # Loop through layers:
        for i,W_layer in enumerate(W_layers):
            
            # Get expected layer sizes:
            if i==0:
                source_size = self.layers['input_n']
            else:
                source_size = self.layers['all_layers_shape'][i-1]
            source_size += self.layers['biases'][i]
            target_size = self.layers['all_layers_shape'][i]
            
            # Check dimensions:
            if len(W_layer.shape)==2:
                W_layer = W_layer.reshape(1,*W_layer.shape)
            elif len(W_layer.shape)==3:
                assert W_layer.shape[0]==S, f"[layer {i+1}] Encounted set of {W_layer.shape[0]} weights but expected {S}."
            else:
                raise ValueError("W should be (S-by-)IN-by-OUT, i.e. (3 or) 2 dimenisions.")
            assert W_layer.shape[-2]==source_size, f"[layer {i+1}] Rows ({W_layer.shape[-2]}) should correspond to input nodes ({source_size})."
            assert W_layer.shape[-1]==target_size, f"[layer {i+1}] Columns ({W_layer.shape[-1]}) should correspond to output nodes ({target_size})."
            
            # Reshape and add to stack:
            W_layer = W_layer.reshape(S,source_size*target_size)
            W.append(W_layer)
        
        W = np.hstack(W)
        assert W.shape == (S,self.D), f"Found {W.shape[1]} weights but expected {self.D}."
        
        return W

    def unstack_weights(self, W):
        """
        Creates a list of the weights for each layer.
        W may be a row-vector of weights, or a matrix
        where each row corresponds to a different
        specification of the model (e.g. we may have S
        sets of weights, representing S different BNNs,
        given as a S-by-D matrix.)
        """
        
        if len(W.shape)==1:
            W = W.reshape(1,-1)
        elif len(W.shape)==2:
            assert W.shape[1] == self.D, "Second dimension must match number of weights."
        else:
            raise ValueError("W should be S-by-D or 1-by-D (i.e. max 2 dimenisions).")
        
        # Determine how many sets of weights we have:
        S = W.shape[0]
            
        # Get sizes of layers:
        source_sizes = [self.layers['input_n']] + self.layers['all_layers_shape'][:-1]
        target_sizes = self.layers['all_layers_shape']
        for i,bias in enumerate(self.layers['biases']):
            source_sizes[i] += bias  # 0 or 1
            
        # Loop through layers:
        W_layers = []
        cursor = 0  # Keep track row of W.
        for source_size, target_size in zip(source_sizes,target_sizes):
            # Get the chunk of weights corresponding to this layer:
            W_layer = W[ :, cursor:(cursor+source_size*target_size) ]
            # Reshape the chunk to stack of matrices (i.e. a 3d tensor)
            # Where each matrix has rows corresponding to source nodes
            # and columns corresponding to target nodes:
            W_layer = W_layer.reshape(S, source_size, target_size)
            # Add tensor to list:
            W_layers.append(W_layer)
            # Update cursor:
            cursor += source_size*target_size
            
        return W_layers

    def set_weights(self, weights):
        '''
        Manually set the weights of the Neural Network
        '''
        if isinstance(weights, list):
            weights = self.stack_weights(weights)
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

    def fit(self, X, y):
        pass

    def forward(self, X, weights=None):
        '''
        Perform a forward pass through the network from input to output.
        Requires the last dimension of X to be the number of inputs (i.e. features).
        X is generally expected by be an N-by-M matrix, but may in fact have
        an arbitrary dimensions instead of N (as long as M is the last dimension).
        The K outputs will be along the last dimension (with prior dimensions
        matching those of X, i.e. the typical output will be an N-by-K matrix).
        '''
        # Check X dimensions:
        M = self.layers['input_n']  # Number of features.
        Y_shape = tuple((*X.shape[:-1], self.layers['output_n']))  # Determine shape of output.
        if len(X.shape) < 2:
            raise ValueError("X should be (at least) 2 dimensional.")
        assert X.shape[-1]==M, f"Last dimenion of X is {X.shape[-1]} bus should correspond to {M} inputs (i.e. features)"
        
        # Get weights for each layer (as tensors):
        weights = self.weights if weights is None else weights
        W_layers = self.unstack_weights(W=weights)

        # Copy data to values to iterate through the network
        values_in = X.copy()

        # Loop through layers:
        #   Reminder: W_layer is an S-by-IN-by-OUT tensor of the weights for S models,
        #   with layer inputs on the rows and layer outputs on the columns.
        for i,weights in enumerate(W_layers):
            
            # If there is a bias, postpend a column of ones to each matrix:
            bias = self.layers['biases'][i]
            if bias==1:
                bias_features = np.ones((*values_in.shape[:-1],1))
                values_in = np.append(values_in,bias_features, axis=-1)

            # Calculate pre-activation values:
            # print("values_in",values_in.shape)
            # print("weights",weights.shape)
            values_pre = np.matmul( values_in, weights )

            # Apply activation fucntion:
            if self.layers['activations'][i] == 'relu':
                values_out = self.relu(values_pre)
            elif self.layers['activations'][i] == 'linear':
                values_out = self.identity(values_pre)
            else:
                raise ValueError(f"Error: Unexpected activation type - {self.layers['activations'][i]}")

            # Pass output values as input to next layer:
            values_in = values_out

        Y = values_out.reshape(Y_shape)
        
        return Y


class BNN_LV(BNN):
    """
    Implementation of Bayesian Neural Network with Latent Variables
    https://arxiv.org/pdf/1605.07127.pdf  Section 2.2.
    """

    def __init__(self, architecture, seed = None, weights = None):
        """
        Uses the same architecture as the BNN superclass,
        but expects additional keys:
            gamma : standard deviation of the input noise (inserted as a feature).
            sigma : list of standard deviations of the output noise
                (added in each dimension of the output).
                Expects a list, where each column corresponds to a dimension of Y.
        """
        # Add a noise input.
        architecture = architecture.copy()
        architecture['input_n'] += 1
        # Get standard deviation of noise:
        if 'gamma' not in architecture:
            raise KeyError("Make sure achitecture includes gamma (standard deviation of the input noise).")
        self.gamma = architecture['gamma']  # Scalar.
        if 'sigma' not in architecture:
            raise KeyError("Make sure achitecture includes sigma (list of standard deviations of the output noise).")
        self.sigma = architecture['sigma']  # List.
        # Check dimensions:
        try:
            self.gamma = float(self.gamma)
        except:
            raise ValueError("The standard deviation of the latent noise feature should be a scalar.")
        self.sigma = np.array([self.sigma]).flatten()  # Coerce to array.
        if len(self.sigma) != architecture['output_n']:
            raise ValueError(f"The standard deviation of the output noise ({len(self.sigma)}) should match the dimension of the output ({architecture['output_n']}).")
        # Build a neural_network:
        super().__init__(architecture, seed=seed, weights=weights)
    
    def add_input_noise(self,X):
        """
        Add a feature of input noise drawn from a Gaussian distribution
        with mean 0 and the standard deviation.
        """
        Z_shape = tuple((*X.shape[:-1],1))
        Z = self.random.normal(loc=0, scale=self.gamma, size=Z_shape)
        return np.append(X,Z, axis=-1)
    
    def add_output_noise(self,Y):
        """
        Corrupt output with additive noise drawn from a Gaussian distribution
        with mean 0 and a standard deviation specified for each dimension of the output.
        """
        Eps_shape = Y.shape
        Eps = self.random.normal(loc=0, scale=self.sigma, size=Eps_shape)
        return Y + Eps
        
    def forward(self, X):
        X_ = self.add_input_noise(X)
        Y_ = super().forward(X_)
        Y = self.add_output_noise(Y_)
        return Y

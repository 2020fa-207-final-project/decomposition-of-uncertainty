"""Tests for `utils` package."""
import numpy as np

from unittest import TestCase, main
from utils.models import neural_network

class exampleTests(TestCase):
    def test_basic(self):
        self.assertEqual(1, 1)

    def test_basic2(self):
        assert isinstance(1, int)

# Neural Network Tests
class nnTests(TestCase):
    def test_feedforward_nobias_all_relu_zeros(self):
        architecture = {'input_n':2, 
             'output_n':1, 
             'hidden_layers':[3,5],
             'biases' : [0,0,0],
             'activations' : ['relu', 'relu', 'relu']}
        nn = neural_network(architecture=architecture)
        data_in = np.array([0,0]).reshape(-1,1)
        results = nn.forward(data_in)
        self.assertEqual(results, np.array(0).reshape(1,1,1))
        assert results.shape == (1,1,1)

    def test_feedforward_nobias_all_linear_zeros(self):
        architecture = {'input_n':2, 
                'output_n':1, 
                'hidden_layers':[3,5],
                'biases' : [0,0,0],
                'activations' : ['linear', 'linear', 'linear']}
        nn = neural_network(architecture=architecture)
        data_in = np.array([0,0]).reshape(-1,1)
        results = nn.forward(data_in)
        self.assertEqual(results, np.array(0).reshape(1,1,1))
        assert results.shape == (1,1,1)

    def test_feedforward_bias_all_relu_zeros(self):
        architecture = {'input_n':2, 
             'output_n':1, 
             'hidden_layers':[3,5],
             'biases' : [1,1,1],
             'activations' : ['relu', 'relu', 'relu']}
        nn = neural_network(architecture=architecture)
        data_in = np.array([0,0]).reshape(-1,1)
        results = nn.forward(data_in)
        self.assertEqual(results, np.array(0).reshape(1,1,1))
        assert results.shape == (1,1,1)

    def test_feedforward_nobias_all_relu_ones(self):
        architecture = {'input_n':3, 
             'output_n':1, 
             'hidden_layers':[2,2,2],
             'biases' : [0,0,0,0],
             'activations' : ['relu', 'relu', 'relu', 'relu']}
        nn = neural_network(architecture=architecture)
        nn.set_weights(np.ones(shape=(1,nn.D)))
        data_in = np.array([1,1,1]).reshape(-1,1)
        results = nn.forward(data_in)
        self.assertEqual(results, np.array(24).reshape(1,1,1))
        assert results.shape == (1,1,1)

    def test_feedforward_bias_all_relu_ones(self):
        architecture = {'input_n':3, 
             'output_n':1, 
             'hidden_layers':[2,2,2],
             'biases' : [1,1,1,1],
             'activations' : ['relu', 'relu', 'relu', 'relu']}
        nn = neural_network(architecture=architecture)
        nn.set_weights(np.ones(shape=(1,nn.D)))
        data_in = np.array([1,1,1]).reshape(-1,1)
        results = nn.forward(data_in)
        self.assertEqual(results, np.array(39).reshape(1,1,1))
        assert results.shape == (1,1,1)

    def test_feedforward_bias_all_relu_negative_ones(self):
        architecture = {'input_n':3, 
             'output_n':1, 
             'hidden_layers':[2,2,2],
             'biases' : [1,1,1,1],
             'activations' : ['relu', 'relu', 'relu', 'relu']}
        nn = neural_network(architecture=architecture)
        nn.set_weights(np.ones(shape=(1,nn.D)))
        data_in = np.array([-1,-1,-1]).reshape(-1,1)
        results = nn.forward(data_in)
        self.assertEqual(results, np.array(7).reshape(1,1,1))
        assert results.shape == (1,1,1)

    def test_feedforward_overall_complex_config(self):
        architecture = {'input_n':2, 
             'output_n':2, 
             'hidden_layers':[2,3,2],
             'biases' : [1,0,0,1],
             'activations' : ['relu', 'relu', 'linear', 'linear']}
        nn = neural_network(architecture=architecture)
        # Manually specified weights - see diagram for details
        nn.set_weights(np.array([ 1,3,0.5,1,-2,1, 0,2,1,3,0,-1, 0.5,2,0,1,-2,2, 1,1,-1,2,1,1]).reshape(1,-1))
        data_in = np.array([4,-2]).reshape(-1,1)
        results = nn.forward(data_in)
        np.testing.assert_array_equal(results, np.array([4,-14]).reshape(1,2,1))
        assert results.shape == (1,2,1)


if __name__ == "__main__":
    main()

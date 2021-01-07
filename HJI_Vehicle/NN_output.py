'''
Run this script to train the NN. Loads the problem and configuration
according to examples/choose_problem.py.
'''

import numpy as np
# import torch
# import scipy.io
# import time

from HJI_Vehicle.utilities.other import int_input, load_NN
from HJI_Vehicle.examples.choose_problem import system, problem, config, time_dependent

if time_dependent:
    from HJI_Vehicle.utilities.neural_networks import HJB_network
    system += '/tspan'
else:
    from HJI_Vehicle.utilities.neural_networks import HJB_network_t0 as HJB_network
    system += '/t0'

# Loads neural network
# start_time = time.time()


def get_Q_value(X, t, U, theta, deltaT=0.05):
    theta1, theta2 = theta
    if theta1 == 5 and theta2 == 5:
        model_path = 'HJI_Vehicle/examples/' + system + '/V_model_na_na.mat'
        parameters, scaling = load_NN(model_path)
    if theta1 == 1 and theta2 == 1:
        model_path = 'HJI_Vehicle/examples/' + system + '/V_model_a_a.mat'
        parameters, scaling = load_NN(model_path)
    if theta1 == 5 and theta2 == 1:
        model_path = 'HJI_Vehicle/examples/' + system + '/V_model_a_na.mat'
        parameters, scaling = load_NN(model_path)
    model = HJB_network(problem, scaling, config, parameters)
    V1, V2 = model.Q_value(X, t, U, theta1, theta2, deltaT)
    return V1, V2

# model_running = time.time() - start_time
# print('Computation time: %.0f' % (model_running), 'sec')

if __name__ == '__main__':
    X = np.array([[15.], [18.], [15.], [18.]])
    t = np.array([[0]])
    U = np.array([[5.], [5.]])
    theta = (5, 5)
    Q1, Q2 = get_Q_value(X, t, U, theta)
    print(Q1)
    print(Q2)



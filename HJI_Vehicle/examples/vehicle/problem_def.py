import numpy as np
import torch
from HJI_Vehicle.examples.problem_def_template import config_prototype, problem_prototype

class config_NN (config_prototype):
    def __init__(self, N_states, time_dependent):
        N_layers = 3
        N_neurons = 128
        self.layers = self.build_layers(N_states,
                                        time_dependent,
                                        N_layers,
                                        N_neurons)

        self.random_seeds = {'train': None}

        self.ODE_solver = 'RK23'
        # Accuracy level of BVP data
        self.data_tol = 1e-3   # 1e-03
        # Max number of nodes to use in BVP
        self.max_nodes = 2500   # 800000
        # Time horizon
        self.t1 = 5.

        # Time subintervals to use in time marching
        Nt = 10  # 10
        self.tseq = np.linspace(0., self.t1, Nt+1)[1:]

        # Time step for integration and sampling
        self.dt = 1e-01
        # Standard deviation of measurement noise
        self.sigma = np.pi * 1e-02

        # Which dimensions to plot when predicting value function V(0,x)?
        # (unspecified dimensions are held at mean value)
        self.plotdims = [0, 3]

        # Number of training trajectories
        self.Ns = {'train': 6, 'val': 1000}

        ##### Options for training #####
        # Number of data points to use in first training rounds
        # Set to None to use whole data set
        self.batch_size = None  # 200

        # Maximum factor to increase data set size each round
        self.Ns_scale = 2
        # Number of candidate points to pick from when selecting large gradient
        # points during adaptive sampling
        self.Ns_cand = 2
        # Maximum size of batch size to use
        self.Ns_max = 8192

        # Convergence tolerance parameter (see paper)
        self.conv_tol = 1e-03

        # maximum and minimum number of training rounds
        self.max_rounds = 1
        self.min_rounds = 1

        # List or array of weights on gradient term, length = max_rounds
        self.weight_A = [1.]  # 1
        # List or array of weights on control learning term, not used in paper
        self.weight_U = [0.]  # 0.1

        # Dictionary of options to be passed to L-BFGS-B optimizer
        # Leave empty for default values
        self.BFGS_opts = {}

class setup_problem(problem_prototype):
    def __init__(self):
        self.N_states = 2
        self.t1 = 5.

        # Parameter setting for the equation X_dot = Ax+Bu
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0], [1]])

        # Initial condition bounds (different initial setting)
        # Currently, we give the initial position is [15m, 20m]
        # initial velocity is [18m/s, 25m/s]
        self.X0_lb = np.array([[15.], [18.], [15.], [18.]])
        self.X0_ub = np.array([[15.], [18.], [15.], [18.]])

        self.beta = 10000   # 10000
        self.theta1 = 1  # [1, 5]
        self.theta2 = 1   # [1, 5]

        # weight for terminal lose
        self.alpha = 1e-06  # 1e-06

        # Length for each vehicle
        self.L1 = 3
        self.L2 = 3

        # Length for each vehicle
        self.W1 = 1.5
        self.W2 = 1.5

        # Road length setting
        self.R1 = 70  # 71 or 62, 73
        self.R2 = 70

    def U_star(self, X_aug):
        '''Control as a function of the costate.'''
        # If we keep collision function L in the value cost function V, we consider dH/du = 0 and get the U*
        A = X_aug[2*self.N_states:3*self.N_states]
        U1 = np.matmul(self.B.T, A) / 2
        A = X_aug[5 * self.N_states:6 * self.N_states]
        U2 = np.matmul(self.B.T, A) / 2

        max_acc = 10
        min_acc = -5
        U1[np.where(U1 > max_acc)] = max_acc
        U1[np.where(U1 < min_acc)] = min_acc
        U2[np.where(U2 > max_acc)] = max_acc
        U2[np.where(U2 < min_acc)] = min_acc

        return U1, U2

    # Control input for neural network
    def make_U_NN_1(self, A):
        '''Makes Pytorch graph of optimal control with NN value gradient.'''
        import torch
        B = torch.tensor(self.B).float()
        U1 = torch.matmul(B.T, A[:self.N_states]) / 2

        U1 = U1.detach().numpy()
        max_acc = 10
        min_acc = -5
        U1[np.where(U1 > max_acc)] = max_acc
        U1[np.where(U1 < min_acc)] = min_acc
        U1 = torch.tensor(U1, requires_grad=True, dtype=torch.float32)

        return U1

    def make_U_NN_2(self, A):
        '''Makes Pytorch graph of optimal control with NN value gradient.'''
        import torch
        B = torch.tensor(self.B).float()
        U2 = torch.matmul(B.T, A[3*self.N_states:4*self.N_states]) / 2

        U2 = U2.detach().numpy()
        max_acc = 10
        min_acc = -5
        U2[np.where(U2 > max_acc)] = max_acc
        U2[np.where(U2 < min_acc)] = min_acc
        U2 = torch.tensor(U2, requires_grad=True, dtype=torch.float32)

        return U2

    # Boundary function for BVP
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:2*self.N_states]
            XT = X_aug_T[:2*self.N_states]
            AT = X_aug_T[2*self.N_states:6*self.N_states]
            VT = X_aug_T[6*self.N_states:]

            # Boundary setting for lambda(T) when it is the final time T
            dFdXT = np.concatenate((np.array([self.alpha]),
                                    np.array([-2 * (XT[1] - X0[1])]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([self.alpha]),
                                    np.array([-2 * (XT[3] - X0[3])])))

            # Terminal cost in the value function, see the new version of HJI equation
            F = np.array((self.alpha * XT[0] - (XT[1]-X0[1])**2, self.alpha * XT[2] - (XT[3]-X0[3])**2))

            return np.concatenate((X0 - X0_in, AT - dFdXT, VT - F))
        return bc

    # Terminal cost function in value function
    def terminal_cost_1(self, X):
        x1 = X[:int(self.N_states/2)]
        return - 1 / 2 * 1 / np.sum(x1 ** 2)

    def terminal_cost_2(self, X):
        x2 = X[:int(self.N_states/2)]
        return - 1 / 2 * 1 / np.sum(x2 ** 2)

    # Dynamic function for neural network
    def dynamics(self, t, X, U_fun):
        '''Evaluation of the dynamics at a single time instance for closed-loop
        ODE integration.'''
        U = U_fun([[t]], X.reshape((-1, 1)))
        U1 = U[-2:-1].flatten()
        U2 = U[-1:].flatten()

        X1 = X[:self.N_states]
        X2 = X[self.N_states:2*self.N_states]

        dXdt_1 = np.matmul(self.A, X1) + np.matmul(self.B, U1)
        dXdt_2 = np.matmul(self.A, X2) + np.matmul(self.B, U2)

        return np.concatenate((dXdt_1, dXdt_2))

    # PMP equation for BVP
    def aug_dynamics(self, t, X_aug):
        '''Evaluation of the augmented dynamics at a vector of time instances'''
        # Control as a function of the costate
        U1, U2 = self.U_star(X_aug)

        # State for each vehicle
        X1 = X_aug[:self.N_states]
        X2 = X_aug[self.N_states:2 * self.N_states]

        # State space function: X_dot = Ax+Bu
        dXdt_1 = np.matmul(self.A, X1) + np.matmul(self.B, U1)
        dXdt_2 = np.matmul(self.A, X2) + np.matmul(self.B, U2)

        # lambda in Hamiltonian equation
        A_11 = X_aug[2*self.N_states:3*self.N_states]
        A_12 = X_aug[3*self.N_states:4*self.N_states]
        A_21 = X_aug[4*self.N_states:5*self.N_states]
        A_22 = X_aug[5*self.N_states:6*self.N_states]

        # Sigmoid function: sigmoid(x1_in)*inverse_sigmoid(x1_out)*sigmoid(x2_in)*inverse_sigmoid(x2_out)
        x1 = torch.tensor(X1[0], requires_grad=True, dtype=torch.float32)  # including x1,v1
        x2 = torch.tensor(X2[0], requires_grad=True, dtype=torch.float32)  # including x1,v1
        # x1_in = (x1 - self.R1 / 2 + self.L1 / 2 + self.theta2 * self.W2 / 2) * 10
        # x1_out = -(x1 - self.R1 / 2 - self.W2 / 2 - self.L1 / 2) * 10
        # x2_in = (x2 - self.R2 / 2 + self.L2 / 2 + self.theta1 * self.W1 / 2) * 10
        # x2_out = -(x2 - self.R2 / 2 - self.W1 / 2 - self.L2 / 2) * 10

        x1_in = (x1 - self.R1 / 2 + self.theta2 * self.W2 / 2) * 10  # 3
        x1_out = -(x1 - self.R1 / 2 - self.W2 / 2 - self.L1) * 10
        x2_in = (x2 - self.R2 / 2 + self.theta1 * self.W1 / 2) * 10
        x2_out = -(x2 - self.R2 / 2 - self.W1 / 2 - self.L2) * 10

        # Collision_F_x = 10000/(0.01*(x1 - self.L1 / 2 - self.R1 / 2) ** 2 + 0.01*(x2 - self.L2 / 2 - self.R2 / 2) ** 2 + 1) \
        #                 * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(x2_in) * torch.sigmoid(x2_out)

        Collision_F_x = self.beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(x2_in) * torch.sigmoid(x2_out)

        Collision_F_x_sum = torch.sum(Collision_F_x)
        Collision_F_x_sum.requires_grad_()

        dL1dx1 = torch.autograd.grad(Collision_F_x_sum, x1, create_graph=True)[0].detach().numpy()
        dL1dx2 = torch.autograd.grad(Collision_F_x_sum, x2, create_graph=True)[0].detach().numpy()
        dL2dx1 = torch.autograd.grad(Collision_F_x_sum, x1, create_graph=True)[0].detach().numpy()
        dL2dx2 = torch.autograd.grad(Collision_F_x_sum, x2, create_graph=True)[0].detach().numpy()

        dL1dv1 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        dL1dv2 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        dL2dv1 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        dL2dv2 = np.zeros(dL2dx2.shape[0], dtype=np.int32)

        # lambda_dot in PMP equation
        dAdt_11 = -np.matmul(self.A.T, A_11) + np.array([dL1dx1, dL1dv1])
        dAdt_12 = -np.matmul(self.A.T, A_12) + np.array([dL1dx2, dL1dv2])
        dAdt_21 = -np.matmul(self.A.T, A_21) + np.array([dL2dx1, dL2dv1])
        dAdt_22 = -np.matmul(self.A.T, A_22) + np.array([dL2dx2, dL2dv2])

        # Collision function L1, L2
        L1 = U1 ** 2 + Collision_F_x.detach().numpy()
        L2 = U2 ** 2 + Collision_F_x.detach().numpy()

        return np.vstack((dXdt_1, dXdt_2, dAdt_11, dAdt_12, dAdt_21, dAdt_22, -L1, -L2))

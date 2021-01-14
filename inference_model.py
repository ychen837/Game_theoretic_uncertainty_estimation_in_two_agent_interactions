"""
Perform inference on other agent, updates the common belief table
In use:
    No_inference: agent does not infer other agent's param
    BVP_empathetic: uses Value network from BVP solver to evaluate agent's model
test models:
    Test_baseline: using the most basic Q function estimation, on how fast goal is reached
          (need to change main.py, environment = intersection)
    Trained_baseline: using NFSP model as Q function (change environment = trained_intersection)
    Empathetic: with NFSP, perform inference on both agent (change environment = trained_intersection)
"""
import torch
import numpy as np
from scipy.special import logsumexp
# from sklearn.preprocessing import normalize
from models.rainbow.set_nfsp_models import get_models
from HJI_Vehicle.utilities.neural_networks import HJB_network_t0 as HJB_network
from HJI_Vehicle.NN_output import get_Q_value
import dynamics
import pdb


class InferenceModel:
    def __init__(self, model, sim):  # model = inference type, sim = simulation class
        if model == 'none':
            self.infer = self.no_inference
        elif model == 'test_baseline':
            self.infer = self.test_baseline_inference
        elif model == 'nfsp_baseline':
            self.infer = self.nfsp_baseline_inference
        elif model == 'trained_baseline_2U':
            self.infer = self.trained_baseline_inference_2U
        elif model == 'empathetic':
            self.infer = self.empathetic_inference
        elif model == 'bvp_empathetic':  # use with bvp decision models and env
            self.infer = self.bvp_empathetic_inference
        elif model == 'bd_empathetic':
            self.infer = self.bd_empathetic
        else:
            # placeholder for future development
            pass

        "---simulation info:---"
        self.sim = sim
        self.agents = sim.agents
        self.n_agents = sim.n_agents
        self.frame = sim.frame
        self.time = sim.time
        self.T = 1  # one step look ahead/ Time Horizon
        self.dt = sim.dt  # default is 1s: assigned in main.py
        self.car_par = sim.env.car_par
        self.min_speed = 0.1
        self.max_speed = 30
        "---goal states (for baseline)---"
        self.goal = [self.car_par[0]["desired_state"], self.car_par[1]["desired_state"]]

        "---parameters(theta and lambda)---"
        # self.lambdas = [0.001, 0.005, 0.01, 0.05]
        # #self.lambdas = [0.05, 0.1, 1, 10]
        # self.thetas = [1, 1000]  # this is for NFSP
        self.lambda_list = self.sim.lambda_list
        self.theta_list = self.sim.theta_list

        "---Params for belief calculation (test baseline, single agent)---"
        self.initial_belief = None  # p0: initial belief of the param distribution
        # self.past_scores = {}  # score for each lambda
        # self.past_scores1 = {}  # for theta1
        # self.past_scores2 = {}  # for theta2
        # self.theta_priors = None #for calculating theta lambda joint probability

        self.theta_priors = self.sim.theta_priors
        "this is for baseline:"
        self.initial_joint_prob = np.ones((len(self.lambda_list), len(self.theta_list))) / (len(self.lambda_list) * len(self.theta_list)) #do this here to increase speed
        self.traj_h = []
        self.traj_m = []

        "getting true intents of self"
        self.true_intents = []
        for i, par_i in enumerate(self.sim.env.car_par):
            self.true_intents.append(par_i["par"])
        self.true_params = self.sim.true_params
        self.belief_params = self.sim.belief_params
        # self.action_set = [-8, -4, 0.0, 4, 8]  # from realistic trained model
        self.action_set = self.sim.action_set
        # self.action_set_combo = [[-8, -1], [-8, 0], [-8, 1], [-4, -1], [-4, 0],
        #                         [-4, 1], [0, -1], [0, 0], [0, 1], [4, -1],
        #                         [4, 0], [4, 1], [8, -1], [8, 0], [8, 1]]  # merging case actions

        "----for empathetic inference results:----"
        self.beta_initial_belief = self.sim.initial_belief
        self.p_betas_prior = self.sim.initial_belief  # this is updated over time
        self.prior_b_i = self.sim.initial_bi
        self.q2_prior = None  # not in use
        self.past_beta = []
        self.beta_set = self.sim.beta_set
        self.action_pair_score = []
        self.belief_count = np.zeros((len(self.beta_set), len(self.beta_set)))
        self.policy_choice = [[], []]  # 0 or not correct, 1 for correct guesses (for 2 agents)

    # @staticmethod
    def no_inference(self, agents, sim):
        print("frame {}".format(sim.frame))
        return

    def test_baseline_inference(self, agents, sim):
        # "Confidence-aware motion prediction for real-time collision avoidance"
        # THIS IS ONLY FOR TEST PURPOSE. NOT IN USE
        """
        for each agent, estimate its par (agent.par) based on its last action (agent.action[-1]),
        system state (agent.state[-1] for all agent), and prior dist. of par
        :return:

        Important equations implemented here:
        - Equation 1 (action_probabilities):
        P(u|x,theta,lambda) = exp(Q*lambda)/sum(exp(Q*lambda)), Q size = action space at state x
        
        - Equation 2 (belief_update):
         #Pseudo code for intent inference to obtain P(lambda, theta) based on past action sequence D(k-1):
        #P(lambda, theta|D(k)) = P(u| x;lambda, theta)P(lambda, theta | D(k-1)) / sum[ P(u|x;lambda', theta') P(lambda', theta' | D(k-1))]
        #equivalent: P = action_prob * P(k-1)/{sum_(lambda,theta)[action_prob * P(k-1)]}
        
        - Equation 3 (belief_resample):
        #resampled_prior = (1 - epsilon)*prior + epsilon * initial_belief
       
        """

        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        curr_state_h = sim.agents[0].state[self.frame]
        last_action_h = sim.agents[0].action[self.frame]
        curr_state_m = sim.agents[1].state[self.frame]
        last_action_m = sim.agents[1].action[self.frame]

        "trajectory: need to process and combine past states and actions together"
        states_h = sim.agents[0].state
        past_actions_h = sim.agents[0].action
        traj_h = []

        for i in range(sim.frame):
            traj_h.append([states_h[i], past_actions_h[i]])
        if sim.frame == 0:
            traj_h.append([states_h[0], past_actions_h[0]])
        "---end of agent info---"

        def q_function(current_s, action, goal_s, dt):
            """
            Calculates Q value for a state action pair (s, a)

            Current Implementation:
                Q is negatively proportional to the time it takes to reach the goal

            Params:
                current_s [tuple?] -- Current state containing x-state, y-state,
                    x-velocity, y-velocity
                action [IntEnum] -- potential action taken by agent H
                goal_s [tuple?] -- Goal state, with the format (sx, sy)
                dt[int / float] -- discrete time interval
            Returns:
                Q: an value correspond to action a in state s
            """
            # Q = -(s_goal - s_current)/v_next #estimates time required to reach goal with current state and chosen action
            u = action
            delta = 1  # to prevent denominator from becoming zero
            sx, sy, vx, vy = current_s[0], current_s[1], current_s[2], current_s[3]

            "Check agent movement axis then calculate Q value for given action"
            if sx == 0 and vx == 0: #For the case of y direction movement
                # Q = FUNCTION MATH HERE USING SY, VY
                #print("Y direction movement detected")
                goal_s = goal_s[0]
                next_v = vy + action * dt
                "Deceleration only leads to 0 velocity!"
                if next_v < 0.0:
                    # let v_next = 0
                    Q = -abs(goal_s - sy) / delta
                else:
                    Q = -abs(goal_s - sy) / (vy + action * dt + delta)
            elif sy == 0 and vy == 0: #For the case of X direction movement
                # Q = FUNCTION MATH HERE USING SX, VX
                print("X direction movement detected")
                goal_s = goal_s[1]
                next_v = vx + action * dt
                "Deceleration only leads to 0 velocity!"
                if next_v < 0.0:
                    # let v_next = 0
                    Q = -abs(goal_s - sx) / delta
                else:
                    Q = -abs(goal_s - sx) / (vx + action * dt + delta)
            else:  # Check for the case of 2D movement
                # Q = FUNCTION FOR 2D MODELS
                goal_x = goal_s[0]
                goal_y = goal_s[1]
                next_vx = vx + action * dt
                next_vy = vy + action * dt

                "Deceleration only leads to 0 velocity!"
                if next_vx < 0:
                    if next_vy < 0:  # both are negative
                        Q = -(abs(goal_y - sy) / delta + abs(goal_x - sx) / delta)
                    else:  # only vx is negative
                        Q = -(abs(goal_y - sy) / (vy + action * dt + delta) + abs(goal_x - sx) / delta)
                elif next_vy < 0:  # only vy is negative
                    Q = -(abs(goal_y - sy) / delta + abs(goal_x - sx) / (vx + action * dt + delta))
                else:  # both are non negative
                    Q = -(abs(goal_y - sy) / (vy + action * dt + delta) + abs(goal_x - sx) / (vx + action * dt + delta))

            return Q

        def q_values(state, goal):
            """
            Calls q_function and return a list of q values corresponding to an action set at a given state
            :param state:
            :param goal:
            return:
                A 1D list of values for a given state s with action set A

            """
            # Q = {} #dict type
            Q = []  # list type
            actions = self.action_set
            for a in actions:  # sets: file for defining sets
                # Q[a] = q_function(state, a, goal, self.dt)  #dict type
                Q.append(q_function(state, a, goal, self.dt))  # list type

            return Q

        def get_state_list(state, T, dt):
            """
            Obtain array of states resulting from list of actions given current state and given
            the number of future timesteps (lookahead distance)
            2D case: calculate an array of state (T x S at depth T)
            1D case: calculate a list of state (1 X (1 + Action_set^T))
            :param
                state: any state
                T: time horizon / look ahead
                dt: time interval where the action will be executed, i.e. u*dt
            :return:
                list of resulting states from taking each action at a given state
            """

            actions = self.action_set

            def get_s_prime(_state_list, _actions):
                _s_prime = []

                "Checking if _states is composed of tuples of state info (initially _state is just a tuple)"
                if not isinstance(_state_list[0], tuple):
                    print("WARNING: state list is not composed of tuples!")
                    _state_list = [_state_list] #put it in a list to iterate

                for s in _state_list:
                    for a in _actions:
                        #print("STATE", s)
                        _s_prime.append(dynamics.dynamics_1d(s, a, dt, self.min_speed, self.max_speed))
                return _s_prime

            i = 0  # row counter
            state_list = {}  # use dict to keep track of time step
            # state_list = []
            # state_list.append(state) #ADDING the current state!
            for t in range(0, T):
                s_prime = get_s_prime(state, actions)  # separate pos and speed!
                state_list[i] = s_prime
                # state_list.append(s_prime)
                state = s_prime  # get s prime for the new states
                i += 1  # move onto next row
            return state_list

        def action_probabilities(state, _lambda):  # equation 1
            """
            Noisy-rational model
            calculates probability distribution of action given hardmax Q values
            Uses:
            1. Softmax algorithm
            2. Q-value given state action pair (s, a)
            3. beta: "rationality coefficient"
            => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
            :return:
                Normalized probability distributions of available actions at a given state and lambda
            """
            # Need to add some filtering for states with no legal action: q = -inf
            Q = q_values(state, self.goal[0])
            #print("Q values array:", Q)

            exp_Q = []

            "Q*lambda"
            # np.multiply(Q,_lambda,out = Q)
            Q = [q * _lambda for q in Q]
            #print("Q*lambda:", Q)

            "Q*lambda/(sum(Q*lambda))"
            # np.exp(Q, out=Q)
            for q in Q:
                exp_Q.append(np.exp(q))
            #print("EXP_Q:", exp_Q)

            "normalizing"
            # normalize(exp_Q, norm = 'l1', copy = False)
            exp_Q /= sum(exp_Q)
            print("exp_Q normalized:", exp_Q)
            return exp_Q

        def traj_probabilities(state, _lambda, dt, prior = None):
            """
            Calculates probability of being in a set of states at time k+1: P(x(k+1)| lambda, theta)
            :params:
                state: current / observed state of H at time k
                _lambda: given lambda/rational coefficient
                dt: length of each time step
                prior: probability of agent being at "state" at time k (default is 1)
            :return:
                possible resulting states at k+1 with probabilities for being at each one of them
            """

            "for now we consider prior = 1 for observed state at time k"
            if prior == None:
                p_traj = 1  # initialize
            T = self.T
            state_list = get_state_list(state, T, dt)  # get list of state given curr_state/init_state from self._init_

            # p_states = np.zeros(shape=state_list)
            p_states = []

            "for the case where state list is 1D, note that len(state list) == number of time steps!"
            for i in range(len(state_list)):
                if i == 0:
                    p_a = action_probabilities(state, _lambda)
                    p_states.append(p_a.tolist())  # 1 step look ahead only depends on action prob
                    # transition is deterministic -> 1, prob state(k) = 1
                    # print("P STATES", p_states)

                else:
                    p_s_t = [] #storing p_states for time t (or i)
                    for j, s in enumerate(state_list[i-1]):
                        # print(state_list[i-1])
                        # print(p_states)
                        # print(type(p_states[0]))
                        # print("Current time:",i,"working on state:", j)
                        # print(p_states[i-1][j])
                        p_a = action_probabilities(s, _lambda) * p_states[i-1][j]
                        p_s_t.extend(p_a.tolist())

                    p_states.append(p_s_t)
            return p_states, state_list

        def belief_resample(priors, epsilon):
            """
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            #initial_belief = np.ones((len(priors), len(priors[0]))) / (len(priors)*len(priors[0]))
            initial_belief = self.initial_joint_prob
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def theta_joint_update(thetas, lambdas, traj, goal, epsilon=0.05, theta_priors=None):
            """
            refer to destination.py
            Calculates joint probability of lambda and theta, P(lambda, theta|D(k))
            :params:
                thetas: list of theta/intent/aggressiveness measure
                lambdas: list of lambda/rational coefficient
                traj: array of past state action pair
                goal: array describing the goal state
                epsilon: resample coefficient/weight
                theta_priors: prior of joint distribution of lambda and theta, P(lambda, theta|D(k-1))
            :return:
                posterior probabilities of each theta and corresponding lambda maximizing the probability
            """

            if theta_priors is None:
                # theta_priors = np.ones((len(lambdas), len(thetas))) / (len(thetas)*len(lambdas))
                theta_priors = self.initial_joint_prob
            print("-----theta priors: {}".format(theta_priors))
            print("traj: {}".format(traj))
            suited_lambdas = np.empty(len(thetas))
            # L = len(lambdas)

            "processing traj data, in the case that data of 2 agents are stored together in tuples:"

            # traj_state = []
            # traj_action = []
            # for i, traj_t in enumerate(traj):
            #     traj_state.append(traj_t[0])
            #     traj_action.append(traj_t[1])
            # h_states = []
            # h_actions = []
            # for j, s_t in enumerate(traj_state):
            #     h_states.append(traj_state[0])
            # for k, a_t in enumerate(traj_action):
            #     h_actions.append(traj_action[0])

            def compute_score(traj, _lambda): # without recording past traj
                # scores = np.empty(L)
                scores = []
                for i, (s, a) in enumerate(traj):  # pp score calculation method
                    # print("--i, (s, a), lambda:", i, (s, a), _lambda)
                    p_a = action_probabilities(s, _lambda)  # get probability of action in each state given a lambda
                    # scores[i] = p_a[s, a]
                    # print("-p_a[a]:", p_a[a_i])
                    # scores[i] = p_a[a]
                    a_i = self.action_set.index(a)
                    scores.append(p_a[a_i])
                # print("scores at frame {}:".format(self.frame), scores)
                log_scores = np.log(scores)
                return np.sum(log_scores)

            "USE THIS to record scores for past traj to speed up run time!"
            def get_last_score(traj, _lambda):  # add score to existing list of score
                p_a = action_probabilities(traj[-1][0], _lambda)
                a = traj[-1][1]
                a_i = self.action_set.index(a)
                if _lambda in self.past_scores:  # add to existing list
                    self.past_scores[_lambda].append(p_a[a_i])
                    scores = self.past_scores[_lambda]
                else:
                    self.past_scores[_lambda] = [p_a[a_i]]
                    scores = self.past_scores[_lambda]
                log_scores = np.log(scores)
                return np.sum(log_scores)

            "Calling compute_score/get_last_score to get the best suited lambdas"
            for i, theta in enumerate(thetas):  # get a best suited lambda for each theta
                score_list = []
                for j, lamb in enumerate(lambdas):
                    # score_list.append(compute_score(traj, lamb, L))
                    score_list.append(get_last_score(traj, lamb))
                max_lambda_j = np.argmax(score_list)
                suited_lambdas[i] = lambdas[max_lambda_j]  # recording the best suited lambda for corresponding theta[i]


            p_theta = np.copy(theta_priors)
            "Re-sampling from initial distribution (shouldn't matter if p_theta = prior?)"
            p_theta = belief_resample(p_theta, epsilon == 0.05)  # resample from uniform belief
            # lengths = len(thetas) * len(lambdas) #size of p_theta = size(thetas)*size(lambdas)
            p_theta_prime = np.empty((len(lambdas), len(thetas)))

            "Compute joint probability p(lambda, theta) for each lambda and theta"
            for t, (s, a) in enumerate(traj):  # enumerate through past traj
                if t == 0:  # initially there's only one state and not past
                    for theta_t in range(len(thetas)):  # cycle through list of thetas
                        for l,_lambda in enumerate(lambdas):
                            # p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                            # a_i = self.action_set.index(a)
                            p_action_l = self.past_scores[_lambda][t]
                            # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
                            p_theta_prime[l][theta_t] = p_action_l * p_theta[l][theta_t]
                else:  # for state action pair that is not at time zero
                    for theta_t in range(len(thetas)):  # cycle through theta at time t or K
                        # p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                        # refer to train_inference.joint_update for this part
                        for l, _lambda in enumerate(lambdas):
                            p_action_l = self.past_scores[_lambda][t]
                            for theta_past in range(len(thetas)):  # cycle through theta probability from past time K-1
                                for l_past in range(len(lambdas)):
                                    p_theta_prime[l][theta_t] += p_action_l * p_theta[l_past][theta_past]

            "In the case p_theta is 2d array:"
            print(p_theta_prime, sum(p_theta_prime))
            p_theta_prime /= np.sum(p_theta_prime)  # normalize

            "(OLD) Joint inference update for (lambda, theta)"
            # for t, (s, a) in enumerate(traj):  # enumerate through past traj
            #     if t == 0:  # initially there's only one state and not past
            #         for theta_t in range(len(thetas)):  # cycle through list of thetas
            #             p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
            #             a_i = self.action_set.index(a)
            #             p_theta_prime[theta_t] = p_action[a_i] * p_theta[theta_t]
            #     else:  # for state action pair that is not at time zero
            #         for theta_t in range(len(thetas)):  # cycle through theta at time t or K
            #             p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
            #             for theta_past in range(len(thetas)):  # cycle through theta probability from past time K-1
            #                 a_i = self.action_set.index(a)
            #                 p_theta_prime[theta_t] += p_action[a_i] * p_theta[theta_past]
            # p_theta_prime /= sum(p_theta_prime)  # normalize

            # print(p_theta_prime)
            # print(np.sum(p_theta_prime))
            assert 0.9 <= np.sum(p_theta_prime) <= 1.1  # check if it is properly normalized
            print("-----p_thetas at frame {0}: {1}".format(self.frame, p_theta_prime))
            # return {'predicted_intent_other': [p_theta_prime, suited_lambdas]}
            return p_theta_prime, suited_lambdas

        def marginal_prob(state, p_theta, best_lambdas, dt):
            """
            Calculates the marginal probability P(x(k+1) | D(k))
            General procedure:
            1. P(lambda, theta) is calculated first and best lambda is obtained
            2. P(x(k+1) | lambda, theta) is calculated with the best lambda from previous step
            3. Calculate P(x(k+1) | D(k)) by multiplying the results from first 2 steps together, with
               the same lambda
            :param
                state: current state that H is in
                p_state_beta: P(x(k+1)|lambda, theta)
                p_theta: P(lambda, theta), the joint probability of theta and corresponding lambda
                best lambdas: lambda with highest score for each theta -> e.g. pair (theta1, lambda1)
            :return:
                p_state_D: P(x(k+1) | D(k)), the marginal distribution of agent being in state x(k+1)
                           given observation D(k)
            """

            "get required information"
            lamb1, lamb2 = best_lambdas
            # print("WATCH for p_state", traj_probabilities(state, lamb1))
            p_state_beta1, state_list = traj_probabilities(state, lamb1, dt)
            p_state_beta2 = traj_probabilities(state, lamb2, dt)[0]
            print("p theta:",p_theta, "sum:", np.sum(p_theta), "len", len(p_theta))
            # print('p_state_beta1 at time ', self.frame, ' :', p_state_beta1)
            # print('p_state_beta2 at time ', self.frame, ' :', p_state_beta2)

            "calculate marginal"
            # p_state_D = p_state_beta1.copy() #<- this creates a list connected to original...? (nested?)
            p_state_D = []
            print(p_state_D)
            for k in range(len(state_list)): #k refers to the number of future time steps: currently max k=1
                p_state_beta1k = p_state_beta1[k]
                p_state_beta2k = p_state_beta2[k]
                p_state_D.append([])
                p_state_Dk = p_state_D[k]
                for i in range(len(p_state_beta1k)):
                    temp = 0
                    for j in range(len(p_theta)):
                        temp += p_state_beta1k[i] * p_theta[j][0] + p_state_beta2k[i] * p_theta[j][1]
                    p_state_Dk.append(temp)
                    # print(p_state_Dk[i])
            print('p_state_D at time ', self.frame, ' :', p_state_D)
            print("state of H:", state_list)  # sx, sy, vx, vy

            assert 0.99 <= np.sum(p_state_D[0]) <= 1.001  # check
            # return {'predicted_policy_other': [p_state_D, state_list]}
            return p_state_D, state_list

        "------------------------------"
        "executing the above functions!"
        "------------------------------"

        "#calling functions for baseline inference"
        joint_probability = theta_joint_update(thetas=self.theta_list, theta_priors=self.theta_priors,
                                               lambdas=self.lambda_list, traj=traj_h, goal=self.goal, epsilon=0.05)

        "#take a snapshot of the theta prob for next time step"
        self.theta_priors, best_lambdas = joint_probability

        "calculate the marginal state distribution / prediction"
        best_lambdas = joint_probability[1]
        marginal_state = marginal_prob(state=curr_state_h, p_theta=self.theta_priors,
                                       best_lambdas=best_lambdas, dt=1)  # IMPORTANT: set dt to desired look ahead

        return {'predicted_intent_other': joint_probability,
                'predicted_states_other': marginal_state}

    def nfsp_baseline_inference(self, agent, sim):
        # not in use, for test purposes
        """
        Use Q function from nfsp models
        Important equations implemented here:
        - Equation 1 (action_probabilities):
        P(u|x,theta,lambda) = exp(Q*lambda)/sum(exp(Q*lambda)), Q size = action space at state x

        - Equation 2 (belief_update):
         #Pseudo code for intent inference to obtain P(lambda, theta) based on past action sequence D(k-1):
        #P(lambda, theta|D(k)) = P(u| x;lambda, theta)P(lambda, theta | D(k-1)) / sum[ P(u|x;lambda', theta') P(lambda', theta' | D(k-1))]
        #equivalent: P = action_prob * P(k-1)/{sum_(lambda,theta)[action_prob * P(k-1)]}

        - Equation 3 (belief_resample):
        #resampled_prior = (1 - epsilon)*prior + epsilon * initial_belief
        :param agent:
        :param sim:
        :return: inferred other agent's parameters (P(k), P(x(k+1))
        """

        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        curr_state_h = sim.agents[0].state[self.frame]
        last_action_h = sim.agents[0].action[self.frame]
        curr_state_m = sim.agents[1].state[self.frame]
        last_action_m = sim.agents[1].action[self.frame]

        # curr_state_h = sim.agents[0].state[-1]
        # last_action_h = sim.agents[0].action[-1]
        # curr_state_m = sim.agents[1].state[-1]
        # last_action_m = sim.agents[1].action[-1]

        self.traj_h.append([curr_state_h, last_action_h])
        self.traj_m.append([curr_state_m, last_action_m])

        def trained_q_function():
            """
            Import Q function from nfsp given states
            :param state_h:
            :param state_m:
            :return:
            """

            q_set = get_models()[0]

            return q_set

        def q_values(state_h, state_m, intent):
            """
            Get q values given the intent (Non-aggressive or aggressive)
            :param state_h:
            :param state_m:
            :param intent:
            :return:
            """
            q_set = trained_q_function()
            if intent == "na_na":
                Q = q_set[0]
            else: #use a_na
                Q = q_set[3]

            "Need state for agent H: xH, vH, xM, vM"
            # state = [state_h[0], state_h[2], state_m[1], state_m[3]]
            state = [-state_h[1], abs(state_h[3]), state_m[0], abs(state_m[2])]

            Q_vals = Q.forward(torch.FloatTensor(state).to(torch.device("cpu")))

            return Q_vals

        def action_prob(state_h, state_m, _lambda, theta):
            """
            Noisy-rational model
            calculates probability distribution of action given hardmax Q values
            Uses:
            1. Softmax algorithm
            2. Q-value given state and theta(intent)
            3. lambda: "rationality coefficient"
            => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
            :param state_h: current H state
            :param state_m: current M state
            :param _lambda: rationality coefficient
            :param theta: aggressiveness/ gracefullness parameter
            :return: Normalized probability distributions of available actions at a given state and lambda
            """
            action_set = self.action_set
            if theta == self.theta_list[0]:
                intent = "na_na"
            else:
                intent = "a_na"

            print(intent)
            q_vals = q_values(state_h, state_m, intent=intent)
            exp_Q = []

            "Q*lambda"
            # np.multiply(Q,_lambda,out = Q)
            q_vals = q_vals.detach().numpy() #detaching tensor
            #print("q values: ",q_vals)
            Q = [q * _lambda for q in q_vals]
            # print("Q*lambda:", Q)
            "Q*lambda/(sum(Q*lambda))"
            # np.exp(Q, out=Q)

            for q in Q:
                exp_Q.append(np.exp(q))
            # print("EXP_Q:", exp_Q)

            "normalizing"
            # normalize(exp_Q, norm = 'l1', copy = False)
            exp_Q /= sum(exp_Q)
            print("exp_Q normalized:", exp_Q)
            return exp_Q

        def get_state_list(state, T, dt):
            """
            2D case: calculate an array of state (T x S at depth T)
            1D case: calculate a list of state (1 X (1 + Action_set^T))
            :param
                state: current state
                T: time horizon / look ahead
                dt: time interval where the action will be executed, i.e. u*dt
            :return:
                list of resulting states from taking each action at a given state
            """

            actions = self.action_set

            def get_s_prime(_state_list, _actions):
                _s_prime = []

                "Checking if _states is composed of tuples of state info (initially _state is just a tuple)"
                if not isinstance(_state_list[0], tuple):
                    # print("WARNING: state list is not composed of tuples!")
                    _state_list = [_state_list]  # put it in a list to iterate

                for s in _state_list:
                    for a in _actions:
                        # print("STATE", s)
                        # _s_prime.append(calc_state(s, a, dt))
                        _s_prime.append(dynamics.dynamics_1d(s, a, dt, self.min_speed, self.max_speed))
                return _s_prime

            i = 0  # row counter
            state_list = {}  # use dict to keep track of time step
            # state_list = []
            # state_list.append(state) #ADDING the current state!
            for t in range(0, T):
                s_prime = get_s_prime(state, actions)  # separate pos and speed!
                state_list[i] = s_prime
                # state_list.append(s_prime)
                state = s_prime  # get s prime for the new states
                i += 1  # move onto next row

            return state_list

        def traj_prob(state_h, state_m, _lambda, theta, dt, prior=None):
            """
                Calculates probability of being in a set of states at time k+1: P(x(k+1)| lambda, theta)
            :params:
                state: current / observed state of H at time k
                _lambda: given lambda/rational coefficient
                dt: length of each time step
                prior: probability of agent being at "state" at time k (default is 1)
            :return:
                possible resulting states at k+1 with probabilities for being at each one of them
            """

            "for now we consider prior = 1 for observed state at time k"
            if prior == None:
                p_traj = 1  # initialize
            T = self.T
            state_list = get_state_list(state_h, T, dt)  # get list of state given curr_state/init_state from self._init_

            # p_states = np.zeros(shape=state_list)
            p_states = []

            "for the case where state list is 1D, note that len(state list) == number of time steps!"
            for i in range(len(state_list)):
                if i == 0:
                    p_a = action_prob(state_h, state_m, _lambda, theta)
                    p_states.append(p_a.tolist())  # 1 step look ahead only depends on action prob
                    # transition is deterministic -> 1, prob state(k) = 1
                    # print("P STATES", p_states)

                else:
                    p_s_t = []  # storing p_states for time t (or i)
                    for j, s in enumerate(state_list[i - 1]):
                        # print(state_list[i-1])
                        # print(p_states)
                        # print(type(p_states[0]))
                        # print("Current time:",i,"working on state:", j)
                        # print(p_states[i-1][j])
                        p_a = action_prob(state_h, state_m, _lambda, theta) * p_states[i - 1][j]
                        p_s_t.extend(p_a.tolist())

                    p_states.append(p_s_t)
            assert round(np.sum(p_states[0])) == 1
            return p_states, state_list

        def resample(priors, epsilon):
            """
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            # initial_belief = np.ones((len(priors), len(priors[0]))) / (len(priors)*len(priors[0]))
            initial_belief = self.initial_joint_prob
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def joint_prob(theta_list, lambdas, traj_h, traj_m, epsilon=0.05, priors=None):
            """
            update belief on P(lambda, theta)
            :param theta_list:
            :param lambdas:
            :param traj_h:
            :param traj_m:
            :param goal:
            :param epsilon:
            :param priors:
            :return: P(lambda, theta) and best lambda
            """
            intent_list = ["na_na", "a_na"]
            if priors is None:
                # theta_priors = np.ones((len(lambdas), len(thetas))) / (len(thetas)*len(lambdas))
                priors = self.initial_joint_prob

            print("-----theta priors: {}".format(priors))
            print("traj: {}".format(traj_h))
            # pdb.set_trace()

            # this is not in use, but it works by recording how each lambda explains the trajectory
            "USE THIS to record scores for past traj to speed up run time!"
            def get_last_score(_traj_h, _traj_m, _lambda, _theta):  # add score to existing list of score
                p_a = action_prob(_traj_h[-1][0], _traj_m[-1][0], _lambda, _theta) #traj: [(s, a), (s2, a2), ..]
                a_h = _traj_h[-1][1]
                #print(_traj_h)
                a_i = self.action_set.index(a_h)
                if _theta == self.theta_list[0]:
                    if _lambda in self.past_scores1:  # add to existing list
                        self.past_scores1[_lambda].append(p_a[a_i])
                        scores = self.past_scores1[_lambda]
                    else:
                        self.past_scores1[_lambda] = [p_a[a_i]]
                        scores = self.past_scores1[_lambda]
                else: #theta2
                    if _lambda in self.past_scores2:  # add to existing list
                        self.past_scores2[_lambda].append(p_a[a_i])
                        scores = self.past_scores2[_lambda]
                    else:
                        self.past_scores2[_lambda] = [p_a[a_i]]
                        scores = self.past_scores2[_lambda]
                log_scores = np.log(scores)
                return np.sum(log_scores)

            "Calling compute_score/get_last_score to get the best suited lambdas for EACH theta"
            # for i, theta in enumerate(theta_list):  # get a best suited lambda for each theta
            #     #score_list = []
            #     for j, lamb in enumerate(lambdas):
            #         get_last_score(traj_h, traj_m, lamb, theta)
                #     score_list.append(get_last_score(traj_h, traj_m, lamb, theta))
                # max_lambda_j = np.argmax(score_list)
                # suited_lambdas[i] = lambdas[max_lambda_j]  # recording the best suited lambda for corresponding theta[i]

            p_joint_prior = np.copy(priors)
            # print("prior:", p_theta)
            "Re-sampling from initial distribution (shouldn't matter if p_theta = prior?)"
            p_joint_prior = resample(p_joint_prior, epsilon == 0.05)  # resample from uniform belief
            # print("resampled:", p_theta)
            # lengths = len(thetas) * len(lambdas) #size of p_theta = size(thetas)*size(lambdas)
            p_joint_prime = np.empty((len(lambdas), len(theta_list)))

            "another joint update algorithm, without cycling through traj and only update with prior:"
            t = self.frame
            for theta_t, theta in enumerate(theta_list):  # cycle through list of thetas
                for l, _lambda in enumerate(lambdas):
                    # p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                    # a_i = self.action_set.index(a)
                    p_action = action_prob(traj_h[-1][0], traj_m[-1][0], _lambda,
                                           theta)  # get prob of action done at time t
                    a_i = self.action_set.index(traj_h[-1][1])
                    p_action_l = p_action[a_i]
                    if theta == theta_list[0]:
                        # p_action_l = self.past_scores1[_lambda][t]  # get prob of action done at time t
                        # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
                        p_joint_prime[l][theta_t] = p_action_l * p_joint_prior[l][theta_t]
                    else:  # theta 2
                        # p_action_l = self.past_scores2[_lambda][t]
                        # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
                        p_joint_prime[l][theta_t] = p_action_l * p_joint_prior[l][theta_t]

            "In the case p_theta is 2d array:"
            # print(p_theta_prime, sum(p_theta_prime))
            p_joint_prime /= np.sum(p_joint_prime)  # normalize

            "get best lambda from distribution"
            suited_lambdas = np.empty(len(intent_list))
            for t, theta in enumerate(p_joint_prime.transpose()):
                suited_lambdas[t] = self.lambda_list[np.argmax(theta)]

            assert 0.9 <= np.sum(p_joint_prime) <= 1.1  # check if it is properly normalized
            print("-----p_thetas at frame {0}: {1}".format(self.frame, p_joint_prime))
            return p_joint_prime, suited_lambdas

        def marginal_state(state_h, state_m, p_theta, dt):
            """
            Get marginal: P(x(k+1)|D(k)) from P(x(k+1)|lambda, theta) and P(lambda, theta|D(k))
            :param state_h:
            :param state_m:
            :param p_theta:
            :param best_lambdas:
            :param thetas:
            :param dt:
            :return:
            """

            "get required information"
            # lamb1, lamb2 = best_lambdas
            lambdas = self.lambda_list
            theta1, theta2 = self.theta_list
            # print("WATCH for p_state", traj_probabilities(state, lamb1))
            "get P(x(k+1)|theta,lambda) for the 2 thetas"
            p_state_beta1= traj_prob(state_h, state_m, lambdas[0], theta1, dt)[0]
            # p_state_beta2 = traj_prob(state_h, state_m, lamb2, theta2, dt)[0] #only probability no state list
            # print("p theta:", p_theta, "sum:", np.sum(p_theta), "len", len(p_theta))
            # print('p_state_beta1 at time ', self.frame, ' :', p_state_beta1)
            # print('p_state_beta2 at time ', self.frame, ' :', p_state_beta2)

            "calculate marginal"
            # p_state_D = p_state_beta1.copy() #<- this creates a list connected to original...? (nested?)
            p_state_D = {}
            print(p_state_beta1)
            for t in range(len(p_state_beta1)):
                p_state_D[t] = np.zeros(len(p_state_beta1[t]))
            for t, theta in enumerate(self.theta_list):
                for l, lamb in enumerate(lambdas):
                    _p_state_beta = traj_prob(state_h, state_m, lamb, theta, dt)[0]  # calculate p_state using beta
                    for k in range(len(_p_state_beta)):  # k refers to the number of future time steps: currently max k=1
                        for i in range(len(_p_state_beta[k])):
                            p_state_D[k][i] += _p_state_beta[k][i] * p_theta[l][t]

                    # print(p_state_Dk[i])
            _state_list = get_state_list(state_h, self.T, dt)
            print('p_state_D at time ', self.frame, ' :', p_state_D)
            print("state of H:", state_h, _state_list)  # sx, sy, vx, vy
            assert round(np.sum(p_state_D[0])) == 1  # check

            return p_state_D, _state_list

        "------------------------------"
        "executing the above functions!"
        "------------------------------"

        "#calling functions for baseline inference"
        joint_probability = joint_prob(theta_list=self.theta_list, lambdas=self.lambda_list,
                                       traj_h=self.traj_h, traj_m=self.traj_m, epsilon=0.05,
                                       priors=self.theta_priors)

        "#take a snapshot of the theta prob for next time step"
        p_joint, best_lambdas = joint_probability

        "calculate the marginal state distribution / prediction"
        marginal_state_prob, states_list = marginal_state(curr_state_h, curr_state_m, p_joint, self.dt)

        "getting the predicted action"
        p_theta = np.zeros(len(self.theta_list))
        for i, p_t in enumerate(p_joint.transpose()):  # get marginal prob of theta: p(theta) from joint prob p(lambda, theta)
            p_theta[i] = sum(p_t)
        theta_idx = np.argmax(p_theta)
        theta_h = self.theta_list[theta_idx]
        p_a = action_prob(curr_state_h, curr_state_m, best_lambdas[theta_idx], theta=theta_h)
        # p_a = action_prob(curr_state_h, curr_state_m, _lambda=self.lambdas[-1], theta=theta_h)  # for verification with decision model
        predicted_action = self.action_set[np.argmax(p_a)]

        "converting from array to list"
        for m in range(len(marginal_state_prob)):
            marginal_state_prob[m] = marginal_state_prob[m].tolist()
        self.theta_priors = p_joint

        # IMPORTANT: set dt to desired look ahead
        return {'predicted_intent_other': [p_joint, best_lambdas],
                'predicted_states_other': [marginal_state_prob, states_list],
                'predicted_actions_other': predicted_action}

    def trained_baseline_inference_2U(self, agent, sim):
        # not in use, test purpose only
        """
        Use Q function from nfsp models
        Important equations implemented here:
        - Equation 1 (action_probabilities):
        P(u|x,theta,lambda) = exp(Q*lambda)/sum(exp(Q*lambda)), Q size = action space at state x

        - Equation 2 (belief_update):
         #Pseudo code for intent inference to obtain P(lambda, theta) based on past action sequence D(k-1):
        #P(lambda, theta|D(k)) = P(u| x;lambda, theta)P(lambda, theta | D(k-1)) / sum[ P(u|x;lambda', theta') P(lambda', theta' | D(k-1))]
        #equivalent: P = action_prob * P(k-1)/{sum_(lambda,theta)[action_prob * P(k-1)]}

        - Equation 3 (belief_resample):
        #resampled_prior = (1 - epsilon)*prior + epsilon * initial_belief
        :param agent:
        :param sim:
        :return: inferred other agent's parameters (P(k), P(x(k+1))
        """
        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        curr_state_h = sim.agents[0].state[self.frame]
        last_action_h = sim.agents[0].action[self.frame]
        curr_state_m = sim.agents[1].state[self.frame]
        last_action_m = sim.agents[1].action[self.frame]

        # curr_state_h = sim.agents[0].state[-1]
        # last_action_h = sim.agents[0].action[-1]
        # curr_state_m = sim.agents[1].state[-1]
        # last_action_m = sim.agents[1].action[-1]

        self.traj_h.append([curr_state_h, last_action_h]) # 5 states and 2 actions
        self.traj_m.append([curr_state_m, last_action_m])


        def trained_q_function():
            """
            Import Q function from nfsp given states
            :param state_h:
            :param state_m:
            :return:
            """

            q_set = get_models()[0]

            return q_set

        def q_values(state_h, state_m, theta_h, theta_m):
            """
            Get q values given the intent (Non-aggressive or aggressive)
            :param state_h:
            :param state_m:
            :param theta_h:
            :param theta_m:
            :return:
            """
            q_set = trained_q_function()

            id_h = self.theta_list.index(theta_h)
            id_m = self.theta_list.index(theta_m)
            if id_h == 0:
                if id_m == 0:
                    Q_h = q_set[0]
                    # Q_m = q_set[1]
                elif id_m == 1:  # M is aggressive
                    Q_h = q_set[2]
                    # Q_m = q_set[3]
                else:
                    print("ID FOR THETA DOES NOT EXIST")
            elif id_h == 1:
                if id_m == 0:
                    Q_h = q_set[3]
                    # Q_m = q_set[2]
                elif id_m == 1:
                    Q_h = q_set[4]
                    # Q_m = q_set[5]
                else:
                    print("ID FOR THETA DOES NOT EXIST")
            "Need state for agent H: xH, vH, xM, vM"
            # state = [state_h[0], state_h[2], state_m[1], state_m[3]]
            state = [-state_h[1], abs(state_h[3]), state_m[0], abs(state_m[2])]

            Q_vals = Q_h.forward(torch.FloatTensor(state).to(torch.device("cpu")))

            return Q_vals

        def action_prob(state_h, state_m, _lambda, theta_h):
            """
            Noisy-rational model
            calculates probability distribution of action given hardmax Q values
            Uses:
            1. Softmax algorithm
            2. Q-value given state and theta(intent)
            3. lambda: "rationality coefficient"
            => P(uH|xH;beta,theta) = exp(beta*QH(xH,uH;theta))/sum_u_tilde[exp(beta*QH(xH,u_tilde;theta))]
            :param state_h: current H state
            :param state_m: current M state
            :param _lambda: rationality coefficient
            :param theta: aggressiveness/ gracefullness parameter
            :return: Normalized probability distributions of available actions at a given state and lambda
            """
            theta_m = self.sim.env.car_par[1]["par"]
            action_set = self.action_set
            #action_set = self.action_set_combo
            # if theta_h == self.thetas[0]:
            #     intent = "na_na"
            # else:
            #     intent = "a_na"

            q_vals = q_values(state_h, state_m, theta_h, theta_m)
            exp_Q = []

            "Q*lambda"
            # np.multiply(Q,_lambda,out = Q)
            q_vals = q_vals.detach().numpy()  # detaching tensor
            # print("q values: ",q_vals)
            Q = [q * _lambda for q in q_vals]
            # print("Q*lambda:", Q)
            "Q*lambda/(sum(Q*lambda))"
            # np.exp(Q, out=Q)

            for q in Q:
                exp_Q.append(np.exp(q))

            "normalizing"
            # normalize(exp_Q, norm = 'l1', copy = False)
            exp_Q /= sum(exp_Q)
            # print("exp_Q normalized:", exp_Q)
            assert (not pa == 0 for pa in exp_Q)
            return exp_Q

        def resample(priors, epsilon):
            """
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            # TODO: generalize this algorithm for difference sizes of matrices(1D, 2D)
            # initial_belief = np.ones((len(priors), len(priors[0]))) / (len(priors)*len(priors[0]))
            initial_belief = self.initial_joint_prob
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def joint_prob(theta_list, lambdas, traj_h, traj_m, epsilon=0.05, priors=None):
            """
            update belief on P(lambda, theta)
            :param theta_list:
            :param lambdas:
            :param traj_h:
            :param traj_m:
            :param goal:
            :param epsilon:
            :param priors:
            :return: P(lambda, theta) and best lambda
            """

            if priors is None:
                #theta_priors = np.ones((len(lambdas), len(thetas))) / (len(thetas)*len(lambdas))
                priors = self.initial_joint_prob

            print("-----theta priors: {}".format(priors))
            print("traj: {}".format(traj_h))
            # pdb.set_trace()

            # "USE THIS to record scores for past traj to speed up run time!"
            # def get_last_score(_traj_h, _traj_m, _lambda, _theta):  # add score to existing list of score
            #     p_a = action_prob(_traj_h[-1][0], _traj_m[-1][0], _lambda, _theta) #traj: [(s, a), (s2, a2), ..]
            #     a_h = _traj_h[-1][1]
            #     #print(_traj_h)
            #     a_i = self.action_set.index(a_h)
            #     if _theta == self.thetas[0]:
            #         if _lambda in self.past_scores1:  # add to existing list
            #             self.past_scores1[_lambda].append(p_a[a_i])
            #             scores = self.past_scores1[_lambda]
            #         else:
            #             self.past_scores1[_lambda] = [p_a[a_i]]
            #             scores = self.past_scores1[_lambda]
            #     else: #theta2
            #         if _lambda in self.past_scores2:  # add to existing list
            #             self.past_scores2[_lambda].append(p_a[a_i])
            #             scores = self.past_scores2[_lambda]
            #         else:
            #             self.past_scores2[_lambda] = [p_a[a_i]]
            #             scores = self.past_scores2[_lambda]
            #     log_scores = np.log(scores)
            #     return np.sum(log_scores)

            "Calling compute_score/get_last_score to get the best suited lambdas for EACH theta"
            # for i, theta in enumerate(theta_list):  # get a best suited lambda for each theta
            #     #score_list = []
            #     for j, lamb in enumerate(lambdas):
            #         get_last_score(traj_h, traj_m, lamb, theta)
                #     score_list.append(get_last_score(traj_h, traj_m, lamb, theta))
                # max_lambda_j = np.argmax(score_list)
                # suited_lambdas[i] = lambdas[max_lambda_j]  # recording the best suited lambda for corresponding theta[i]

            p_joint_prior = np.copy(priors)
            # print("prior:", p_theta)
            "Re-sampling from initial distribution (shouldn't matter if p_theta = prior?)"
            p_joint_prior = resample(p_joint_prior, epsilon == 0.05)  # resample from uniform belief
            # print("resampled:", p_theta)
            # lengths = len(thetas) * len(lambdas) #size of p_theta = size(thetas)*size(lambdas)
            p_joint_prime = np.empty((len(lambdas), len(theta_list)))

            "another joint update algorithm, without cycling through traj and only update with prior:"
            t = self.frame
            for theta_t, theta in enumerate(theta_list):  # cycle through list of thetas
                for l, _lambda in enumerate(lambdas):
                    # p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                    # a_i = self.action_set.index(a)
                    p_action = action_prob(traj_h[-1][0], traj_m[-1][0], _lambda,
                                           theta)  # get prob of action done at time t
                    a_i = self.action_set.index(traj_h[-1][1]) # extract action index
                    p_action_l = p_action[a_i] #
                    if theta == theta_list[0]:
                        # p_action_l = self.past_scores1[_lambda][t]  # get prob of action done at time t
                        # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
                        p_joint_prime[l][theta_t] = p_action_l * p_joint_prior[l][theta_t]
                    else:  # theta 2
                        # p_action_l = self.past_scores2[_lambda][t]
                        # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
                        p_joint_prime[l][theta_t] = p_action_l * p_joint_prior[l][theta_t] # update p]

            "In the case p_theta is 2d array:"
            # print(p_theta_prime, sum(p_theta_prime))
            p_joint_prime /= np.sum(p_joint_prime)  # normalize

            "get best lambda from distribution"
            suited_lambdas = np.empty(len(theta_list))
            for t, theta in enumerate(p_joint_prime.transpose()):
                suited_lambdas[t] = self.lambda_list[np.argmax(theta)]

            assert 0.9 <= np.sum(p_joint_prime) <= 1.1  # check if it is properly normalized
            print("-----p_thetas at frame {0}: {1}".format(self.frame, p_joint_prime))
            return p_joint_prime, suited_lambdas\

        def get_state_list(state, T, dt):
            # check if it works for this model
            """
            2D case: calculate an array of state (T x S at depth T)
            1D case: calculate a list of state (1 X (1 + Action_set^T))
            :param
                state: current state
                T: time horizon / look ahead
                dt: time interval where the action will be executed, i.e. u*dt
            :return:
                list of resulting states from taking each action at a given state
            """

            actions = self.action_set

            def get_s_prime(_state_list, _actions):
                _s_prime = []

                "Checking if _states is composed of tuples of state info (initially _state is just a tuple)"
                # TODO: fix this!!!
                if not isinstance(_state_list[0], tuple):
                    # print("WARNING: state list is not composed of tuples!")
                    _state_list = [_state_list]  # put it in a list to iterate

                for s in _state_list:
                    for a in _actions:
                        # print("STATE", s)
                        # _s_prime.append(calc_state(s, a, dt))
                        #_s_prime.append(dynamics.dynamics_1d(s, a, dt, self.min_speed, self.max_speed))
                        _s_prime.append(dynamics.dynamics_2d(s, a, dt, self.min_speed, self.max_speed))
                return _s_prime

            i = 0  # row counter
            state_list = {}  # use dict to keep track of time step
            # state_list = []
            # state_list.append(state) #ADDING the current state!
            for t in range(0, T):
                s_prime = get_s_prime(state, actions)  # separate pos and speed!
                state_list[i] = s_prime
                # state_list.append(s_prime)
                state = s_prime  # get s prime for the new states
                i += 1  # move onto next row

            return state_list

        def traj_prob(state_h, state_m, _lambda, theta, dt, prior=None):
            """
                Calculates probability of being in a set of states at time k+1: P(x(k+1)| lambda, theta)
            :params:
                state: current / observed state of H at time k
                _lambda: given lambda/rational coefficient
                dt: length of each time step
                prior: probability of agent being at "state" at time k (default is 1)
            :return:
                possible resulting states at k+1 with probabilities for being at each one of them
            """

            "for now we consider prior = 1 for observed state at time k"
            if prior == None:
                p_traj = 1  # initialize
            T = self.T
            state_list = get_state_list(state_h, T, dt)  # get list of state given curr_state/init_state from self._init_

            # p_states = np.zeros(shape=state_list)
            p_states = []

            "for the case where state list is 1D, note that len(state list) == number of time steps!"
            for i in range(len(state_list)):
                if i == 0:
                    p_a = action_prob(state_h, state_m, _lambda, theta)
                    p_states.append(p_a.tolist())  # 1 step look ahead only depends on action prob
                    # transition is deterministic -> 1, prob state(k) = 1
                    # print("P STATES", p_states)

                else:
                    p_s_t = []  # storing p_states for time t (or i)
                    for j, s in enumerate(state_list[i - 1]):
                        # print(state_list[i-1])
                        # print(p_states)
                        # print(type(p_states[0]))
                        # print("Current time:",i,"working on state:", j)
                        # print(p_states[i-1][j])
                        p_a = action_prob(state_h, state_m, _lambda, theta) * p_states[i - 1][j]
                        p_s_t.extend(p_a.tolist())

                    p_states.append(p_s_t)
            assert round(np.sum(p_states[0])) == 1
            return p_states, state_list

        def marginal_state(state_h, state_m, p_theta, dt):
            """
            Equation 5
            Get marginal: P(x(k+1)|D(k)) from P(x(k+1)|lambda, theta) and P(lambda, theta|D(k))
            :param state_h:
            :param state_m:
            :param p_theta:
            :param best_lambdas:
            :param thetas:
            :param dt:
            :return:
            """

            "get required information"
            # lamb1, lamb2 = best_lambdas
            lambdas = self.lambda_list
            theta1, theta2 = self.theta_list
            # print("WATCH for p_state", traj_probabilities(state, lamb1))
            "get P(x(k+1)|theta,lambda) for the 2 thetas"
            p_state_beta1= traj_prob(state_h, state_m, lambdas[0], theta1, dt)[0]
            # p_state_beta2 = traj_prob(state_h, state_m, lamb2, theta2, dt)[0] #only probability no state list
            # print("p theta:", p_theta, "sum:", np.sum(p_theta), "len", len(p_theta))
            # print('p_state_beta1 at time ', self.frame, ' :', p_state_beta1)
            # print('p_state_beta2 at time ', self.frame, ' :', p_state_beta2)

            "calculate marginal"
            # p_state_D = p_state_beta1.copy() #<- this creates a list connected to original...? (nested?)
            p_state_D = {}
            print(p_state_beta1)
            for t in range(len(p_state_beta1)):
                p_state_D[t] = np.zeros(len(p_state_beta1[t]))
            for t, theta in enumerate(self.theta_list):
                for l, lamb in enumerate(lambdas):
                    _p_state_beta = traj_prob(state_h, state_m, lamb, theta, dt)[0]  # calculate p_state using beta
                    for k in range(len(_p_state_beta)):  # k refers to the number of future time steps: currently max k=1
                        for i in range(len(_p_state_beta[k])):
                            p_state_D[k][i] += _p_state_beta[k][i] * p_theta[l][t]

                    # print(p_state_Dk[i])
            _state_list = get_state_list(state_h, self.T, dt)
            print('p_state_D at time ', self.frame, ' :', p_state_D)
            print("state of H:", state_h, _state_list)  # sx, sy, vx, vy
            assert round(np.sum(p_state_D[0])) == 1  # check

            return p_state_D, _state_list

        # --------------------- Beginning of the algorithm -----------------------
        "#calling functions for baseline inference"
        joint_probability = joint_prob(theta_list=self.theta_list, lambdas=self.lambda_list,
                                       traj_h=self.traj_h, traj_m=self.traj_m, epsilon=0.05,
                                       priors=self.theta_priors)

        "#take a snapshot of the theta prob for next time step"
        p_joint, best_lambdas = joint_probability

        "calculate the marginal state distribution / prediction"
        marginal_state_prob, states_list = marginal_state(curr_state_h, curr_state_m, p_joint, self.dt)

        "getting the predicted action"
        p_theta = np.zeros(len(self.theta_list))
        for i, p_t in enumerate(p_joint.transpose()):  # get marginal prob of theta: p(theta) from joint prob p(lambda, theta)
            p_theta[i] = sum(p_t)
        theta_idx = np.argmax(p_theta)
        theta_h = self.theta_list[theta_idx]
        p_a = action_prob(curr_state_h, curr_state_m, best_lambdas[theta_idx], theta_h=theta_h)
        # p_a = action_prob(curr_state_h, curr_state_m, _lambda=self.lambdas[-1], theta=theta_h)  # for verification with decision model
        predicted_action = self.action_set[np.argmax(p_a)]

        "converting from array to list"
        for m in range(len(marginal_state_prob)):
            marginal_state_prob[m] = marginal_state_prob[m].tolist()
        self.theta_priors = p_joint


        return {'predicted_intent_other': [p_joint, best_lambdas],
                'predicted_states_other': [marginal_state_prob, states_list],
                'predicted_actions_other': predicted_action}

    def empathetic_inference(self, agent, sim):
        """
        When QH also depends on xM,uM
        :return:P(beta_h, beta_m_hat | D(k))
        """
        # ----------------------------#
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # ----------------------------#

        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        assert len(sim.agents[0].state) == len(sim.agents[0].action)
        curr_state_h = sim.agents[0].state[self.frame]
        last_action_h = sim.agents[0].action[self.frame - 1]
        last_state_h = sim.agents[0].state[self.frame - 1]

        curr_state_m = sim.agents[1].state[self.frame]
        last_action_m = sim.agents[1].action[self.frame - 1]
        last_state_m = sim.agents[1].state[self.frame - 1]

        self.traj_h.append([last_state_h, last_action_h])
        self.traj_m.append([last_state_m, last_action_m])

        "place holder: using NFSP Q function in place of NE Q function pair"
        def trained_q_function(state_h, state_m):
            """
            Import Q function from nfsp given states
            :param state_h:
            :param state_m:
            :return:
            """
            # Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2
            q_set = get_models()[0]  # 0: q func, 1: policy
            # Q = q_set[0]  # use na_na for now
            "Q values for given state over a set of actions:"
            # Q_vals = Q.forward(torch.FloatTensor(state).to(torch.device("cpu")))
            return q_set

        def q_values_pair(state_h, state_m, theta_h, theta_m):
            """
            extracts the Q function and obtain Q value given current state configuration
            :param state_h:
            :param state_m:
            :param intent:
            :return:
            """
            q_set = trained_q_function(state_h, state_m)
            "Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2"

            "thetas: na, a"
            id_h = self.theta_list.index(theta_h)
            id_m = self.theta_list.index(theta_m)
            if id_h == 0:
                if id_m == 0:
                    Q_h = q_set[0]
                    Q_m = q_set[1]
                elif id_m == 1:  # M is aggressive
                    Q_h = q_set[2]
                    Q_m = q_set[3]
                else:
                    print("ID FOR THETA DOES NOT EXIST")
            elif id_h == 1:
                if id_m == 0:
                    Q_h = q_set[3]
                    Q_m = q_set[2]
                elif id_m == 1:
                    Q_h = q_set[4]
                    Q_m = q_set[5]
                else:
                    print("ID FOR THETA DOES NOT EXIST")
            else:
                print("ID FOR THETA DOES NOT EXIST")

            "Need state for agent H: xH, vH, xM, vM"
            p1_state_nn = [-state_h[1], abs(state_h[3]), state_m[0], abs(state_m[2])]
            p2_state_nn = [state_m[0], abs(state_m[2]), -state_h[1], abs(state_h[3])]

            "Q values for each action"
            Q_vals_h = Q_h.forward(torch.FloatTensor(p1_state_nn).to(torch.device("cpu")))
            Q_vals_m = Q_m.forward(torch.FloatTensor(p2_state_nn).to(torch.device("cpu")))
            "detaching tensor"
            Q_vals_h = Q_vals_h.detach().numpy()
            Q_vals_m = Q_vals_m.detach().numpy()
            Q_vals_h = Q_vals_h.tolist()
            Q_vals_m = Q_vals_m.tolist()
            return [Q_vals_h, Q_vals_m]

        def action_prob(state_h, state_m, beta_h, beta_m):
            """
            calculate action prob for both agents
            :param state_h:
            :param state_m:
            :param _lambda:
            :param theta:
            :return: [p_action_H, p_action_M], where p_action = [p_a1, ..., p_a5]
            """

            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m
            action_set = self.action_set

            q_vals_pair = q_values_pair(state_h, state_m, theta_h, theta_m)

            "Q*lambda"
            # q_vals_h = q_vals_h.detach().numpy()  # detaching tensor
            # q_vals_m = q_vals_m.detach().numpy()
            # q_vals_pair = [q_vals_h, q_vals_m]
            # print("q values: ",q_vals)
            exp_Q_pair = []
            _lambda = [lambda_h, lambda_m]
            for i, q_vals in enumerate(q_vals_pair):
                exp_Q = []
                Q = [q * _lambda[i] for q in q_vals]
                # print("Q*lambda:", Q)
                "Q*lambda/(sum(Q*lambda))"
                # np.exp(Q, out=Q)

                for q in Q:
                    #print(exp_Q)
                    exp_Q.append(np.exp(q))
                # print("EXP_Q:", exp_Q)

                "normalizing"
                # normalize(exp_Q, norm = 'l1', copy = False)
                exp_Q /= sum(exp_Q)
                assert round(np.sum(exp_Q)) == 1
                exp_Q_pair.append(exp_Q)

            return exp_Q_pair  # [exp_Q_h, exp_Q_m]

        def action_prob_Q(state_h, state_m, Q_h, Q_m, beta_h, beta_m):
            """
            Equation 1
            calculate action prob for both agents given Q_h and Q_m
            :param state_h:
            :param state_m:
            :param _lambda:
            :param theta:
            :return: [p_action_H, p_action_M], where p_action = [p_a1, ..., p_a5]
            """
            action_set = self.action_set

            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m

            "Q*lambda"
            q_vals_pair = [Q_h, Q_m]
            exp_Q_pair = []
            _lambda = [lambda_h, lambda_m]
            for i, q_vals in enumerate(q_vals_pair):
                exp_Q = []
                Q = [q * _lambda[i] for q in q_vals]
                # print("Q*lambda:", Q)
                "Q*lambda/(sum(Q*lambda))"
                # np.exp(Q, out=Q)

                for q in Q:
                    exp_Q.append(np.exp(q))
                # print("EXP_Q:", exp_Q)

                "normalizing"
                # normalize(exp_Q, norm = 'l1', copy = False)
                exp_Q /= sum(exp_Q)
                # print("exp_Q normalized:", exp_Q)
                assert round(np.sum(exp_Q)) == 1
                exp_Q_pair.append(exp_Q)

            return exp_Q_pair  # [exp_Q_h, exp_Q_m]

        def resample(priors, epsilon):
            """
            Equation 3
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            # initial_belief should be from sim!!!
            if isinstance(priors[0], list):
                initial_belief = np.ones((len(priors), len(priors[0]))) / (len(priors)*len(priors[0]))
            elif type(priors[0]) is np.array:
                initial_belief = np.ones((len(priors), len(priors[0]))) / (len(priors) * len(priors[0]))
            else:  # 1D array
                initial_belief = np.ones(len(priors))/len(priors)
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def prob_q_vals(prior, state_h, state_m, traj_h, traj_m, beta_h, beta_m):
            """
            Equation 6
            Calculates Q function pairs probabilities for use in beta pair probabilities calculation,
            since each beta pair may map to MULTIPLE Q function/value pair.

            :requires: p_action_h, p_pair_prob(k-1) or prior, q_pairs
            :param:
            q_pairs: all Q function pairs (QH, QM)
            p_action_h
            p_q2

            :return:[Q_pairs, P(QH, QM|D(k))], where Q_pairs = [Q_NA_NA, ..., Q_A_A]
            """

            # get all q pairs
            q_pairs = []
            "q_pairs (QH, QM): [[Q_na_na, Q_na_na2], [Q_na_a, Q_a_na], [Q_a_na, Q_na_a], [Q_a_a, Q_a_a2]]"
            "or think of it: [[Q_na_na, Q_na_na2], [Q_na_a, Q_a_na], "
            "                 [Q_a_na, Q_na_a],     [Q_a_a, Q_a_a2]] "
            for t_h in self.theta_list:
                for t_m in self.theta_list:
                    q_pairs.append(q_values_pair(state_h, state_m, t_h, t_m))  # [q_vals_h, q_vals_m]

            # Size of P(Q2|D) should be the size of possible Q2
            if prior is None:
                prior = np.ones(len(q_pairs))/len(q_pairs)
            else:
                "resample from initial/uniform distribution"
                prior = resample(prior, epsilon=0.05)

            p_q2 = np.empty(len(q_pairs))
            scores = []
            # assuming 1D array of q functions
            "Calculating probability of each Q pair: Equation 6"
            past_state_h, action_h = traj_h[-1]
            past_state_m, action_m = traj_m[-1]
            ah = self.action_set.index(action_h)
            am = self.action_set.index(action_m)
            for i, q in enumerate(q_pairs):  # iterate through pairs of equilibria
                p_action = action_prob_Q(past_state_h, past_state_m, q[0], q[1], beta_h, beta_m)  # action prob for last time step!
                p_action_h = p_action[0][ah]
                p_action_m = p_action[1][am]
                p_a_pair = p_action_h * p_action_m
                scores.append(p_a_pair)
                "P(Q2|D(k)) = P(uH, uM|x(k), QH, QM) * P(Q2|D(k-1)) / sum(~)"
                p_q2[i] = p_a_pair * prior[i]

            self.action_pair_score.append(scores)  # to compare the difference between action prob for q pairs
            p_q2 /= sum(p_q2)  # normalize
            assert round(sum(p_q2)) == 1  # check if properly normalized

            return q_pairs, p_q2

        def prob_q2_beta(index, q_id):
            """
            Pre-requisite for Equation 8
            Get probability of pair QH, QM given betas
            :param index: 2 entries: (i, j), where i is the index for beta_h, j is the index for beta_m
            i(0:7) = theta1 = na, i(8:15) = theta2 = a
            j(0:7) = theta1 = na, j(8:15) = theta2 = a
            :return:
            """

            "get intent of q_pair from q_pairs"
            "q_pairs (QH, QM): [[Q_na_na, Q_na_na2], [Q_na_a, Q_a_na], [Q_a_na, Q_na_a], [Q_a_a, Q_a_a2]]"
            # q_id = q_pairs.index(q_pair)  # 0, 1, 2, 3
            # print("Q id:", q_id)
            if q_id == 0:
                th = 0; tm = 0
            elif q_id == 1:
                th = 0; tm = 1
            elif q_id == 2:
                th = 1; tm = 0
            else:
                th = 1; tm = 1
            id = []
            "checking if given beta is a or na, in 1D betas:"
            half = len(self.beta_set)/2
            assert half == 2
            for i in index:  # (i, j) = (row, col). 4x4 2D array
                if i < half:  # NA
                    id.append(0)
                else:  # A
                    id.append(1)

            if id[0] == th and id[1] == tm:  # if beta matches with Q
                return 1
            else:
                return 0

        def prob_beta_given_q(p_betas_prior, q_id):
            """
            using Bayes rule
            Calculates probability of beta pair (Bh, BM_hat) given Q pair (QH, QM): P(Bh, BM_hat | QH, QM),
            for beta_pair_prob formula.
            P(beta_H, beta_m|QH, QM) = P(QH, QM | beta_H, beta_m) * P(beta_H, beta_m | D(k-1)) / sum (P(QH, QM | beta_H, beta_m) * P(beta_H, beta_m | D(k-1)))
            --> GIVEN A PARTICULAR Q PAIR
            :return: P(beta_H, beta_m|QH, QM): 8x8 matrix given particular Q pair
            """

            "import prob of beta pair given D(k-1) from Equation 7: P(betas|D(k-1))"
            if p_betas_prior is None:  # this should be using initial belief from sim
                betas_len = len(self.beta_set)
                p_betas_prior = np.ones((betas_len, betas_len)) / (betas_len * betas_len)  # uniform prior
            else:
                p_betas_prior = resample(p_betas_prior, epsilon=0.05)

            "calculate prob of beta pair given Q pair"
            p_beta_q2 = np.empty((len(p_betas_prior), len(p_betas_prior[0])))
            for i in range(len(p_betas_prior)):
                for j in range(len(p_betas_prior[i])):
                    'getting P(Q2|betas), given beta id (i, j)'
                    p_q2_beta = prob_q2_beta((i, j), q_id)  # scalar
                    p_beta_q2[i][j] = p_q2_beta * p_betas_prior[i][j]

            # print(p_beta_q2)
            p_beta_q2 /= np.sum(p_beta_q2)
            # assert 0.99 <= np.sum(p_beta_q2) <= 1.01  # check if properly normalized

            return p_beta_q2

        def prob_beta_pair(p_q2, q_pairs):
            """
            Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
            :return: P(beta_H, beta_M | D(k)), 8x8
            """
            p_betas_d = np.zeros((len(self.beta_set), len(self.beta_set)))
            "Calculate prob of beta pair given D(k) by summing over Q pair"
            for i, q2 in enumerate(q_pairs):  # cycle through q pairs
                p_betas_q2 = prob_beta_given_q(p_betas_prior=self.p_betas_prior, q_id=i)
                for j in range(len(p_betas_q2)):
                    for k in range(len(p_betas_q2[j])):
                        p_betas_d[j][k] += p_betas_q2[j][k] * p_q2[i]

            assert round(np.sum(p_betas_d)) == 1  # make sure this calculation is correct; no need to normalize
            return p_betas_d

        def get_state_list(state, T, dt):
            """
            2D case: calculate an array of state (T x S at depth T)
            1D case: calculate a list of state (1 X (1 + Action_set^T))
            :param
                state: any state
                T: time horizon / look ahead
                dt: time interval where the action will be executed, i.e. u*dt
            :return:
                list of resulting states from taking each action at a given state
            """

            actions = self.action_set

            def get_s_prime(_state_list, _actions):
                _s_prime = []

                "Checking if _states is composed of tuples of state info (initially _state is just a tuple)"
                if not isinstance(_state_list[0], tuple):
                    print("WARNING: state list is not composed of tuples!")
                    _state_list = [_state_list]  # put it in a list to iterate

                for s in _state_list:
                    for a in _actions:
                        # print("STATE", s)
                        # _s_prime.append(calc_state(s, a, dt))
                        _s_prime.append(dynamics.dynamics_1d(s, a, dt, self.min_speed, self.max_speed))
                return _s_prime

            i = 0  # row counter
            _state_list = {}  # use dict to keep track of time step
            # state_list = []
            # state_list.append(state) #ADDING the current state!
            for t in range(0, T):
                s_prime = get_s_prime(state, actions)  # separate pos and speed!
                _state_list[i] = s_prime
                # state_list.append(s_prime)
                state = s_prime  # get s prime for the new states
                i += 1  # move onto next row
            return _state_list

        def joint_action_prob(state_h, state_m, beta_h, beta_m):
            """
            Equation 10 (prerequisite for Equation 9)
            Multiplying the two action prob together for both agents
            :param state_h:
            :param state_m:
            :param lambdas:
            :param thetas:
            :return: 5x5 P(u_H, u_M| x(k), beta_H, beta_m)
            """

            "in the case where action prob is calculated separately"
            # lambda_h, lambda_m = lambdas
            # theta_h, theta_m = thetas
            # p_action_h = action_prob(state_h, state_m, lambda_h, theta_h)
            # p_action_m = action_prob(state_m, state_h, lambda_m, theta_m)

            "in the case where action prob is calculated TOGETHER"
            p_action_h, p_action_m = action_prob(state_h, state_m, beta_h, beta_m)

            p_action_joint = np.empty((len(p_action_h), len(p_action_m)))
            for i, p_a_h in enumerate(p_action_h):
                for j, p_a_m in enumerate(p_action_m):
                    p_action_joint[i][j] = p_a_h * p_a_m
                    # p_action_joint.append(p_a_h * p_a_m)
            assert round(np.sum(p_action_joint)) == 1

            return p_action_joint

        def joint_action_prob_Q(state_h, state_m, Q_h, Q_m):
            """
            Multiplying the two action prob together for both agents
            USING Q PAIR INSTEAD OF BETAS
            :return: noisy rational model
            """

            "in the case where action prob is calculated separately"
            # lambda_h, lambda_m = lambdas
            # theta_h, theta_m = thetas
            # p_action_h = action_prob(state_h, state_m, lambda_h, theta_h)
            # p_action_m = action_prob(state_m, state_h, lambda_m, theta_m)

            "in the case where action prob is calculated TOGETHER"
            p_action_h, p_action_m = action_prob(state_h, state_m, last_beta_h, last_beta_m)

            p_action_joint = np.empty((len(p_action_h), len(p_action_m)))
            for i, p_a_h in enumerate(p_action_h):
                for j, p_a_m in enumerate(p_action_m):
                    p_action_joint[i][j] = p_a_h * p_a_m

            assert round(np.sum(p_action_joint)) == 1

            return p_action_joint

        def traj_prob(state_h, state_m, Q_h, Q_m, dt, prior=None):
            """
                Calculates probability of being in a set of states at time k+1: P(x(k+1)| QH, QM)
            :params:
                state: current / observed state of H at time k
                _lambda: given lambda/rational coefficient
                dt: length of each time step
                prior: probability of agent being at "state" at time k (default is 1)
            :return:
                possible resulting states at k+1 with probabilities for being at each one of them
            """

            "for now we consider prior = 1 for observed state at time k"
            if prior == None:
                p_traj = 1  # initialize
            T = self.T
            state_list_h = get_state_list(state_h, T, dt)  # get list of state given curr_state/init_state from self._init_
            state_list_m = get_state_list(state_m, T, dt)

            state_list = {}
            "joining H and M's states together (2D array)"
            for i in range(len(state_list_h)):  # time step
                state_list[i] = []
                for j in range(len(state_list_h[i])):  # state_h
                    state_list[i].append([])
                    for p in range(len(state_list_m)):  # time step
                        for q in range(len(state_list_m[p])):  # state_m
                            state_list[i][j].append([state_list_h[i][j], state_list_m[p][q]])

            # p_states = np.zeros(shape=state_list)
            p_states = {}

            "for the case where state list is 1D, note that len(state list) == number of time steps!"
            for i in range(len(state_list_h)):  # over state list at time i/ t
                p_a = joint_action_prob_Q(state_h, state_m, Q_h, Q_m)

                if i == 0:
                    p_states[i] = p_a  # 1 step look ahead only depends on action prob
                    # transition is deterministic -> 1, prob state(k) = 1
                    # print("P STATES", p_states)

                else:  # time steps at i > 0
                    p_s_t = []  # storing p_states for time t (or i)
                    for j, ps in enumerate(state_list_h[i - 1]):
                        # print(state_list[i-1])
                        # print(p_states)
                        # print(type(p_states[0]))
                        # print("Current time:",i,"working on state:", j)
                        # print(p_states[i-1][j])
                        p_s = p_a * p_states[i - 1][j]  # assume same action prob at all time
                        p_s_t.extend(p_s.tolist())
                    p_states[i] = p_s_t

            "state_list contains (state_h, state_m)_k"
            return p_states, state_list

        def marginal_state_prob(state_h, state_m, p_q2, Q_pairs, dt):
            """
            Calculate the marginal state prob from P(x(k+1)|QH, QM)
            :param p_traj:
            :param p_q2:
            :return: P(x(k+1)|D(k)): {0:[x1, x2,...], 1:[x1, x2, ...]}, where 0, 1, .. is the future time k
            """

            marginal = {}
            p_traj = traj_prob(state_h, state_m, Q_pairs[0][0], Q_pairs[0][1], dt=dt)[0]  # this is just for getting sizes
            for t in range(len(p_traj)):
                for r in range(len(p_traj[t])):
                    marginal[t] = np.zeros((len(p_traj[t]), len(p_traj[t][r])))

            for i, q2 in enumerate(Q_pairs):
                p_traj = traj_prob(state_h, state_m, q2[0], q2[1], dt=dt)[0]

                for t in range(len(p_traj)):  # time step
                    for r in range(len(p_traj[t])):  # resulting states
                        for c in range(len(p_traj[t][r])): # rows
                            marginal[t][r][c] += p_traj[t][r][c] * p_q2[i]

            assert round(np.sum(marginal[0])) == 1
            return marginal

        # function to rearrange for 2D p(theta, lambda|D(k))
        def marginal_joint_intent(id, _p_beta_d):
            """
            Get the marginal P(Beta_i|D(k)) from P(beta_H, beta_M|D(k))
            :param id:
            :param _p_beta_d:
            :return: 4x2 matrix, where row is the lambda and col is the theta, for visualization purposes
            """
            marginal = []
            for t in self.theta_list:
                marginal.append([])
            "create a 2D array of (lambda, theta) pairs distribution like single agent case"
            half = round(len(self.beta_set) / 2)
            if id == 0:  # H agent
                for i, row in enumerate(_p_beta_d):  # get sum of row
                    if i < half:  # in 1D self.beta, first half are NA, or theta1
                        marginal[0].append(sum(row))
                    else:
                        marginal[1].append(sum(row))
            else:
                for i, col in enumerate(zip(*_p_beta_d)):
                    if i < half:
                        marginal[0].append(sum(col))
                    else:
                        marginal[1].append(sum(col))
            # i-4 if i>3
            id_lambda = marginal.index(max(marginal))
            _best_lambda = self.lambda_list[id_lambda] if id_lambda < half else self.lambda_list[id_lambda - half]
            marginal = np.array(marginal)
            marginal = marginal.transpose()  # Lambdas x Thetas
            return marginal, _best_lambda

        def marginal_action(state_h, state_m, _p_beta_D):
            # not in use
            """
            Get marginal action prob P(u|x;D) from P(u|x;beta), for analysis purposes
            :return:
                p(uH|D), p(uM|D)
            """

            _p_action_h = np.zeros(len(self.action_set))  # 1D
            _p_action_m = np.zeros(len(self.action_set))  # 1D
            #_p_beta_D = _p_beta_D.ravel()
            for i in range(len(self.beta_set)):
                for j in range(len(self.beta_set)):
                    p_a_b_h, p_a_b_m = action_prob(state_h, state_m, beta_h=self.beta_set[i], beta_m=self.beta_set[j])
                    for k in range(len(p_a_b_h)):
                        _p_action_h[k] += p_a_b_h[k] * _p_beta_D[i][j]
                        _p_action_m[k] += p_a_b_h[k] * _p_beta_D[i][j]

            assert round(np.sum(_p_action_h)) == 1 and round(np.sum(_p_action_m)) == 1
            return _p_action_h, _p_action_m

        "---------------------------------------------------"
        "calling functions: P(Q2|D), P(beta2|D), P(x(k+1)|D)"
        "---------------------------------------------------"

        if self.frame == 0:  # initially guess the beta
            # last_theta_h = self.thetas[0]
            # last_lambda_h = self.lambdas[-1]  # start large
            # last_theta_m = self.thetas[0]
            # last_lambda_m = self.lambdas[-1]
            p_beta = self.sim.initial_belief
            beta_pair_id = np.unravel_index(p_beta.argmax(), p_beta.shape)
            last_beta_h = self.beta_set[beta_pair_id[0]]
            last_beta_m = self.beta_set[beta_pair_id[1]]
            last_theta_h, last_lambda_h = last_beta_h
            last_theta_m, last_lambda_m = last_beta_m

        else:  # get previously predicted beta
            last_beta_h, last_beta_m = self.past_beta[-1]
            last_theta_h, last_lambda_h = last_beta_h
            last_theta_m, last_lambda_m = last_beta_m
            "TEST: fixing the lambda to check"
            # lambda_h, lambda_m = self.lambdas[-1], self.lambdas[-1]
            # beta_h, beta_m = self.betas[-1], self.betas[3]  #H:A, M: NA

        'intent and rationality inference'
        # q2 = q_values_pair(curr_state_h, curr_state_m, theta_h, theta_m)
        q_pairs, p_q2_d = prob_q_vals(self.q2_prior, curr_state_h, curr_state_m,
                                      traj_h=self.traj_h, traj_m=self.traj_m,
                                      beta_h=last_beta_h, beta_m=last_beta_m)  # betas are used for action_prob
        # print("Q pairs at time {0}:".format(self.frame), q_pairs, len(q_pairs))
        # p_beta_q = prob_beta_given_q(beta_h, beta_m,
        #                              p_betas_prior=self.p_betas_prior, q_pairs=[q2], q_pair=q2)
        p_beta_d = prob_beta_pair(p_q2=p_q2_d, q_pairs=q_pairs)  # this calls p_beta_q internally

        'future state prediction'
        p_traj, state_list = traj_prob(curr_state_h, curr_state_m, last_beta_h, last_beta_m, dt=self.dt)
        p_q2 = np.array(p_q2_d).tolist()
        print('p_traj:', p_traj)

        marginal_state = marginal_state_prob(curr_state_h, curr_state_m,
                                             p_q2=p_q2, Q_pairs=q_pairs, dt=self.dt)

        'recording prior'
        self.q2_prior = p_q2_d
        self.p_betas_prior = p_beta_d

        'getting best predicted betas'
        beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
        # print("best betas ID at time {0}".format(self.frame), beta_pair_id)

        new_beta_h = self.beta_set[beta_pair_id[0]]
        new_beta_m = self.beta_set[beta_pair_id[1]]
        self.past_beta.append([new_beta_h, new_beta_m])

        "getting marginal prob for beta_h or beta_m: THIS IS ONLY FOR PLOTTING, NOT DECISION"
        # only for estimation:
        p_beta_d_h, best_lambda_h = marginal_joint_intent(id=0, _p_beta_d=p_beta_d)
        p_beta_d_m, best_lambda_m = marginal_joint_intent(id=1, _p_beta_d=p_beta_d)

        "getting most likely action for analysis purpose"
        # only for estimation purpose
        p_actions = action_prob(curr_state_h, curr_state_m, new_beta_h, new_beta_m)  # for testing with decision
        predicted_actions = []
        for i, p_a in enumerate(p_actions):
            # p_a = marginal_action(p_a, p_beta_d_pair[i][round(beta_pair_id[i]/2)][beta_pair_id[i] % 2])
            p_a = np.array(p_a).tolist()
            id = p_a.index(max(p_a))
            predicted_actions.append(self.action_set[id])

        "obtaining marginal state distribution for both agents"
        marginal_state_h = {}
        marginal_state_m = {}
        for t, marginal_s in enumerate(marginal_state):  # time step
            print(marginal_state[t])
            marginal_state_h[t] = [sum(marg) for marg in marginal_state[t]]
            marginal_state_m[t] = [sum(marg) for marg in zip(*marginal_state[t])]
            print(marginal_state[t])
            assert round(sum(marginal_state_m[t])) == 1 and round(sum(marginal_state_h[t])) == 1

        # p_theta_prime, suited_lambdas <- predicted_intent other
        # p_betas: [BH x BM]
        # print("state list and prob for H: ", state_list, marginal_state)
        # print("size of state list at t=1", len(state_list[0]))  # should be 5x5 2D
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # print("-inf- marginal state for m: ", marginal_state_m)
        # print("-Intent_inf- marginal state H: ", marginal_state_h)
        return {'predicted_states_other': (marginal_state_h, get_state_list(curr_state_h, self.T, self.dt)),  # col of 2D should be M
                'predicted_actions_other': predicted_actions[0],
                'predicted_intent_other': [p_beta_d_h, new_beta_h],
                'predicted_states_self': (marginal_state_m, get_state_list(curr_state_m, self.T, self.dt)),
                'predicted_actions_self': predicted_actions[1],
                'predicted_intent_self': [p_beta_d_m, new_beta_m],
                'predicted_intent_all': [p_beta_d, [new_beta_h, new_beta_m]]}

    def bvp_empathetic_inference(self, agent, sim):
        """
        Using Q values from value network trained from BVP solver
        When QH also depends on xM,uM
        :return:P(beta_h, beta_m_hat | D(k))
        """

        # ----------------------------#
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # ----------------------------#

        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        self.time = self.sim.time
        assert len(sim.agents[0].state) == len(sim.agents[0].action)
        curr_state_h = sim.agents[0].state[self.frame]
        last_action_h = sim.agents[0].action[self.frame - 1]
        last_state_h = sim.agents[0].state[self.frame - 1]

        curr_state_m = sim.agents[1].state[self.frame]
        last_action_m = sim.agents[1].action[self.frame - 1]
        last_state_m = sim.agents[1].state[self.frame - 1]

        self.traj_h.append([last_state_h, last_action_h])
        self.traj_m.append([last_state_m, last_action_m])

        def action_prob(state_h, state_m, beta_h, beta_m):
            # bvp_action prob is in use instead of this, for faster speed
            """
            calculate action prob for both agents
            :param state_h:
            :param state_m:
            :return: [p_action_H, p_action_M], where p_action = [p_a1, ..., p_a5]
            """

            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m
            action_set = self.action_set
            p1_state_nn = np.array([[state_h[1]], [state_h[3]], [state_m[0]], [state_m[2]]])
            p2_state_nn = np.array([[state_m[0]], [state_m[2]], [state_h[1]], [state_h[3]]])
            _lambda = [lambda_h, lambda_m]

            _p_action_1 = np.zeros(((len(action_set)), len(action_set)))
            _p_action_2 = np.zeros(((len(action_set)), len(action_set)))
            time = np.array([[self.time]])
            dt = self.sim.dt

            for i, p_a_h in enumerate(_p_action_1):
                for j, p_a_m in enumerate(_p_action_1[i]):
                    new_p2_s = dynamics.bvp_dynamics_1d(state_m, action_set[j], dt)
                    new_p1_s = dynamics.bvp_dynamics_1d(state_h, action_set[i], dt)
                    if (theta_h, theta_m) == (1, 5):  # Flip A_AN to NA_A
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[action_set[j]], [action_set[i]]]),
                                             (theta_m, theta_h))  # NA_A
                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[action_set[i]], [action_set[j]]]),
                                             (theta_h, theta_m))
                    lamb_Q1 = q1 * lambda_h
                    _p_action_1[i][j] = lamb_Q1
                    lamb_Q2 = q2 * lambda_m
                    _p_action_2[i][j] = lamb_Q2

            "using logsumexp to prevent nan"
            Q1_logsumexp = logsumexp(_p_action_1)
            Q2_logsumexp = logsumexp(_p_action_2)
            "normalizing"
            _p_action_1 -= Q1_logsumexp
            _p_action_2 -= Q2_logsumexp
            _p_action_1 = np.exp(_p_action_1)
            _p_action_2 = np.exp(_p_action_2)
            assert round(np.sum(_p_action_1)) == 1
            assert round(np.sum(_p_action_2)) == 1
            'p action based on observed action of other agent'
            _last_action_h = self.traj_h[-1][1]
            _last_action_m = self.traj_m[-1][1]
            ah = action_set.index(_last_action_h)
            am = action_set.index(_last_action_m)
            _pa_1_t = np.transpose(_p_action_1)
            _pa_1 = _pa_1_t[am]  # column of pa
            _pa_2 = _p_action_2[ah]
            _pa_1 /= np.sum(_pa_1)
            _pa_2 /= np.sum(_pa_2)
            assert round(np.sum(_pa_1)) == 1
            assert round(np.sum(_pa_2)) == 1

            return [_pa_1, _pa_2]  # [exp_Q_h[past_am], exp_Q_m[past_ah]]

        def past_action_prob(state_h, state_m, beta_h, beta_m, action_h, action_m):
            """
            calculate action prob for both agents
            :param state_h:
            :param state_m:
            :param _lambda:
            :param theta:
            :return: [p_action_H, p_action_M], where p_action = [p_a1, ..., p_a5]
            """

            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m
            action_set = self.action_set
            p1_state_nn = np.array([[state_h[1]], [state_h[3]], [state_m[0]], [state_m[2]]])
            p2_state_nn = np.array([[state_m[0]], [state_m[2]], [state_h[1]], [state_h[3]]])
            "get action index"
            ah = action_set.index(action_h)
            am = action_set.index(action_m)
            _lambda = [lambda_h, lambda_m]

            _p_action_1 = np.zeros(((len(action_set)), len(action_set)))
            _p_action_2 = np.zeros(((len(action_set)), len(action_set)))
            time = np.array([[self.time]])
            dt = self.sim.dt

            for i, p_a_h in enumerate(_p_action_1):
                for j, p_a_m in enumerate(_p_action_1[i]):
                    new_p2_s = dynamics.bvp_dynamics_1d(state_m, action_set[j], dt)
                    new_p1_s = dynamics.bvp_dynamics_1d(state_h, action_set[i], dt)
                    if (theta_h, theta_m) == (1, 5):  # Flip A_AN to NA_A
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[action_set[j]], [action_set[i]]]),
                                             (theta_m, theta_h))  # NA_A
                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[action_set[i]], [action_set[j]]]),
                                             (theta_h, theta_m))
                    lamb_Q1 = q1 * lambda_h
                    _p_action_1[i][j] = lamb_Q1
                    lamb_Q2 = q2 * lambda_m
                    _p_action_2[i][j] = lamb_Q2

            "using logsumexp to prevent nan"
            Q1_logsumexp = logsumexp(_p_action_1)
            Q2_logsumexp = logsumexp(_p_action_2)
            "normalizing"
            _p_action_1 -= Q1_logsumexp
            _p_action_2 -= Q2_logsumexp
            _p_action_1 = np.exp(_p_action_1)
            _p_action_2 = np.exp(_p_action_2)
            assert round(np.sum(_p_action_1)) == 1
            assert round(np.sum(_p_action_2)) == 1
            'p action based on observed action of other agent'
            _pa_1_t = np.transpose(_p_action_1)
            _pa_1 = _pa_1_t[am]  # column of pa
            _pa_2 = _p_action_2[ah]
            _pa_1 /= np.sum(_pa_1)
            _pa_2 /= np.sum(_pa_2)
            assert round(np.sum(_pa_1)) == 1
            assert round(np.sum(_pa_2)) == 1

            p_action_past = [_pa_1[ah], _pa_2[am]]

            return p_action_past, [_p_action_1, _p_action_2]  # [exp_Q_h, exp_Q_m]

        def bvp_action_prob_2(id, p1_state, p2_state, beta_1, beta_2, last_action):
            """
            calculate action prob for one agent: simplifies the calculation
            :param p1_state:
            :param p2_state:
            :param _lambda:
            :param theta:
            :return: p_action of p_i, where p_action = [p_a1, ..., p_a5]
            """

            theta_1, lambda_1 = beta_1
            theta_2, lambda_2 = beta_2
            action_set = self.action_set
            _p_action = np.zeros((len(action_set)))  # 1D

            time = np.array([[self.time]])
            dt = self.sim.dt
            "Need state for agent H: xH, vH, xM, vM"
            if id == 0:
                # p1_state_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])
                new_p2_s = dynamics.bvp_dynamics_1d(p2_state, last_action, dt)  # only one new state for p2
                for i, p_a_h in enumerate(_p_action):
                    new_p1_s = dynamics.bvp_dynamics_1d(p1_state, action_set[i], dt)
                    if (theta_1, theta_2) == (1, 5):  # Flip A_AN to NA_A
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[last_action], [action_set[i]]]),
                                             (theta_2, theta_1))  # NA_A
                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[action_set[i]], [last_action]]),
                                             (theta_1, theta_2))
                    _p_action[i] = q1 * lambda_1
            elif id == 1:  # p2
                # p2_state_nn = np.array([[p2_state[0]], [p2_state[2]], [p1_state[1]], [p1_state[3]]])
                new_p1_s = dynamics.bvp_dynamics_1d(p1_state, last_action, dt)  # only one new state for p2
                for i, p_a_h in enumerate(_p_action):
                    new_p2_s = dynamics.bvp_dynamics_1d(p2_state, action_set[i], dt)
                    if (theta_1, theta_2) == (1, 5):  # Flip A_AN to NA_A
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[action_set[i]], [last_action]]),
                                             (theta_2, theta_1))  # NA_A
                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[last_action], [action_set[i]]]),
                                             (theta_1, theta_2))
                    _p_action[i] = q2 * lambda_2
            else:
                print("WARNING! AGENT COUNT EXCEEDED 2")

            "using logsumexp to prevent nan"
            Q_logsumexp = logsumexp(_p_action)
            "normalizing"
            _p_action -= Q_logsumexp
            _p_action = np.exp(_p_action)

            print("action prob 1 from bvp:", _p_action)
            assert round(np.sum(_p_action)) == 1

            return _p_action  # exp_Q

        def resample(priors, epsilon):
            """
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = self.p_betas_prior
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def prob_beta_pair(prior, traj_h, traj_m):
            """
            MAIN BELIEF UPDATE ALGORITHM
            Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
            :return: P(beta_H, beta_M | D(k)), 4x4 (2 theta, 2 lambda, 2 agents)
            """

            p_betas_d = np.zeros((len(self.beta_set), len(self.beta_set)))
            assert len(p_betas_d) == len(prior)
            beta_set = self.beta_set
            "Calculate prob of beta pair given D(k)"
            past_state_h, last_action_h = traj_h[-1]
            past_state_m, last_action_m = traj_m[-1]
            last_actions = [last_action_h, last_action_m]
            ah = self.action_set.index(last_action_h)
            am = self.action_set.index(last_action_m)
            ai = [ah, am]
            # prior = resample(prior, epsilon=0.05)
            for i in range(len(p_betas_d)):
                for j in range(len(p_betas_d[i])):

                    "method 1: get whole action prob table"
                    # p_a_past, p_action = past_action_prob(past_state_h, past_state_m, beta_set[i], beta_set[j],
                    #                                       last_action_h, last_action_m)  # action prob for last step!

                    "method 2: only get 1 row of p_action for faster speed"
                    p_a_past2 = []
                    for id in range(self.sim.n_agents):
                        p_a_i = bvp_action_prob_2(id, past_state_h, past_state_m, beta_set[i], beta_set[j], last_actions[id-1])
                        p_a_past2.append(p_a_i[ai[id]])
                    "for confirming the two algorithm converges to same value"
                    # assert np.round(p_a_past[0], 4) == np.round(p_a_past2[0], 4)
                    # assert np.round(p_a_past[1], 4) == np.round(p_a_past2[1], 4)
                    p_a_pair = p_a_past2[0] * p_a_past2[1]
                    "P(Q2|D(k)) = P(uH, uM|x(k), QH, QM) * P(Q2|D(k-1)) / sum(~)"
                    p_betas_d[i][j] = p_a_pair * prior[i][j]
            p_betas_d /= np.sum(p_betas_d)
            p_betas_d = resample(p_betas_d, epsilon=0.05)
            assert round(np.sum(p_betas_d)) == 1  # make sure this calculation is correct; no need to normalize
            return p_betas_d

        # function to rearrange for 2D p(theta, lambda|D(k))
        def marginal_joint_intent(id, _p_beta_d):
            """
            Get the marginal P(Beta_i|D(k)) from P(beta_H, beta_M|D(k))
            :param id:
            :param _p_beta_d:
            :return: 4x2 matrix, where row is the lambda and col is the theta, for visualization purposes
            """
            marginal = []
            for t in self.theta_list:
                marginal.append([])
            "create a 2D array of (lambda, theta) pairs distribution like single agent case"
            half = round(len(self.beta_set) / 2)
            if id == 0:  # H agent
                for i, row in enumerate(_p_beta_d):  # get sum of row
                    if i < half:  # in 1D self.beta, first half are NA, or theta1
                        marginal[0].append(sum(row))
                    else:
                        marginal[1].append(sum(row))
            else:
                for i, col in enumerate(zip(*_p_beta_d)):
                    if i < half:
                        marginal[0].append(sum(col))
                    else:
                        marginal[1].append(sum(col))
            # i-4 if i>3
            id_lambda = marginal.index(max(marginal))
            _best_lambda = self.lambda_list[id_lambda] if id_lambda < half else self.lambda_list[id_lambda - half]
            marginal = np.array(marginal)
            marginal = marginal.transpose()  # Lambdas x Thetas
            return marginal, _best_lambda

        "---------------------------------------------------"
        "calling functions: P(Q2|D), P(beta2|D), P(x(k+1)|D)"
        "---------------------------------------------------"
        "get last predicted beta pair"
        # if self.frame == 0:  # initially guess the beta
        #     belief_beta_h, belief_beta_m = self.belief_params
        #     last_beta_h = belief_beta_h
        #     last_beta_m = belief_beta_m
        #     last_theta_h, last_lambda_h = last_beta_h
        #     last_theta_m, last_lambda_m = last_beta_m
        #
        # else:  # get previously predicted beta
        #     last_beta_h, last_beta_m = self.past_beta[-1]
        #     last_theta_h, last_lambda_h = last_beta_h
        #     last_theta_m, last_lambda_m = last_beta_m
        #     "TEST: fixing the lambda to check"
        #     # lambda_h, lambda_m = self.lambdas[-1], self.lambdas[-1]
        #     # beta_h, beta_m = self.betas[-1], self.betas[3]  #H:A, M: NA
        # p_betas_prior = self.sim.initial_belief
        'intent and rationality inference'
        # using traj[-1] to calculate p_action
        p_beta_d = prob_beta_pair(prior=self.p_betas_prior, traj_h=self.traj_h, traj_m=self.traj_m)

        'recording prior for the next step'
        self.p_betas_prior = p_beta_d

        'getting best predicted betas, only for empathetic decision'
        beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
        # print("best betas ID at time {0}".format(self.frame), beta_pair_id)

        new_beta_h = self.beta_set[beta_pair_id[0]]
        new_beta_m = self.beta_set[beta_pair_id[1]]
        self.past_beta.append([new_beta_h, new_beta_m])

        "getting marginal prob for beta_h or beta_m: THIS IS ONLY FOR PLOTTING, NOT DECISION"
        # for estimating distribution
        p_beta_d_h, best_lambda_h = marginal_joint_intent(id=0, _p_beta_d=p_beta_d)
        p_beta_d_m, best_lambda_m = marginal_joint_intent(id=1, _p_beta_d=p_beta_d)

        "getting most likely action for analysis purpose"
        # # these are only for estimation
        # if self.sim.decision_type_m == 'bvp_empathetic':
        #     p_actions = action_prob(curr_state_h, curr_state_m, new_beta_h, new_beta_m)  # for testing with decision
        # elif self.sim.decision_type_m == 'bvp_non_empathetic':
        #     p_action_1, p_action_2_n = bvp_action_prob_2(curr_state_h, curr_state_m, self.true_params[0], new_beta_m)
        #     p_action_1_n, p_action_2 = action_prob(curr_state_h, curr_state_m, new_beta_h,
        #                                            self.true_params[1])  # for testing with decision
        # for i, p_a in enumerate(p_actions):
        #     # p_a_id = np.unravel_index(p_a.argmax(), p_a.shape)  # choose action for self based on the NE
        #     p_a_id = np.argmax(p_a)
        #     predicted_actions.append(self.action_set[p_a_id])
        last_action_set = [last_action_h, last_action_m]
        a_id = [self.action_set.index(last_action_h), self.action_set.index(last_action_m)]  # index of actions_t-1
        predicted_actions = []
        for id in range(self.sim.n_agents):
            if self.sim.decision_type_m == 'bvp_empathetic':
                if id == 0:
                    self.policy_choice[0].append(beta_pair_id[1])
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_m, last_action_set[id - 1])
                elif id == 1:
                    self.policy_choice[1].append(beta_pair_id[0])
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              new_beta_h, self.true_params[1], last_action_set[id - 1])
            elif self.sim.decision_type_m == 'bvp_non_empathetic':
                true_id = self.sim.true_params_id[id]  # true self beta id (ie. NA, NN --> 1)
                if id == 0:
                    p_beta_d_i = p_beta_d[true_id]  # get row based on P1's true beta
                    pred_beta_m_id = np.argmax(p_beta_d_i)
                    pred_beta_m = self.beta_set[pred_beta_m_id]
                    self.policy_choice[0].append(pred_beta_m_id)
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              self.true_params[0], pred_beta_m, last_action_set[id - 1])
                elif id == 1:
                    p_beta_d_i = np.transpose(p_beta_d)[true_id]  # get row based on P1's true beta
                    pred_beta_h_id = np.argmax(p_beta_d_i)
                    pred_beta_h = self.beta_set[pred_beta_h_id]
                    self.policy_choice[1].append(pred_beta_h_id)
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              pred_beta_h, self.true_params[1], last_action_set[id - 1])
            best_action_i = self.action_set[np.argmax(p_a_i)]
            predicted_actions.append(best_action_i)

        "Counting chosen parameter in the belief table"
        self.belief_count[beta_pair_id[0]][beta_pair_id[1]] += 1

        # IMPORTANT: Best beta pair =/= Best beta !!!
        # p_theta_prime, suited_lambdas <- predicted_intent other
        # p_betas: [BH x BM]
        # print("state list and prob for H: ", state_list, marginal_state)
        # print("size of state list at t=1", len(state_list[0]))  # should be 5x5 2D
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # print("-Intent_inf- marginal state H: ", marginal_state_h)
        return {# 'predicted_states_other': (marginal_state_h, get_state_list(curr_state_h, self.T, self.dt)),  # col of 2D should be H
                'predicted_actions_other': predicted_actions[0],
                'predicted_intent_other': [p_beta_d_h, new_beta_h],
                # 'predicted_states_self': (marginal_state_m, get_state_list(curr_state_m, self.T, self.dt)),
                'predicted_actions_self': predicted_actions[1],
                'predicted_intent_self': [p_beta_d_m, new_beta_m],
                'predicted_intent_all': [p_beta_d, [new_beta_h, new_beta_m]],
                'belief_count': self.belief_count,
                'policy_choice': self.policy_choice}

    def bd_empathetic(self, agent, sim):
        """
        Keeping a distribution of belief
        Using Q values from value network trained from BVP solver
        When QH also depends on xM,uM
        :return:P(beta_h, beta_m_hat | D(k))
        """
        # TODO: change inference method
        # ----------------------------#
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # ----------------------------#

        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        self.time = self.sim.time
        assert len(sim.agents[0].state) == len(sim.agents[0].action)
        curr_state_h = sim.agents[0].state[self.frame]
        last_action_h = sim.agents[0].action[self.frame - 1]
        last_state_h = sim.agents[0].state[self.frame - 1]

        curr_state_m = sim.agents[1].state[self.frame]
        last_action_m = sim.agents[1].action[self.frame - 1]
        last_state_m = sim.agents[1].state[self.frame - 1]

        self.traj_h.append([last_state_h, last_action_h])
        self.traj_m.append([last_state_m, last_action_m])

        def bd_action_prob(id, p1_state, p2_state, beta_1, beta_2, last_action):
            """
            belief distribution action probability p(u_i|x, theta)
            calculate action prob for one agent: simplifies the calculation
            :param p1_state:
            :param p2_state:
            :param _lambda:
            :param theta:
            :return: p_action of p_i, where p_action = [p_a1, ..., p_a5]
            """
            # TODO: implement/import the new belief Q function
            theta_1, lambda_1 = beta_1
            theta_2, lambda_2 = beta_2
            action_set = self.action_set
            _p_action = np.zeros((len(action_set)))  # 1D

            time = np.array([[self.time]])
            dt = self.sim.dt
            "Need state for agent H: xH, vH, xM, vM"
            if id == 0:
                # p1_state_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])
                new_p2_s = dynamics.bvp_dynamics_1d(p2_state, last_action, dt)  # only one new state for p2
                for i, p_a_h in enumerate(_p_action):
                    new_p1_s = dynamics.bvp_dynamics_1d(p1_state, action_set[i], dt)
                    if (theta_1, theta_2) == (1, 5):  # Flip A_AN to NA_A  # TODO: no flipping needed for new Q func?
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        # q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[last_action], [action_set[i]]]),
                        #                      (theta_2, theta_1))  # NA_A
                        "Q value based on belief distribution b_i"
                        q1 = "new_Q(new_p1_state_nn, time, np.array([[action_set[i]]), theta_1, belief)"

                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        # q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[action_set[i]], [last_action]]),
                        #                      (theta_1, theta_2))
                        "Q value based on belief distribution b_i"
                        q1 = "new_Q(new_p1_state_nn, time, np.array([[action_set[i]]), theta_1, belief)"
                    _p_action[i] = q1 * lambda_1
            elif id == 1:  # p2
                # p2_state_nn = np.array([[p2_state[0]], [p2_state[2]], [p1_state[1]], [p1_state[3]]])
                new_p1_s = dynamics.bvp_dynamics_1d(p1_state, last_action, dt)  # only one new state for p2
                for i, p_a_h in enumerate(_p_action):
                    new_p2_s = dynamics.bvp_dynamics_1d(p2_state, action_set[i], dt)
                    if (theta_1, theta_2) == (1, 5):  # Flip A_AN to NA_A TODO: no more flipping needed??
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        # q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[action_set[i]], [last_action]]),
                        #                      (theta_2, theta_1))  # NA_A
                        "Q value based on belief distribution b_i"
                        q2 = "new_Q(new_p2_state_nn, time, np.array([[action_set[i]]), theta_2, belief)"
                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        # q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[last_action], [action_set[i]]]),
                        #                      (theta_1, theta_2))
                        "Q value based on belief distribution b_i"
                        q2 = "new_Q(new_p2_state_nn, time, np.array([[action_set[i]]), theta_2, belief)"
                    _p_action[i] = q2 * lambda_2
            else:
                print("WARNING! AGENT COUNT EXCEEDED 2")

            "using logsumexp to prevent nan"
            Q_logsumexp = logsumexp(_p_action)
            "normalizing"
            _p_action -= Q_logsumexp
            _p_action = np.exp(_p_action)

            print("action prob 1 from bvp:", _p_action)
            assert round(np.sum(_p_action)) == 1

            return _p_action  # exp_Q

        def resample(priors, epsilon):
            """
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = self.p_betas_prior
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def prob_beta_pair(prior, traj_h, traj_m):
            """
            MAIN BELIEF UPDATE ALGORITHM
            Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
            :return: P(beta_i | D(k)), 2x4
            """

            p_betas_d = np.zeros((self.n_agents, len(self.beta_set)))  # 2x4
            assert len(p_betas_d) == len(prior)
            beta_set = self.beta_set
            "Calculate prob of beta pair given D(k)"
            past_state_h, last_action_h = traj_h[-1]
            past_state_m, last_action_m = traj_m[-1]
            last_actions = [last_action_h, last_action_m]
            ah = self.action_set.index(last_action_h)
            am = self.action_set.index(last_action_m)
            ai = [ah, am]
            # prior = resample(prior, epsilon=0.05)

            "updating belief table"
            for i in range(len(beta_set)):
                id = 0
                p_a_i = bd_action_prob(id, past_state_h, past_state_m, beta_set[i], beta_set[i],
                                       last_actions[id - 1])  # TODO: not implemented yet
                p_betas_d[0][i] = p_a_i * prior[0][i]
            p_betas_d[0] /= np.sum(p_betas_d[0])
            p_betas_d[0] = resample(p_betas_d[0], epsilon=0.05)  # resampling from the initial prior
            for i in range(len(beta_set)):
                id = 1
                p_a_i = bd_action_prob(id, past_state_h, past_state_m, beta_set[i], beta_set[i],
                                       last_actions[id - 1])  # TODO: not implemented yet
                p_betas_d[id][i] = p_a_i * prior[id][i]
            p_betas_d[1] /= np.sum(p_betas_d[1])
            p_betas_d[1] = resample(p_betas_d[1], epsilon=0.05)  # resampling from the initial prior

            # for i in range(len(p_betas_d)):
            #     for j in range(len(p_betas_d[i])):
            #
            #         "method 1: get whole action prob table"
            #         # p_a_past, p_action = past_action_prob(past_state_h, past_state_m, beta_set[i], beta_set[j],
            #         #                                       last_action_h, last_action_m)  # action prob for last step!
            #
            #         "method 2: only get 1 row of p_action for faster speed"
            #         p_a_past2 = []
            #         for id in range(self.sim.n_agents):
            #             p_a_i = bd_action_prob(id, past_state_h, past_state_m, beta_set[i], beta_set[j], last_actions[id-1])  # TODO: not implemented yet
            #             p_a_past2.append(p_a_i[ai[id]])
            #         "for confirming the two algorithm converges to same value"
            #         # assert np.round(p_a_past[0], 4) == np.round(p_a_past2[0], 4)
            #         # assert np.round(p_a_past[1], 4) == np.round(p_a_past2[1], 4)
            #         p_a_pair = p_a_past2[0] * p_a_past2[1]
            #         "P(Q2|D(k)) = P(uH, uM|x(k), QH, QM) * P(Q2|D(k-1)) / sum(~)"
            #         p_betas_d[i][j] = p_a_pair * prior[i][j]

            assert round(np.sum(p_betas_d[0])) == 1  # make sure this calculation is correct; no need to normalize
            assert round(np.sum(p_betas_d[1])) == 1
            return p_betas_d

        # function to rearrange for 2D p(theta, lambda|D(k))
        def marginal_joint_intent(id, _p_beta_d):
            """
            Get the marginal P(Beta_i|D(k)) from P(beta_H, beta_M|D(k))
            :param id:
            :param _p_beta_d:
            :return: 4x2 matrix, where row is the lambda and col is the theta, for visualization purposes
            """
            marginal = []
            for t in self.theta_list:
                marginal.append([])
            "create a 2D array of (lambda, theta) pairs distribution like single agent case"
            half = round(len(self.beta_set) / 2)
            if id == 0:  # H agent
                for i, row in enumerate(_p_beta_d):  # get sum of row
                    if i < half:  # in 1D self.beta, first half are NA, or theta1
                        marginal[0].append(sum(row))
                    else:
                        marginal[1].append(sum(row))
            else:
                for i, col in enumerate(zip(*_p_beta_d)):
                    if i < half:
                        marginal[0].append(sum(col))
                    else:
                        marginal[1].append(sum(col))
            # i-4 if i>3
            id_lambda = marginal.index(max(marginal))
            _best_lambda = self.lambda_list[id_lambda] if id_lambda < half else self.lambda_list[id_lambda - half]
            marginal = np.array(marginal)
            marginal = marginal.transpose()  # Lambdas x Thetas
            return marginal, _best_lambda

        def belief_marginal(belief_table):
            """
            calculates marginal b_i from common belief table
            :return: _b_i, probabilities of each agent being aggressive
            """
            _b_i = np.zeros(self.n_agents)  # 2x1 probabilities of each agent being aggressive
            # _b_i[0] = np.sum(belief_table[0]) + np.sum(belief_table[1])
            _b_i[0] = np.sum(belief_table[0][2]) + np.sum(belief_table[0][3])
            belief_table_t = np.transpose(belief_table)
            _b_i[1] = np.sum(belief_table[1][2]) + np.sum(belief_table[1][3])

            return _b_i

        "---------------------------------------------------"
        "calling functions: P(Q2|D), P(beta2|D), P(x(k+1)|D)"
        "---------------------------------------------------"
        "get last predicted beta pair"
        # if self.frame == 0:  # initially guess the beta
        #     belief_beta_h, belief_beta_m = self.belief_params
        #     last_beta_h = belief_beta_h
        #     last_beta_m = belief_beta_m
        #     last_theta_h, last_lambda_h = last_beta_h
        #     last_theta_m, last_lambda_m = last_beta_m
        #
        # else:  # get previously predicted beta
        #     last_beta_h, last_beta_m = self.past_beta[-1]
        #     last_theta_h, last_lambda_h = last_beta_h
        #     last_theta_m, last_lambda_m = last_beta_m
        #     "TEST: fixing the lambda to check"
        #     # lambda_h, lambda_m = self.lambdas[-1], self.lambdas[-1]
        #     # beta_h, beta_m = self.betas[-1], self.betas[3]  #H:A, M: NA
        # p_betas_prior = self.sim.initial_belief
        'UPDATING COMMON BELIEF TABLE: intent and rationality inference'
        # using traj[-1] to calculate p_action
        p_beta_d = prob_beta_pair(prior=self.p_betas_prior, traj_h=self.traj_h, traj_m=self.traj_m)

        'Get the marginal belief b_i, as discussed in the paper:'
        # TODO: implement a function that calculate the marginal b_i from the common belief p_beta_d
        b_i = belief_marginal(p_beta_d)
        self.prior_b_i = b_i

        'recording prior for the next step'
        self.p_betas_prior = p_beta_d

        'getting best predicted betas, ONLY WORKS for empathetic decision'
        beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
        # print("best betas ID at time {0}".format(self.frame), beta_pair_id)

        new_beta_h = self.beta_set[beta_pair_id[0]]
        new_beta_m = self.beta_set[beta_pair_id[1]]
        self.past_beta.append([new_beta_h, new_beta_m])

        "getting marginal prob for beta_h or beta_m: THIS IS ONLY FOR PLOTTING, NOT DECISION"
        # for estimating distribution
        p_beta_d_h, best_lambda_h = marginal_joint_intent(id=0, _p_beta_d=p_beta_d)
        p_beta_d_m, best_lambda_m = marginal_joint_intent(id=1, _p_beta_d=p_beta_d)

        "getting most likely action for analysis purpose"
        last_action_set = [last_action_h, last_action_m]
        a_id = [self.action_set.index(last_action_h), self.action_set.index(last_action_m)]  # index of actions_t-1
        predicted_actions = []
        # TODO: belief distribution action prob not implemented yet
        for id in range(self.sim.n_agents):
            if self.sim.decision_type_m == 'bvp_empathetic':
                if id == 0:
                    self.policy_choice[0].append(beta_pair_id[1])
                    p_a_i = bd_action_prob(id, curr_state_h, curr_state_m,
                                           self.true_params[0], new_beta_m, last_action_set[id - 1])
                elif id == 1:
                    self.policy_choice[1].append(beta_pair_id[0])
                    p_a_i = bd_action_prob(id, curr_state_h, curr_state_m,
                                           new_beta_h, self.true_params[1], last_action_set[id - 1])
            elif self.sim.decision_type_m == 'bvp_non_empathetic':
                true_id = self.sim.true_params_id[id]  # true self beta id (ie. NA, NN --> 1)
                if id == 0:
                    p_beta_d_i = p_beta_d[true_id]  # get row based on P1's true beta
                    pred_beta_m_id = np.argmax(p_beta_d_i)
                    pred_beta_m = self.beta_set[pred_beta_m_id]
                    self.policy_choice[0].append(pred_beta_m_id)
                    p_a_i = bd_action_prob(id, curr_state_h, curr_state_m,
                                              self.true_params[0], pred_beta_m, last_action_set[id - 1])
                elif id == 1:
                    p_beta_d_i = np.transpose(p_beta_d)[true_id]  # get row based on P1's true beta
                    pred_beta_h_id = np.argmax(p_beta_d_i)
                    pred_beta_h = self.beta_set[pred_beta_h_id]
                    self.policy_choice[1].append(pred_beta_h_id)
                    p_a_i = bd_action_prob(id, curr_state_h, curr_state_m,
                                              pred_beta_h, self.true_params[1], last_action_set[id - 1])
            best_action_i = self.action_set[np.argmax(p_a_i)]
            predicted_actions.append(best_action_i)

        "Counting chosen parameter in the belief table"
        self.belief_count[beta_pair_id[0]][beta_pair_id[1]] += 1

        # IMPORTANT: Best beta pair =/= Best beta !!!
        # p_theta_prime, suited_lambdas <- predicted_intent other
        # p_betas: [BH x BM]
        # print("state list and prob for H: ", state_list, marginal_state)
        # print("size of state list at t=1", len(state_list[0]))  # should be 5x5 2D
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # print("-Intent_inf- marginal state H: ", marginal_state_h)
        return {'predicted_actions_other': predicted_actions[0],
                'predicted_intent_other': [p_beta_d_h, b_i[0]],
                'predicted_actions_self': predicted_actions[1],
                'predicted_intent_self': [p_beta_d_m, b_i[1]],
                'predicted_intent_all': [p_beta_d, b_i],
                'belief_count': self.belief_count,
                'policy_choice': self.policy_choice}

    def bd_non_empathetic(self, agent, sim):
        """
        Keeping a distribution of belief
        Using Q values from value network trained from BVP solver
        When QH also depends on xM,uM
        :return:P(beta_h, beta_m_hat | D(k))
        """
        # TODO: change inference method
        # ----------------------------#
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # ----------------------------#

        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        self.time = self.sim.time
        assert len(sim.agents[0].state) == len(sim.agents[0].action)
        curr_state_h = sim.agents[0].state[self.frame]
        last_action_h = sim.agents[0].action[self.frame - 1]
        last_state_h = sim.agents[0].state[self.frame - 1]

        curr_state_m = sim.agents[1].state[self.frame]
        last_action_m = sim.agents[1].action[self.frame - 1]
        last_state_m = sim.agents[1].state[self.frame - 1]

        self.traj_h.append([last_state_h, last_action_h])
        self.traj_m.append([last_state_m, last_action_m])

        def bd_action_prob(id, p1_state, p2_state, beta_1, beta_2, last_action):
            """
            belief distribution action probability p(u_i|x, theta)
            calculate action prob for one agent: simplifies the calculation
            :param p1_state:
            :param p2_state:
            :param _lambda:
            :param theta:
            :return: p_action of p_i, where p_action = [p_a1, ..., p_a5]
            """
            # TODO: implement/import the new belief Q function
            theta_1, lambda_1 = beta_1
            theta_2, lambda_2 = beta_2
            action_set = self.action_set
            _p_action = np.zeros((len(action_set)))  # 1D

            time = np.array([[self.time]])
            dt = self.sim.dt
            "Need state for agent H: xH, vH, xM, vM"
            if id == 0:
                # p1_state_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])
                new_p2_s = dynamics.bvp_dynamics_1d(p2_state, last_action, dt)  # only one new state for p2
                for i, p_a_h in enumerate(_p_action):
                    new_p1_s = dynamics.bvp_dynamics_1d(p1_state, action_set[i], dt)
                    if (theta_1, theta_2) == (1, 5):  # Flip A_AN to NA_A  # TODO: no flipping needed for new Q func?
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        # q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[last_action], [action_set[i]]]),
                        #                      (theta_2, theta_1))  # NA_A
                        "Q value based on belief distribution b_i"
                        q1 = "new_Q(new_p1_state_nn, time, np.array([[action_set[i]]), theta_1, belief)"

                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        # q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[action_set[i]], [last_action]]),
                        #                      (theta_1, theta_2))
                        "Q value based on belief distribution b_i"
                        q1 = "new_Q(new_p1_state_nn, time, np.array([[action_set[i]]), theta_1, belief)"
                    _p_action[i] = q1 * lambda_1
            elif id == 1:  # p2
                # p2_state_nn = np.array([[p2_state[0]], [p2_state[2]], [p1_state[1]], [p1_state[3]]])
                new_p1_s = dynamics.bvp_dynamics_1d(p1_state, last_action, dt)  # only one new state for p2
                for i, p_a_h in enumerate(_p_action):
                    new_p2_s = dynamics.bvp_dynamics_1d(p2_state, action_set[i], dt)
                    if (theta_1, theta_2) == (1, 5):  # Flip A_AN to NA_A TODO: no more flipping needed??
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        # q2, q1 = get_Q_value(new_p2_state_nn, time, np.array([[action_set[i]], [last_action]]),
                        #                      (theta_2, theta_1))  # NA_A
                        "Q value based on belief distribution b_i"
                        q2 = "new_Q(new_p2_state_nn, time, np.array([[action_set[i]]), theta_2, belief)"
                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        # q1, q2 = get_Q_value(new_p1_state_nn, time, np.array([[last_action], [action_set[i]]]),
                        #                      (theta_1, theta_2))
                        "Q value based on belief distribution b_i"
                        q2 = "new_Q(new_p2_state_nn, time, np.array([[action_set[i]]), theta_2, belief)"
                    _p_action[i] = q2 * lambda_2
            else:
                print("WARNING! AGENT COUNT EXCEEDED 2")

            "using logsumexp to prevent nan"
            Q_logsumexp = logsumexp(_p_action)
            "normalizing"
            _p_action -= Q_logsumexp
            _p_action = np.exp(_p_action)

            print("action prob 1 from bvp:", _p_action)
            assert round(np.sum(_p_action)) == 1

            return _p_action  # exp_Q

        def resample(priors, epsilon):
            """
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = self.p_betas_prior
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def prob_beta_pair(prior, traj_h, traj_m):
            """
            MAIN BELIEF UPDATE ALGORITHM: belief dsitri NE version
            Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
            :return: P(beta_i | D(k)), 2x4
            """

            p_betas_d = np.zeros((self.n_agents, len(self.beta_set)))  # 2x4
            assert len(p_betas_d) == len(prior)
            beta_set = self.beta_set
            "Calculate prob of beta pair given D(k)"
            past_state_h, last_action_h = traj_h[-1]
            past_state_m, last_action_m = traj_m[-1]
            last_actions = [last_action_h, last_action_m]
            ah = self.action_set.index(last_action_h)
            am = self.action_set.index(last_action_m)
            ai = [ah, am]
            # prior = resample(prior, epsilon=0.05)

            "updating belief table"
            for i in range(len(beta_set)):
                id = 0  # update ONLY agent 1 in agent 0's table
                p_a_i = bd_action_prob(id, past_state_h, past_state_m, beta_set[i], beta_set[i],
                                       last_actions[id - 1])  # TODO: not implemented yet
                p_betas_d[0][1][i] = p_a_i * prior[0][1][i]
            p_betas_d[0][1] /= np.sum(p_betas_d)
            p_betas_d[0][1] = resample(p_betas_d[0][1], epsilon=0.05)  # resampling from the initial prior

            for i in range(len(beta_set)):
                id = 1  # update ONLY agent 0 in agent 1's table
                p_a_i = bd_action_prob(id, past_state_h, past_state_m, beta_set[i], beta_set[i],
                                       last_actions[id - 1])  # TODO: not implemented yet
                p_betas_d[id][id-1][i] = p_a_i * prior[id][id-1][i]
            p_betas_d[1][0] /= np.sum(p_betas_d)
            p_betas_d[1][0] = resample(p_betas_d[1][0], epsilon=0.05)  # resampling from the initial prior

            # for i in range(len(p_betas_d)):
            #     for j in range(len(p_betas_d[i])):
            #
            #         "method 1: get whole action prob table"
            #         # p_a_past, p_action = past_action_prob(past_state_h, past_state_m, beta_set[i], beta_set[j],
            #         #                                       last_action_h, last_action_m)  # action prob for last step!
            #
            #         "method 2: only get 1 row of p_action for faster speed"
            #         p_a_past2 = []
            #         for id in range(self.sim.n_agents):
            #             p_a_i = bd_action_prob(id, past_state_h, past_state_m, beta_set[i], beta_set[j], last_actions[id-1])  # TODO: not implemented yet
            #             p_a_past2.append(p_a_i[ai[id]])
            #         "for confirming the two algorithm converges to same value"
            #         # assert np.round(p_a_past[0], 4) == np.round(p_a_past2[0], 4)
            #         # assert np.round(p_a_past[1], 4) == np.round(p_a_past2[1], 4)
            #         p_a_pair = p_a_past2[0] * p_a_past2[1]
            #         "P(Q2|D(k)) = P(uH, uM|x(k), QH, QM) * P(Q2|D(k-1)) / sum(~)"
            #         p_betas_d[i][j] = p_a_pair * prior[i][j]
            # p_betas_d /= np.sum(p_betas_d)
            # p_betas_d = resample(p_betas_d, epsilon=0.05)  # resampling from the initial prior
            assert round(np.sum(p_betas_d[0][1])) == 1  # make sure this calculation is correct; no need to normalize
            assert round(np.sum(p_betas_d[1][0])) == 1  # make sure this calculation is correct; no need to normalize
            return p_betas_d

        # function to rearrange for 2D p(theta, lambda|D(k))
        def marginal_joint_intent(id, _p_beta_d):
            """
            Get the marginal P(Beta_i|D(k)) from P(beta_H, beta_M|D(k))
            :param id:
            :param _p_beta_d:
            :return: 4x2 matrix, where row is the lambda and col is the theta, for visualization purposes
            """
            marginal = []
            for t in self.theta_list:
                marginal.append([])
            "create a 2D array of (lambda, theta) pairs distribution like single agent case"
            half = round(len(self.beta_set) / 2)
            if id == 0:  # H agent
                for i, row in enumerate(_p_beta_d):  # get sum of row
                    if i < half:  # in 1D self.beta, first half are NA, or theta1
                        marginal[0].append(sum(row))
                    else:
                        marginal[1].append(sum(row))
            else:
                for i, col in enumerate(zip(*_p_beta_d)):
                    if i < half:
                        marginal[0].append(sum(col))
                    else:
                        marginal[1].append(sum(col))
            # i-4 if i>3
            id_lambda = marginal.index(max(marginal))
            _best_lambda = self.lambda_list[id_lambda] if id_lambda < half else self.lambda_list[id_lambda - half]
            marginal = np.array(marginal)
            marginal = marginal.transpose()  # Lambdas x Thetas
            return marginal, _best_lambda

        def belief_marginal(belief_table):
            """
            calculates marginal b_i from common belief table
            :return: _b_i, probabilities of each agent being aggressive
            """
            _b_i = np.zeros(self.n_agents)  # 2x1 probabilities of each agent being aggressive
            # _b_i[0] = np.sum(belief_table[0]) + np.sum(belief_table[1])
            _b_i[0] = np.sum(belief_table[1][0][2]) + np.sum(belief_table[1][0][3])  # 0's belief on 1 being A
            belief_table_t = np.transpose(belief_table)
            _b_i[1] = np.sum(belief_table[0][1][2]) + np.sum(belief_table[0][1][3])

            return _b_i

        "---------------------------------------------------"
        "calling functions: P(Q2|D), P(beta2|D), P(x(k+1)|D)"
        "---------------------------------------------------"
        "get last predicted beta pair"
        # if self.frame == 0:  # initially guess the beta
        #     belief_beta_h, belief_beta_m = self.belief_params
        #     last_beta_h = belief_beta_h
        #     last_beta_m = belief_beta_m
        #     last_theta_h, last_lambda_h = last_beta_h
        #     last_theta_m, last_lambda_m = last_beta_m
        #
        # else:  # get previously predicted beta
        #     last_beta_h, last_beta_m = self.past_beta[-1]
        #     last_theta_h, last_lambda_h = last_beta_h
        #     last_theta_m, last_lambda_m = last_beta_m
        #     "TEST: fixing the lambda to check"
        #     # lambda_h, lambda_m = self.lambdas[-1], self.lambdas[-1]
        #     # beta_h, beta_m = self.betas[-1], self.betas[3]  #H:A, M: NA
        # p_betas_prior = self.sim.initial_belief
        'UPDATING COMMON BELIEF TABLE: intent and rationality inference'
        # using traj[-1] to calculate p_action
        p_beta_d = prob_beta_pair(prior=self.p_betas_prior, traj_h=self.traj_h, traj_m=self.traj_m)

        'Get the marginal belief b_i, as discussed in the paper:'
        # TODO: implement a function that calculate the marginal b_i from the common belief p_beta_d
        b_i = belief_marginal(p_beta_d)
        self.prior_b_i = b_i

        'recording prior for the next step'
        self.p_betas_prior = p_beta_d

        'getting best predicted betas, ONLY WORKS for empathetic decision'
        beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
        # print("best betas ID at time {0}".format(self.frame), beta_pair_id)

        new_beta_h = self.beta_set[beta_pair_id[0]]
        new_beta_m = self.beta_set[beta_pair_id[1]]
        self.past_beta.append([new_beta_h, new_beta_m])

        "getting marginal prob for beta_h or beta_m: THIS IS ONLY FOR PLOTTING, NOT DECISION"
        # for estimating distribution
        p_beta_d_h, best_lambda_h = marginal_joint_intent(id=0, _p_beta_d=p_beta_d)
        p_beta_d_m, best_lambda_m = marginal_joint_intent(id=1, _p_beta_d=p_beta_d)

        "getting most likely action for analysis purpose"
        last_action_set = [last_action_h, last_action_m]
        a_id = [self.action_set.index(last_action_h), self.action_set.index(last_action_m)]  # index of actions_t-1
        predicted_actions = []
        # TODO: belief distribution action prob not implemented yet
        for id in range(self.sim.n_agents):
            if self.sim.decision_type_m == 'bvp_empathetic':
                if id == 0:
                    self.policy_choice[0].append(beta_pair_id[1])
                    p_a_i = bd_action_prob(id, curr_state_h, curr_state_m,
                                           self.true_params[0], new_beta_m, last_action_set[id - 1])
                elif id == 1:
                    self.policy_choice[1].append(beta_pair_id[0])
                    p_a_i = bd_action_prob(id, curr_state_h, curr_state_m,
                                           new_beta_h, self.true_params[1], last_action_set[id - 1])
            elif self.sim.decision_type_m == 'bvp_non_empathetic':
                true_id = self.sim.true_params_id[id]  # true self beta id (ie. NA, NN --> 1)
                if id == 0:
                    p_beta_d_i = p_beta_d[true_id]  # get row based on P1's true beta
                    pred_beta_m_id = np.argmax(p_beta_d_i)
                    pred_beta_m = self.beta_set[pred_beta_m_id]
                    self.policy_choice[0].append(pred_beta_m_id)
                    p_a_i = bd_action_prob(id, curr_state_h, curr_state_m,
                                              self.true_params[0], pred_beta_m, last_action_set[id - 1])
                elif id == 1:
                    p_beta_d_i = np.transpose(p_beta_d)[true_id]  # get row based on P1's true beta
                    pred_beta_h_id = np.argmax(p_beta_d_i)
                    pred_beta_h = self.beta_set[pred_beta_h_id]
                    self.policy_choice[1].append(pred_beta_h_id)
                    p_a_i = bd_action_prob(id, curr_state_h, curr_state_m,
                                              pred_beta_h, self.true_params[1], last_action_set[id - 1])
            best_action_i = self.action_set[np.argmax(p_a_i)]
            predicted_actions.append(best_action_i)

        "Counting chosen parameter in the belief table"
        self.belief_count[beta_pair_id[0]][beta_pair_id[1]] += 1

        # IMPORTANT: Best beta pair =/= Best beta !!!
        # p_theta_prime, suited_lambdas <- predicted_intent other
        # p_betas: [BH x BM]
        # print("state list and prob for H: ", state_list, marginal_state)
        # print("size of state list at t=1", len(state_list[0]))  # should be 5x5 2D
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # print("-Intent_inf- marginal state H: ", marginal_state_h)
        return {# 'predicted_states_other': (marginal_state_h, get_state_list(curr_state_h, self.T, self.dt)),  # col of 2D should be H
                'predicted_actions_other': predicted_actions[0],
                'predicted_intent_other': [p_beta_d_h, new_beta_h],
                # 'predicted_states_self': (marginal_state_m, get_state_list(curr_state_m, self.T, self.dt)),
                'predicted_actions_self': predicted_actions[1],
                'predicted_intent_self': [p_beta_d_m, new_beta_m],
                'predicted_intent_all': [p_beta_d, [new_beta_h, new_beta_m]],
                'belief_count': self.belief_count,
                'policy_choice': self.policy_choice}


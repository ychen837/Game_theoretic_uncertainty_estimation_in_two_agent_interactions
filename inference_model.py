"""
Perform inference on other agent
Baseline: using the most basic Q function estimation, on how fast goal is reached
          (need to change main.py, environment = intersection)
Trained_baseline: using NFSP model as Q function (change environment = trained_intersection)
Empathetic: with NFSP, perform inference on both agent (change environment = trained_intersection)
"""
import torch
import numpy as np
from sklearn.preprocessing import normalize
from models.rainbow.set_nfsp_models import get_models
# TODO pytorch version
from autonomous_vehicle import AutonomousVehicle
import discrete_sets as sets

#class Lambdas(FloatEnums):

class InferenceModel:
    def __init__(self, model, sim): #model = inference type, sim = simulation class
        if model == 'none':
            self.infer = self.no_inference
        elif model == 'baseline':
            self.infer = self.baseline_inference
        elif model == 'trained_baseline':
            self.infer = self.trained_baseline_inference
        elif model == 'empathetic':
            self.infer = self.empathetic_inference
        else:
            # placeholder for future development
            pass

        "---simulation info: only the static variables!---"
        self.sim = sim
        self.agents = sim.agents
        self.n_agents = sim.n_agents
        self.frame = sim.frame
        self.T = 1  # one step look ahead/ Time Horizon
        self.dt = sim.dt #default is 1s: assigned in main.py
        self.car_par = sim.env.car_par
        # self.min_speed = sim.agents[0].min_speed
        # self.max_speed = sim.agents[0].max_speed
        self.min_speed = 0.1
        self.max_speed = 30
        "---goal states---"
        self.goal = [self.car_par[0]["desired_state"], self.car_par[1]["desired_state"]]

        "---parameters(theta and lambda)---"
        self.lambdas = [0.001, 0.005, 0.01, 0.05]  # range?
        #self.lambdas = [0.05, 0.1, 1, 10]
        self.thetas = [1, 1000]  # range?

        "---Params for belief calculation---"
        self.past_scores = {} #score for each lambda
        self.past_scores1 = {} #for theta1
        self.past_scores2 = {} #for theta2
        #self.theta_priors = None #for calculating theta lambda joint probability
        self.theta_priors = self.sim.theta_priors
        self.initial_joint_prob = np.ones((len(self.lambdas), len(self.thetas))) / (len(self.lambdas) * len(self.thetas)) #do this here to increase speed


        self.traj_h = []
        self.traj_m = []

        #self.action_set = [-1, -0.2, 0.0, 0.1, 0.5]  # accelerations (m/s^2)
        self.action_set = [-8, -4, 0.0, 4, 8]  # from realistic trained model
        #  safe deceleration: 15ft/s

        "for empathetic inference:" #TODO: add necessary data
        self.p_betas_prior = None
        self.q2_prior = None
        self.past_beta = []
        # beta: [theta1, lambda1], [theta1, lambda2], ... [theta2, lambda4] (2x4 = 8 set of betas)
        # betas: [ [[theta1, lambda1], [theta1, lambda2], [theta1, lambda3], [theta1, lambda4]],
        #          [[theta2, lambda1], [theta2, lambda2], [theta2, lambda3], [theta2, lambda4]] ]
        self.betas = []
        '2D version of beta'
        # for i, theta in enumerate(self.thetas):
        #     self.betas.append([])
        #     for j, _lambda in enumerate(self.lambdas):
        #         self.betas[i].append([theta, _lambda])
        '1D version of beta'
        for i, theta in enumerate(self.thetas):
            for j, _lambda in enumerate(self.lambdas):
                self.betas.append([theta, _lambda])


        "---Agent info---"
        # "importing initial agents information"
        # self.initial_state = [self.car_par[0]["initial_state"], self.car_par[1]["initial_state"]]
        # if self.frame == 0: #no agent info yet
        #     self.actions = [self.car_par[0]["initial_action"],self.car_par[1]["initial_action"]]
        #     self.curr_state = [self.car_par[0]["initial_state"], self.car_par[1]["initial_state"]]
        #     self.traj_h = [self.curr_state[0], self.actions[0]]
        #     self.traj_m = [self.curr_state[1], self.actions[1]]
        #     self.traj = [[self.curr_state[0], self.curr_state[1]],[self.actions[0], self.actions[1]]]

        # else:
        #     "importing agents information from Autonomous Vehicle (sim.agents)"
        #     for i in range(self.n_agents):
        #         if i == 0:
        #             self.curr_state_h = self.sim.agents[i].state[-1]
        #         if i == 1:
        #             self.curr_state_m = self.sim.agents[i].state[-1]
        #     self.curr_state = self.sim.agents.state[-1]
        #

        #     self.actions = [-2, -0.5, 0, 0.5, 2]  # accelerations (m/s^2)
        #
        #     "trajectory: need to process and combine past states and actions together"
        #     self.states_h = self.sim.agents[0].state
        #     self.states_m = self.sim.agents[1].state
        #     self.past_actions_h = sim.agents[0].action
        #     self.past_actions_m = sim.agents[1].action
        #     self.traj = []
        #     self.traj_h = []
        #     for i in range(self.frame):
        #         self.traj.append([[self.states_h[i],self.states_m[i]],
        #                           [self.past_actions_h[i], self.past_actions_m[i]]])
        #         self.traj_h.append([self.states_h[i], self.past_actions_h[i]])
        # # self.traj = AutonomousVehicle.planned_trajectory_set
        "---end of agent info---"



        #"--------------------------------------------------------"
        "Some reference variables from pedestrian prediction"
        #self.q_cache = {}
        #"defining reward for (s, a) pair"
        #self.default_reward = -1
        #self.rewards = np.zeros[self.agents.s, self.agents.a] #Needs fixing
        #self.rewards.fill(self.default_reward)
        #"--------------------------------------------------------"

    #@staticmethod
    def no_inference(self, agents, sim):
        #pass
        print("frame {}".format(sim.frame))
        return

    #@staticmethod
    def baseline_inference(self, agents, sim):
        # TODO: implement Fridovich-Keil et al. "Confidence-aware motion prediction for real-time collision avoidance"
        """
        for each agent, estimate its par (agent.par) based on its last action (agent.action[-1]),
        system state (agent.state[-1] for all agent), and prior dist. of par
        :return: #TODO: check what to return

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
        curr_state_h = sim.agents[0].state[-1]

        "trajectory: need to process and combine past states and actions together"
        states_h = sim.agents[0].state
        past_actions_h = sim.agents[0].action
        traj_h = []

        for i in range(sim.frame):
            traj_h.append([states_h[i], past_actions_h[i]])
        if sim.frame == 0:
            traj_h.append([states_h[0], past_actions_h[0]])
        "---end of agent info---"
        self.frame = sim.frame #updating curr frame count

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
            else: #TODO: Check for the case of 2D movement
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
                # TODO: add a ceiling for how fast they can go
            return Q

        def q_values(state, goal):
            # TODO documentation for function
            """
            Calls q_function and return a list of q values corresponding to an action set at a given state
            :param state:
            :param goal:
            return:
                A 1D list of values for a given state s with action set A


            """
            #print("q_values function is being called,{0}, {1}".format(state, goal))
            # current_s = states[-1]
            # Q = {} #dict type
            Q = []  # list type
            actions = self.action_set  # TODO: check that actions are imported in init
            for a in actions:  # sets: file for defining sets
                # Q[a] = q_function(state, a, goal, self.dt)  #dict type
                Q.append(q_function(state, a, goal, self.dt))  # list type

            return Q


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

                def calc_state(x, u, dt):
                    sx, sy, vx, vy = x[0], x[1], x[2], x[3]
                    "Deceleration only leads to zero velocity!"
                    if sx == 0 and vx == 0:  # y axis movement
                        vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vy_new < 0:
                            vy_new = 0
                        sx_new = sx  # + (vx + vx_new) * dt * 0.5
                        sy_new = sy + (vy + vy_new) * dt * 0.5
                    elif sy == 0 and vy == 0:  # x axis movement
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy  # + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vx_new < 0:
                            vx_new = 0
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy  # + (vy + vy_new) * dt * 0.5
                    else:  # TODO: assume x axis movement for single agent case!!
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy  # + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vx_new < 0:
                            vx_new = 0
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy  # + (vy + vy_new) * dt * 0.5

                    # TODO: add a cieling for how fast they can go
                    return sx_new, sy_new, vx_new, vy_new

                "Checking if _states is composed of tuples of state info (initially _state is just a tuple)"
                if not isinstance(_state_list[0], tuple):
                    print("WARNING: state list is not composed of tuples!")
                    _state_list = [_state_list] #put it in a list to iterate

                for s in _state_list:
                    for a in _actions:
                        #print("STATE", s)
                        _s_prime.append(calc_state(s, a, dt))
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
            refer to mdp.py
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
            # exp_Q_list[i] = exp_Q

            # return exp_Q_list #array of exp_Q for an array of states


        def traj_probabilities(state, _lambda, dt, prior = None):
            # TODO: What does summarizing over x(k) and u(k) do?
            """
            refer to mdp.py
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

            #TODO: verify if it is working properly (plotting states? p_state seems correct)
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
            Equation 3
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            #TODO: generalize this algorithm for difference sizes of matrices(1D, 2D)
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
            # TODO: simplify the code and reduce the redundant calculation

            if theta_priors is None:
                #theta_priors = np.ones((len(lambdas), len(thetas))) / (len(thetas)*len(lambdas))
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


            def compute_score(traj, _lambda): #without recording past traj
                # scores = np.empty(L)
                scores = []
                for i, (s, a) in enumerate(traj):  # pp score calculation method
                    #print("--i, (s, a), lambda:", i, (s, a), _lambda)
                    p_a = action_probabilities(s, _lambda)  # get probability of action in each state given a lambda
                    # scores[i] = p_a[s, a]
                    #print("-p_a[a]:", p_a[a_i])
                    # scores[i] = p_a[a]
                    a_i = self.action_set.index(a)
                    scores.append(p_a[a_i])
                #print("scores at frame {}:".format(self.frame), scores)
                log_scores = np.log(scores)
                return np.sum(log_scores)

            "USE THIS to record scores for past traj to speed up run time!"
            def get_last_score(traj, _lambda): #add score to existing list of score
                p_a = action_probabilities(traj[-1][0], _lambda)
                a = traj[-1][1]
                a_i = self.action_set.index(a)
                if _lambda in self.past_scores: #add to existing list
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
                    #score_list.append(compute_score(traj, lamb, L))
                    score_list.append(get_last_score(traj, lamb))
                max_lambda_j = np.argmax(score_list)
                suited_lambdas[i] = lambdas[max_lambda_j]  # recording the best suited lambda for corresponding theta[i]


            p_theta = np.copy(theta_priors)
            "Re-sampling from initial distribution (shouldn't matter if p_theta = prior?)"
            p_theta = belief_resample(p_theta, epsilon == 0.05)  # resample from uniform belief
            #lengths = len(thetas) * len(lambdas) #size of p_theta = size(thetas)*size(lambdas)
            p_theta_prime = np.empty((len(lambdas), len(thetas)))

            "Compute joint probability p(lambda, theta) for each lambda and theta"
            for t, (s, a) in enumerate(traj):  # enumerate through past traj
                if t == 0:  # initially there's only one state and not past
                    for theta_t in range(len(thetas)):  # cycle through list of thetas
                        for l,_lambda in enumerate(lambdas):
                            #p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                            #a_i = self.action_set.index(a)
                            p_action_l = self.past_scores[_lambda][t]
                            #print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
                            p_theta_prime[l][theta_t] = p_action_l * p_theta[l][theta_t]
                else:  # for state action pair that is not at time zero
                    for theta_t in range(len(thetas)):  # cycle through theta at time t or K
                        #p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                        for l, _lambda in enumerate(lambdas):
                            p_action_l = self.past_scores[_lambda][t]
                            for theta_past in range(len(thetas)):  # cycle through theta probability from past time K-1
                                for l_past in range(len(lambdas)):
                                    p_theta_prime[l][theta_t] += p_action_l * p_theta[l_past][theta_past]

            "In the case p_theta is 2d array:"
            print(p_theta_prime, sum(p_theta_prime))
            p_theta_prime /= np.sum(p_theta_prime) #normalize



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

            print(p_theta_prime)
            print(np.sum(p_theta_prime))
            assert 0.9<=np.sum(p_theta_prime) <= 1.1  # check if it is properly normalized
            print("-----p_thetas at frame {0}: {1}".format(self.frame, p_theta_prime))
            #return {'predicted_intent_other': [p_theta_prime, suited_lambdas]}
            return p_theta_prime, suited_lambdas


        def marginal_prob(state, p_theta, best_lambdas, dt):
            """
            Calculates the marginal probability P(x(k+1) | D(k))
            General procedure:
            1. P(lambda, theta) is calculated first and best lambda is obtained
            2. P(x(k+1) | lambda, theta) is calculated with the best lambda from previous step
            3. Calculate P(x(k+1) | D(k)) by multiplying the results from first 2 steps together, with
               the same lambda #TODO: need confirmation!
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
            #print("WATCH for p_state", traj_probabilities(state, lamb1))
            p_state_beta1, state_list = traj_probabilities(state, lamb1, dt)
            p_state_beta2 = traj_probabilities(state, lamb2, dt)[0]
            print("p theta:",p_theta, "sum:", np.sum(p_theta), "len", len(p_theta))
            #print('p_state_beta1 at time ', self.frame, ' :', p_state_beta1)
            #print('p_state_beta2 at time ', self.frame, ' :', p_state_beta2)

            "calculate marginal"
            #p_state_D = p_state_beta1.copy() #<- this creates a list connected to original...? (nested?)
            p_state_D = []
            print(p_state_D)
            for k in range(len(state_list)): #k refers to the number of future time steps: currently max k=1
                p_state_beta1k = p_state_beta1[k] #TODO: generalize for multiple thetas!
                p_state_beta2k = p_state_beta2[k]
                p_state_D.append([])
                p_state_Dk = p_state_D[k]
                for i in range(len(p_state_beta1k)):
                    #TODO: multiply by the p_theta with the corresponding lambda????
                    temp = 0
                    for j in range(len(p_theta)):
                        #TODO: check if this is right!
                        temp += p_state_beta1k[i] * p_theta[j][0] + p_state_beta2k[i] * p_theta[j][1]
                    p_state_Dk.append(temp)
                    #print(p_state_Dk[i])
            print('p_state_D at time ', self.frame, ' :', p_state_D)
            print("state of H:", state_list) #sx, sy, vx, vy

            assert 0.99 <= np.sum(p_state_D[0]) <= 1.001  #check
            #return {'predicted_policy_other': [p_state_D, state_list]}
            return p_state_D, state_list

        "------------------------------"
        "executing the above functions!"
        "------------------------------"

        "#calling functions for baseline inference"
        joint_probability = theta_joint_update(thetas=self.thetas, theta_priors=self.theta_priors,
                                               lambdas=self.lambdas, traj=traj_h, goal=self.goal, epsilon=0.05)

        "#take a snapshot of the theta prob for next time step"
        self.theta_priors, best_lambdas = joint_probability

        "calculate the marginal state distribution / prediction"
        best_lambdas = joint_probability[1]
        marginal_state = marginal_prob(state=curr_state_h, p_theta=self.theta_priors,
                                       best_lambdas=best_lambdas, dt=1) #IMPORTANT: set dt to desired look ahead
        #TODO: logging the data for verification?
        return {'predicted_intent_other': joint_probability,
                'predicted_states_other': marginal_state} #TODO: CHECK WHAT TO RETURN


        # def lambda_update( self, lambdas, traj, priors, goals, k):
        #     #This function is not in use! But it is a good reference for update algorithm
        #     """
        #     refer to beta.py
        #     Simple lambda updates without theta joint inference
        #     Update belief over set of beta with Baysian update
        #     params:
        #     traj: for obtaining trajectory up to time step k
        #     k: current time step
        #     trajectory probabilities: calculates probability of action taken at given state and params
        #     :return: Posterior belief over betas
        #     """
        #
        #     if priors is None:
        #         priors = np.ones(len(lambdas)) #assume uniform priors
        #         priors /= len(lambdas) #normalize
        #
        #     resampled_prior = self.belief_resample(priors, epsilon = 0.05) #0.05 is what they used
        #
        #     if k is not None:
        #         traj = traj[-k:]
        #
        #     #calculating posteriors
        #     post_lambda = np.copy(priors)
        #     for i,beta in enumerate(priors):
        #         #multiply by action probability given state and params
        #         post_lambda[i] *= self.trajectory_probabilities(goals, traj=traj, beta=beta)
        #
        #     np.divide(post_lambda, np.sum(post_lambda), out=post_lambda) #normalize
        #
        #     return post_lambda
        #
        #     pass


        "functions for references, not in use:"
            # def state_probabilities_infer(self, traj, goal, state_priors, thetas, theta_priors, lambdas, T):
        #     #TODO: maybbe we dont need this function? as our transition is deterministic and have only one destination
        #     #Not in use
        #     """
        #     refer to state.py and occupancy.py
        #     Infer the state probabilities before observation of lambda and theta.
        #     Equation 4: P(x(k+1);lambda, theta) = P(u(k)|x(k);lambda,theta) * P(x(k),lambda, theta)
        #                                         = action_probabilities * p_state(t-1)
        #     params:
        #     T: time horizon for state probabilities inference
        #     :return:
        #     probability of theta
        #     probability the agent is at each state given correct theta
        #     corresponding lambda
        #     """
        #     #TODO theta_priors could be None!! Design the function around it!
        #     p_theta, lambdas = self.theta_joint_update(thetas, theta_priors, traj,goal, epsilon = 0.05)
        #     # TODO: modify the following sample code:
        #
        #
        #     def infer_simple(self, T):
        #         p_state = np.zeros([T+1, self.states])
        #         p_state[0][self.initial_state] = 1 #start with prob of 1 at starting point
        #         p_action = self.baseline_inference.action_probabilities()
        #         for t in range(1, T + 1):
        #             p = p_state[t - 1]
        #             p_prime = p_state[t]
        #             p_prime *= p_action #TODO: check this calculation! Maybe need p_action for each time step
        #         return p_state
        #
        #     # Take the weighted sum of the occupancy given each individual destination.
        #     # iterate through multiple thetas
        #     for theta, lamb, p_action, p_theta in zip(thetas, lambdas, p_actions, p_theta):
        #         p_state = infer_simple(T)
        #         np.multiply(p_state, p_theta, out=p_state)
        #         #TODO: I am confused... need to check back for how to calculate p_state in our case
        #
        #
        #     pass

        # def value_iter():
        #     #Not in use
        #     """
        #     refer to hardmax.py
        #     Calculate V where ith entry is the value of starting at state s till i
        #     :return:
        #     """
        #     pass

    def trained_baseline_inference(self, agent, sim):
        """
        Use Q function from nfsp models
        :param agent:
        :param sim:
        :return:
        """

        "importing agents information from Autonomous Vehicle (sim.agents)"
        curr_state_h = sim.agents[0].state[-1]
        last_action_h = sim.agents[0].action[-1]
        curr_state_m = sim.agents[1].state[-1]
        last_action_m = sim.agents[1].action[-1]

        self.traj_h.append([curr_state_h, last_action_h])
        self.traj_m.append([curr_state_m, last_action_m])
        self.frame = sim.frame
        if not self.frame == 0:
            self.theta_priors = self.sim.agents[1].predicted_intent_other[-1][0]

        def trained_q_function(state_h, state_m):
            """
            Import Q function from nfsp given states
            :param state_h:
            :param state_m:
            :return:
            """

            q_set = get_models()[0]
            #Q = q_set[0]  # use na_na for now

            "Q values for given state over a set of actions:"
            #Q_vals = Q.forward(torch.FloatTensor(state).to(torch.device("cpu")))
            return q_set

        def q_values(state_h, state_m, intent):
            """
            Get q values given the intent (Non-aggressive or aggressive)
            :param state_h:
            :param state_m:
            :param intent:
            :return:
            """
            q_set = trained_q_function(state_h, state_m)
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
            action_set = self.action_set
            if theta == self.thetas[0]:
                intent = "na_na"
            else:
                intent = "a_na"

            print(intent)
            q_vals = q_values(state_h, state_m, intent=intent)
            #TODO: boltzmann noisily rational model
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
            #print("EXP_Q:", exp_Q)

            "normalizing"
            # normalize(exp_Q, norm = 'l1', copy = False)
            exp_Q /= sum(exp_Q)
            print("exp_Q normalized:", exp_Q)
            return exp_Q

        def get_state_list(state, T, dt):
            #TODO: check if it works for this model
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

                def calc_state(x, u, dt):
                    sx, sy, vx, vy = x[0], x[1], x[2], x[3]
                    "Deceleration only leads to small/min velocity!"
                    if sx == 0 and vx == 0:  # y axis movement
                        print("Y axis movement detected")
                        vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vy_new < self.min_speed:
                            vy_new = self.min_speed
                        else:
                            vy_new = max(min(vy_new, self.max_speed), self.min_speed)
                        sx_new = sx  # + (vx + vx_new) * dt * 0.5
                        sy_new = sy + (vy + vy_new) * dt * 0.5
                    elif sy == 0 and vy == 0:  # x axis movement
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy  # + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vx_new > -self.min_speed:
                            print("vehicle M is exceeding min speed", vx_new, u)
                            vx_new = -self.min_speed
                        else:
                            vx_new = min(max(vx_new, -self.max_speed),
                                         -self.min_speed)  # negative vel, so min and max is flipped
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy  # + (vy + vy_new) * dt * 0.5
                    else:  # TODO: assume Y axis movement for single agent case!!
                        print("Y axis movement detected (else)")
                        vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vy_new < 0:
                            vy_new = 0
                        sx_new = sx  # + (vx + vx_new) * dt * 0.5
                        sy_new = sy + (vy + vy_new) * dt * 0.5

                    # TODO: add a cieling for how fast they can go
                    return sx_new, sy_new, vx_new, vy_new

                "Checking if _states is composed of tuples of state info (initially _state is just a tuple)"
                if not isinstance(_state_list[0], tuple):
                    print("WARNING: state list is not composed of tuples!")
                    _state_list = [_state_list]  # put it in a list to iterate

                for s in _state_list:
                    for a in _actions:
                        # print("STATE", s)
                        _s_prime.append(calc_state(s, a, dt))
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
            refer to mdp.py
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

            # TODO: verify if it is working properly (plotting states? p_state seems correct)
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
            return p_states, state_list

        def resample(priors, epsilon):
            """
            Equation 3
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            # TODO: generalize this algorithm for difference sizes of matrices(1D, 2D)
            # initial_belief = np.ones((len(priors), len(priors[0]))) / (len(priors)*len(priors[0]))
            initial_belief = self.initial_joint_prob
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def joint_prob(theta_list, lambdas, traj_h, traj_m, goal, epsilon=0.05, priors=None):
            #TODO: study how joint prob is calculated when distribution depends on intent
            intent_list = ["na_na", "a_na"]
            if priors is None:
                #theta_priors = np.ones((len(lambdas), len(thetas))) / (len(thetas)*len(lambdas))
                priors = self.initial_joint_prob
            print("-----theta priors: {}".format(priors))
            print("traj: {}".format(traj_h))
            suited_lambdas = np.empty(len(intent_list))

            # TODO: modify score list to include all thetas x lambda pairs, instead of lambdas
            "USE THIS to record scores for past traj to speed up run time!"
            def get_last_score(_traj_h, _traj_m, _lambda, _theta):  # add score to existing list of score
                p_a = action_prob(_traj_h[-1][0], _traj_m[-1][0], _lambda, _theta) #traj: [(s, a), (s2, a2), ..]
                a_h = _traj_h[-1][1]
                #print(_traj_h)
                a_i = self.action_set.index(a_h)
                if _theta == self.thetas[0]:
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
            for i, theta in enumerate(theta_list):  # get a best suited lambda for each theta
                score_list = []
                for j, lamb in enumerate(lambdas):
                    score_list.append(get_last_score(traj_h, traj_m, lamb, theta))
                max_lambda_j = np.argmax(score_list)
                suited_lambdas[i] = lambdas[max_lambda_j]  # recording the best suited lambda for corresponding theta[i]

            p_theta = np.copy(priors)
            print("prior:", p_theta)
            "Re-sampling from initial distribution (shouldn't matter if p_theta = prior?)"
            p_theta = resample(p_theta, epsilon == 0.05)  # resample from uniform belief
            print("resampled:", p_theta)
            # lengths = len(thetas) * len(lambdas) #size of p_theta = size(thetas)*size(lambdas)
            p_theta_prime = np.empty((len(lambdas), len(theta_list)))

            "Compute joint probability p(lambda, theta) for each lambda and theta"
            # for t, (s, a) in enumerate(traj_h):  # enumerate through past traj
            #     if t == 0:  # initially there's only one state and not past
            #         for theta_t, theta in enumerate(theta_list):  # cycle through list of thetas
            #             for l,_lambda in enumerate(lambdas):
            #                 #p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
            #                 #a_i = self.action_set.index(a)
            #                 if theta == theta_list[0]:
            #                     p_action_l = self.past_scores1[_lambda][t] #get prob of action done at time t
            #                     # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
            #                     p_theta_prime[l][theta_t] = p_action_l * p_theta[l][theta_t]
            #                 else: #theta 2
            #                     p_action_l = self.past_scores2[_lambda][t]
            #                     # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
            #                     p_theta_prime[l][theta_t] = p_action_l * p_theta[l][theta_t]
            #
            #     else:  # for state action pair that is not at time zero
            #         for theta_t, theta in enumerate(theta_list):  # cycle through theta at time t or K
            #             #p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
            #             for l, _lambda in enumerate(lambdas):
            #                 if theta == theta_list[0]:
            #                     p_action_l = self.past_scores1[_lambda][t]
            #                     for theta_past in range(
            #                             len(theta_list)):  # cycle through theta probability from past time K-1
            #                         for l_past in range(len(lambdas)):
            #                             p_theta_prime[l][theta_t] += p_action_l * p_theta[l_past][theta_past]
            #                 else:
            #                     p_action_l = self.past_scores2[_lambda][t]
            #                     for theta_past in range(
            #                             len(theta_list)):  # cycle through theta probability from past time K-1
            #                         for l_past in range(len(lambdas)):
            #                             p_theta_prime[l][theta_t] += p_action_l * p_theta[l_past][theta_past]

            "another joint update algorithm, without cycling through traj and only update with prior:"
            t = self.frame
            # TODO: use [-1] or [t]???
            for theta_t, theta in enumerate(theta_list):  # cycle through list of thetas
                for l, _lambda in enumerate(lambdas):
                    # p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
                    # a_i = self.action_set.index(a)
                    if theta == theta_list[0]:
                        p_action_l = self.past_scores1[_lambda][t]  # get prob of action done at time t
                        # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
                        p_theta_prime[l][theta_t] = p_action_l * p_theta[l][theta_t]
                    else:  # theta 2
                        p_action_l = self.past_scores2[_lambda][t]
                        # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
                        p_theta_prime[l][theta_t] = p_action_l * p_theta[l][theta_t]
            "this version iterates through prior separately!"
            # t = self.frame
            # if t == 0:  # initially there's only one state and not past
            #     for theta_t, theta in enumerate(theta_list):  # cycle through list of thetas
            #         for l, _lambda in enumerate(lambdas):
            #             # p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
            #             # a_i = self.action_set.index(a)
            #             if theta == theta_list[0]:
            #                 p_action_l = self.past_scores1[_lambda][t]  # get prob of action done at time t
            #                 # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
            #                 p_theta_prime[l][theta_t] = p_action_l * p_theta[l][theta_t]
            #             else:  # theta 2
            #                 p_action_l = self.past_scores2[_lambda][t]
            #                 # print("p_a:{0}, p_t:{1}".format(p_action_l, p_theta))
            #                 p_theta_prime[l][theta_t] = p_action_l * p_theta[l][theta_t]
            #
            # else:  # for state action pair that is not at time zero
            #     for theta_t, theta in enumerate(theta_list):  # cycle through theta at time t or K
            #         # p_action = action_probabilities(s, suited_lambdas[theta_t])  # 1D array
            #         for l, _lambda in enumerate(lambdas):
            #             if theta == theta_list[0]:
            #                 p_action_l = self.past_scores1[_lambda][t]
            #                 for theta_past in range(
            #                         len(theta_list)):  # cycle through theta probability from past time K-1
            #                     for l_past in range(len(lambdas)):
            #                         p_theta_prime[l][theta_t] = p_action_l * p_theta[l_past][theta_past]
            #             else:
            #                 p_action_l = self.past_scores2[_lambda][t]
            #                 for theta_past in range(
            #                         len(theta_list)):  # cycle through theta probability from past time K-1
            #                     for l_past in range(len(lambdas)):
            #                         p_theta_prime[l][theta_t] = p_action_l * p_theta[l_past][theta_past]

            "In the case p_theta is 2d array:"
            # print(p_theta_prime, sum(p_theta_prime))
            p_theta_prime /= np.sum(p_theta_prime)  # normalize

            # print(p_theta_prime)
            # print(np.sum(p_theta_prime))
            assert 0.9 <= np.sum(p_theta_prime) <= 1.1  # check if it is properly normalized
            print("-----p_thetas at frame {0}: {1}".format(self.frame, p_theta_prime))
            #return {'predicted_intent_other': [p_theta_prime, suited_lambdas]}
            return p_theta_prime, suited_lambdas

        def marginal_state(state_h, state_m, p_theta, best_lambdas, thetas, dt):
            """

            :param state_h:
            :param state_m:
            :param p_theta:
            :param best_lambdas:
            :param thetas:
            :param dt:
            :return:
            """

            "get required information"
            lamb1, lamb2 = best_lambdas
            theta1, theta2 = thetas
            # print("WATCH for p_state", traj_probabilities(state, lamb1))
            #TODO: check thetas
            p_state_beta1, state_list = traj_prob(state_h, state_m, lamb1, theta1, dt)
            p_state_beta2 = traj_prob(state_h, state_m, lamb2, theta2, dt)[0] #only probability no state list
            print("p theta:", p_theta, "sum:", np.sum(p_theta), "len", len(p_theta))
            # print('p_state_beta1 at time ', self.frame, ' :', p_state_beta1)
            # print('p_state_beta2 at time ', self.frame, ' :', p_state_beta2)

            "calculate marginal"
            # p_state_D = p_state_beta1.copy() #<- this creates a list connected to original...? (nested?)
            p_state_D = []
            for k in range(len(state_list)):  # k refers to the number of future time steps: currently max k=1
                p_state_beta1k = p_state_beta1[k]  # TODO: generalize for multiple thetas!
                p_state_beta2k = p_state_beta2[k]
                p_state_D.append([])
                p_state_Dk = p_state_D[k]
                for i in range(len(p_state_beta1k)):
                    # TODO: multiply by the p_theta with the corresponding lambda????
                    temp = 0
                    for j in range(len(p_theta)):
                        # TODO: check if this is right!
                        temp += p_state_beta1k[i] * p_theta[j][0] + p_state_beta2k[i] * p_theta[j][1]
                    p_state_Dk.append(temp)
                    # print(p_state_Dk[i])
            print('p_state_D at time ', self.frame, ' :', p_state_D)
            print("state of H:", state_h, state_list)  # sx, sy, vx, vy

            assert 0.99 <= np.sum(p_state_D[0]) <= 1.001  # check
            # return {'predicted_policy_other': [p_state_D, state_list]}
            return p_state_D, state_list

        "------------------------------"
        "executing the above functions!"
        "------------------------------"

        "#calling functions for baseline inference"
        joint_probability = joint_prob(theta_list=self.thetas, lambdas=self.lambdas,
                                       traj_h=self.traj_h, traj_m=self.traj_m, goal=self.goal, epsilon=0.05,
                                       priors=self.theta_priors)

        "#take a snapshot of the theta prob for next time step"
        self.theta_priors, best_lambdas = joint_probability

        "calculate the marginal state distribution / prediction"
        best_lambdas = joint_probability[1]
        marginal_state_prob = marginal_state(state_h=curr_state_h, state_m=curr_state_m,
                                             p_theta=self.theta_priors,
                                             best_lambdas=best_lambdas, thetas=self.thetas, dt=self.dt)
        "getting the predicted action"
        p_state_0 = marginal_state_prob[0][0]
        idx = p_state_0.index(max(p_state_0))
        predicted_action = self.action_set[idx]

        #IMPORTANT: set dt to desired look ahead
        #TODO: logging the data for verification?
        return {'predicted_intent_other': joint_probability,
                'predicted_states_other': marginal_state_prob,
                'predicted_actions_other': predicted_action}


    #@staticmethod
    def empathetic_inference(self, agent, sim):
        """
        When QH also depends on xM,uM
        :return:P(beta_h, beta_m_hat | D(k))
        """
        #TODO: documentation
        """
        Equation 6
        Equation 7
        Equation 8
        Equation 9
        Equation 10
        Equation 11
        """
        #----------------------------#
        # TODO: implement proposed:
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        #----------------------------#

        #NOTE: action prob is considering only one Nash Equilibrium (Qh, Qm) instead of a set of them!!!
        "importing agents information from Autonomous Vehicle (sim.agents)"
        curr_state_h = sim.agents[0].state[-1]
        last_action_h = sim.agents[0].action[-1]
        curr_state_m = sim.agents[1].state[-1]
        last_action_m = sim.agents[1].action[-1]

        self.traj_h.append([curr_state_h, last_action_h])
        self.traj_m.append([curr_state_m, last_action_m])
        self.frame = sim.frame
        # if not self.frame == 0:
        #     self.theta_priors = self.sim.agents[1].predicted_intent_other[-1][0]

        "place holder: using NFSP Q function in place of NE Q function pair"
        def trained_q_function(state_h, state_m):
            """
            Import Q function from nfsp given states
            :param state_h:
            :param state_m:
            :return:
            """
            # Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2
            q_set = get_models()[0] #0: q func, 1: policy
            # Q = q_set[0]  # use na_na for now
            #TODO: maybe store it as a dictionary?
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
            #TODO: implement when there are 2 Q pairs for na_na and a_na
            #TODO: generalize to iterate
            "thetas: na, a"
            id_h = self.thetas.index(theta_h)
            id_m = self.thetas.index(theta_m)
            if id_h == 0:
                if id_m == 0: #TODO: IMPORTANT: CHECK WHICH ONE IS NA2 IN DECISION
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
                elif id_m == 1:  #TODO: IMPORTANT: CHECK WHICH ONE IS A2 IN DECISION
                    Q_h = q_set[4]
                    Q_m = q_set[5]
                else:
                    print("ID FOR THETA DOES NOT EXIST")
            # if intent == "na_na":
            #     Q_h = q_set[0]
            #     Q_m = q_set[1]
            # else:  # use a_na
            #     Q_h = q_set[3]
            #     Q_m = q_set[2]

            "Need state for agent H: xH, vH, xM, vM"
            # state_h = [state_h[0], state_h[2], state_m[1], state_m[3]]
            # state_m = [state_m[1], state_m[3], state_h[0], state_h[2]]
            state_h = [-state_h[1], abs(state_h[3]), state_m[0], abs(state_m[2])]
            state_m = [state_m[0], abs(state_m[2]), -state_h[1], abs(state_h[3])]

            "Q values for each action"
            Q_vals_h = Q_h.forward(torch.FloatTensor(state_h).to(torch.device("cpu")))
            Q_vals_m = Q_m.forward(torch.FloatTensor(state_m).to(torch.device("cpu")))
            "detaching tensor"
            Q_vals_h = Q_vals_h.detach().numpy()
            Q_vals_m = Q_vals_m.detach().numpy()
            Q_vals_h = Q_vals_h.tolist()
            Q_vals_m = Q_vals_m.tolist()
            # TODO: add multiple Q pairs under certain beta pair (ie. Q_na_na)
            return [Q_vals_h, Q_vals_m]

        def action_prob(state_h, state_m, beta_h, beta_m):
            """
            calculate action prob for both agents
            :param state_h:
            :param state_m:
            :param _lambda:
            :param theta:
            :return:
            """

            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m
            action_set = self.action_set
            # if theta == self.thetas[0]:
            #     intent = "na_na"
            # else:
            #     intent = "a_na"

            #q_vals = q_values(state_h, state_m, intent=intent)

            q_vals_pair = q_values_pair(state_h, state_m, theta_h, theta_m)

            "Q*lambda"
            # np.multiply(Q,_lambda,out = Q)
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
                print("exp_Q normalized:", exp_Q)
                exp_Q_pair.append(exp_Q)

            return exp_Q_pair  # [exp_Q_h, exp_Q_m]

        def action_prob_Q(state_h, state_m, Q_h, Q_m, beta_h, beta_m):
            """
            calculate action prob for both agents given Q_h and Q_m
            :param state_h:
            :param state_m:
            :param _lambda:
            :param theta:
            :return:
            """
            action_set = self.action_set

            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m

            "Q*lambda"
            # np.multiply(Q,_lambda,out = Q)
            # q_vals_h = q_vals_h.detach().numpy()  # detaching tensor
            # q_vals_m = q_vals_m.detach().numpy()
            q_vals_pair = [Q_h, Q_m]
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
                    exp_Q.append(np.exp(q))
                # print("EXP_Q:", exp_Q)

                "normalizing"
                # normalize(exp_Q, norm = 'l1', copy = False)
                exp_Q /= sum(exp_Q)
                print("exp_Q normalized:", exp_Q)
                exp_Q_pair.append(exp_Q)

            return exp_Q_pair  # [exp_Q_h, exp_Q_m]

        def resample(priors, epsilon):
            """
            Equation 3
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """

            if isinstance(priors[0], list):
                initial_belief = np.ones((len(priors), len(priors[0]))) / (len(priors)*len(priors[0]))
            elif type(priors[0]) is np.array:
                initial_belief = np.ones((len(priors), len(priors[0]))) / (len(priors) * len(priors[0]))
            else:  # 1D array
                initial_belief = np.ones(len(priors))/len(priors)
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def prob_q_vals(prior, state_h, state_m, traj_h, traj_m, beta_h, beta_m):
            #TODO: documentation
            """
            Equation 6
            Calculates Q function pairs probabilities for use in beta pair probabilities calculation,
            since each beta pair may map to MULTIPLE Q function/value pair.

            :requires: p_action_h, p_pair_prob(k-1) or prior, q_pairs
            :param:
            q_pairs: all Q function pairs (QH, QM)
            p_action_h
            p_q2

            :return:
            """
            # if theta_h == self.thetas[0]:
            #     intent = 'na_na'
            # else:
            #     intent = 'a_na'
            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m

            # get all q pairs
            q_pairs = []
            "q_pairs (QH, QM): [[Q_na_na, Q_na_na2], [Q_na_a, Q_a_na], [Q_a_na, Q_na_a], [Q_a_a, Q_a_a2]]"
            for t_m in self.thetas:
                for t_h in self.thetas:
                    q_pairs.append(q_values_pair(state_h, state_m, t_h, t_m))  # [q_vals_h, q_vals_m]
            #q_pairs = [q_pairs]

            # Size of P(Q2|D) should be the size of possible Q2
            if prior is None:
                prior = np.ones(len(q_pairs))/len(q_pairs)
            else:
                "resample from initial/uniform distribution"
                prior = resample(prior, epsilon=0.05)

            p_q2 = np.empty(len(q_pairs))

            # assuming 1D array of q functions
            "Calculating probability of each Q pair: Equation 6"
            action_h = traj_h[-1][1]
            action_m = traj_m[-1][1]
            ah = self.action_set.index(action_h)
            am = self.action_set.index(action_m)
            for i, q in enumerate(q_pairs):  # iterate through pairs of equilibriums
                #  TODO: check if it's correct to use predicted beta
                p_action = action_prob_Q(state_h, state_m, q[0], q[1], beta_h, beta_m)
                p_action_h = p_action[0][ah]
                p_action_m = p_action[1][am]
                p_a_pair = p_action_h * p_action_m
                "P(Q2|D(k)) = P(uH, uM|x(k), QH, QM) * P(Q2|D(k-1)) / sum(~)"
                p_q2[i] = p_a_pair * prior[i]

            p_q2 /= sum(p_q2) #normalize
            assert 0.99 <= sum(p_q2) <= 1.01 #check if properly normalized

            return q_pairs, p_q2

        def prob_q2_beta(index, q_id):
            """
            Get probability of pair QH, QM given betas
            :param index: 2 entries: (i, j), where i is the index for beta_h, j is the index for beta_m
            i(0:7) = theta1 = na, i(8:15) = theta2 = a
            j(0:7) = theta1 = na, j(8:15) = theta2 = a
            :return:
            """

            "get intent of q_pair from q_pairs"
            "q_pairs (QH, QM): [[Q_na_na, Q_na_na2], [Q_na_a, Q_a_na], [Q_a_na, Q_na_a], [Q_a_a, Q_a_a2]]"
            # print(q_pairs)
            # print(q_pair)
            #q_id = q_pairs.index(q_pair)  # 0, 1, 2, 3
            print("Q id:", q_id)
            #TODO: for 1 and 3 make the probability 0.5
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
            for i in index:  # (i, j) = (row, col). 8x8 2D array
                if i < 4:
                    id.append(0)
                else:
                    id.append(1)

            if id[0] == th and id[1] == tm:
                return 1
            else:
                return 0

        def prob_beta_given_q(p_betas_prior, q_id):
            #TODO: modify this so that it takes a certain pair of Qs and calculat P(B2) based on the Q2
            """
            Equation 8: using Bayes rule
            Calculates probability of beta pair (Bh, BM_hat) given Q pair (QH, QM): P(Bh, BM_hat | QH, QM),
            for beta_pair_prob formula.
            --> GIVEN A PARTICULAR Q PAIR
            :param self:
            :return:
            """

            "import prob of beta pair given D(k-1) from Equation 7: P(betas|D(k-1))"
            if p_betas_prior is None:
                betas_len = len(self.betas)
                p_betas_prior = np.ones((betas_len, betas_len)) / (betas_len * betas_len)
            else:
                p_betas_prior = resample(p_betas_prior, epsilon=0.05)
            "prob of Q pair given beta: equally distributed probabilities P(Qm, Qh | betas)"

            "calculate prob of beta pair given Q pair"
            # this is assuming 1D arrays
            p_beta_q2 = np.empty((len(p_betas_prior), len(p_betas_prior[0])))
            for i in range(len(p_betas_prior)):
                for j in range(len(p_betas_prior[i])):
                    'getting P(Q2|betas), given beta id (i, j)'
                    p_q2_beta = prob_q2_beta((i, j), q_id)
                    p_beta_q2[i][j] = p_q2_beta * p_betas_prior[i][j]
                    # for k in range(len(p_q2_beta)):
                    #     if k == 0:
                    #         p_beta_q2[i][j] = p_betas_prior[i][j] * p_q2_beta[k]
                    #     else:
                    #         p_beta_q2[i][j] += p_betas_prior[i][j] * p_q2_beta[k]
            p_beta_q2 /= np.sum(p_beta_q2)
            # TODO: do they sum up to 1?
            #assert 0.99 <= np.sum(p_beta_q2) <= 1.01  # check if properly normalized

            return p_beta_q2

        def prob_beta_pair(p_q2, q_pairs):
            """
            Equation 7
            Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
            :param self:
            :return:
            """
            #TODO: USE THE P_Q2 WITH THE Q2 CORRESPONDING TO THE P_BETA_Q2???

            "Calculate prob of beta pair given D(k) by summing over Q pair"

            # TODO: resample from initial belief! HOW??? (no prior is used!)
            "resample from initial/uniform distribution"
            p_betas_d = np.zeros((len(self.betas), len(self.betas)))

            #intent = (1, 1) #assume na_na & theta_H = theta_M = theta_set[0]
            # 2D version
            # for i in range(len(p_betas_q2)):
            #     for j in range(len(p_betas_q2[0])):
            #         for k in range(len(p_q2)):
            #             #TODO: give the function a particular q_pair!
            #             p_betas_q2 = prob_beta_given_q(beta_H=beta_h, beta_M=beta_m,
            #                                            p_betas_prior=self.p_betas_prior,
            #                                            q_pairs=p_q2, q_pair=q_pairs[k])
            #             #print(p_betas_q2, p_q2)
            #             #print(p_betas_d)
            #             if k == 0:
            #                 p_betas_d[i][j] = p_betas_q2[i][j] * p_q2[k]
            #             else:
            #                 p_betas_d[i][j] += p_betas_q2[i][j] * p_q2[k]
            for i, q2 in enumerate(q_pairs):
                p_betas_q2 = prob_beta_given_q(p_betas_prior=self.p_betas_prior,
                                               q_id=i)
                for j in range(len(p_betas_q2)):
                    for k in range(len(p_betas_q2[j])):
                        p_betas_d[j][k] += p_betas_q2[j][k] * p_q2[i]

            return p_betas_d #TODO: does it need to be normalized?

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
            # TODO: check if this works for both agents!
            #  ---------------------------------------
            def get_s_prime(_state_list, _actions):
                _s_prime = []

                def calc_state(x, u, dt):  #TODO: Implement dynamics in the outer scope!
                    sx, sy, vx, vy = x[0], x[1], x[2], x[3]
                    "Deceleration only leads to small/min velocity!"
                    if sx == 0 and vx == 0:  # y axis movement
                        #print("Y axis movement detected")
                        vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vy_new < self.min_speed:
                            vy_new = self.min_speed
                        else:
                            vy_new = max(min(vy_new, self.max_speed), self.min_speed)
                        sx_new = sx  # + (vx + vx_new) * dt * 0.5
                        sy_new = sy + (vy + vy_new) * dt * 0.5
                    elif sy == 0 and vy == 0:  # x axis movement
                        vx_new = vx + u * dt  # * vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy  # + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vx_new > -self.min_speed:
                            #print("vehicle M is exceeding min speed", vx_new, u)
                            vx_new = -self.min_speed
                        else:
                            vx_new = min(max(vx_new, -self.max_speed),
                                         -self.min_speed)  # negative vel, so min and max is flipped
                        sx_new = sx + (vx + vx_new) * dt * 0.5
                        sy_new = sy  # + (vy + vy_new) * dt * 0.5
                    else:  # TODO: in the case of more than 1D movement??
                        print("Y axis movement detected (else)")
                        vx_new = vx  # + u * dt #* vx / (np.linalg.norm([vx, vy]) + 1e-12)
                        vy_new = vy + u * dt  # * vy / (np.linalg.norm([vx, vy]) + 1e-12)
                        if vy_new < 0:
                            vy_new = 0
                        sx_new = sx  # + (vx + vx_new) * dt * 0.5
                        sy_new = sy + (vy + vy_new) * dt * 0.5

                    # TODO: add a cieling for how fast they can go
                    return sx_new, sy_new, vx_new, vy_new

                "Checking if _states is composed of tuples of state info (initially _state is just a tuple)"
                if not isinstance(_state_list[0], tuple):
                    print("WARNING: state list is not composed of tuples!")
                    _state_list = [_state_list]  # put it in a list to iterate

                for s in _state_list:
                    for a in _actions:
                        # print("STATE", s)
                        _s_prime.append(calc_state(s, a, dt))
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

        def joint_action_prob(state_h, state_m, beta_h, beta_m):
            """
            Multiplying the two action prob together for both agents
            :param state_h:
            :param state_m:
            :param lambdas:
            :param thetas:
            :return:
            """
            # p_state_joint = []
            # for i in range(len(p_state_h)):
            #     p_state_h.append([])
            #     for j in range(len(p_state_h[i])):
            #         for p in range(len(p_state_m)):
            #             for q in range(len(p_state_m)):
            #                 p_state_u = p_state_h[i][j] * p_state_m[p][q]
            #                 p_state_joint[i].append()
            "in the case where action prob is calculated separately"
            # lambda_h, lambda_m = lambdas
            # theta_h, theta_m = thetas
            # p_action_h = action_prob(state_h, state_m, lambda_h, theta_h)
            # p_action_m = action_prob(state_m, state_h, lambda_m, theta_m)

            "in the case where action prob is calculated TOGETHER"
            p_action_h, p_action_m = action_prob(state_h, state_m, beta_h, beta_m)

            #p_action_joint = np.empty((len(p_action_h), len(p_action_m)))
            p_action_joint = []
            for p_a_h in p_action_h:
                for p_a_m in p_action_m:
                    #p_action_joint[i][j] = p_action_h[i] * p_action_m[j]
                    p_action_joint.append(p_a_h * p_a_m)
            assert 0.98 < sum(p_action_joint) < 1.02

            return p_action_joint

        def traj_prob(state_h, state_m, beta_h, beta_m, dt, prior=None):
            """
            refer to mdp.py
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
            state_list_h = get_state_list(state_h, T, dt)  # get list of state given curr_state/init_state from self._init_
            state_list_m = get_state_list(state_m, T, dt)
            state_list = []
            "joining H and M's states together"
            for i in range(len(state_list_h)):
                state_list.append([])
                for j in range(len(state_list_h[i])):
                    for p in range(len(state_list_m)):
                        for q in range(len(state_list_m[p])):
                            state_list[i].append([state_list_h[i][j], state_list_m[p][q]])

            # p_states = np.zeros(shape=state_list)
            p_states = []

            "for the case where state list is 1D, note that len(state list) == number of time steps!"
            for i in range(len(state_list_h)):  # over state list at time i/ t
                #p_a = action_prob(state_h, state_m, _lambda, theta)  # assume same action prob at all time
                p_a = joint_action_prob(state_h, state_m, beta_h, beta_m)
                if i == 0:
                    p_states.append(p_a)  # 1 step look ahead only depends on action prob
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
                    p_states.append(p_s_t)

            #TODO: verify that state prob corresponds to the right state
            return p_states, state_list

        def marginal_state_prob(p_traj, p_q2, which_q2):
            """

            :param p_traj:
            :param p_q2:
            :param which_q2:
            :return:
            """
            #marginal = np.empty(len(p_traj))
            p_q2 = p_q2[which_q2] #TODO: assuming p_traj is only using this particular q2???
            marginal = []
            for i in range(len(p_traj)):
                marginal.append([])
                for j in range(len(p_traj[i])):
                    # for p in range(len(p_q2)):  # number of q pairs
                    #     #marginal[i].append(p_traj[i][j]*p_q2[p])
                    print(p_traj[i][j])
                    pm = p_traj[i][j] * p_q2
                    marginal[i].append(pm)

            return marginal

        # function to rearrange for 2D p(theta, lambda|D(k))
        def marginal_joint_intent(id, _p_beta_d):
            marginal = []
            for t in range(len(self.thetas)):
                marginal.append([])
            if id == 0:  # H agent
                for i, row in enumerate(_p_beta_d):  # get sum of row
                    if i < 4:  # in 1D self.beta, 0~3 are NA, or theta1
                        marginal[0].append(sum(row))
                    else:
                        marginal[1].append(sum(row))
            else:
                for i, col in enumerate(zip(*_p_beta_d)):
                    if i < 4:
                        marginal[0].append(sum(col))
                    else:
                        marginal[1].append(sum(col))
            # TODO: get most likely lambda???
            return marginal

        "---------------------------------------------------"
        "calling functions: P(Q2|D), P(beta2|D), P(x(k+1)|D)"
        "---------------------------------------------------"

        if self.frame == 0:  # initially guess the beta
            theta_h = self.thetas[0]
            lambda_h = self.lambdas[0]
            theta_m = self.thetas[0]
            lambda_m = self.lambdas[0]
            beta_h, beta_m = [theta_h, lambda_h], [theta_m, lambda_m]
            intent_pair = 'na_na'
            # TODO: beta M is assumed to be the equilibrium set
        else:  # get previously predicted beta
            beta_h, beta_m = self.past_beta[-1]
            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m
            if theta_h == self.thetas[0]:
                intent_pair = 'na_na'
            else:
                intent_pair = 'a_na'

        'intent and rationality inference'
        # TODO: check if this is right to use predicted betas
        #q2 = q_values_pair(curr_state_h, curr_state_m, theta_h, theta_m)
        q_pairs, p_q2_d = prob_q_vals(self.q2_prior, curr_state_h, curr_state_m,
                                      traj_h=self.traj_h, traj_m=self.traj_m, beta_h=beta_h, beta_m=beta_m)
        #print("Q pairs at time {0}:".format(self.frame), q_pairs, len(q_pairs) )
        # p_beta_q = prob_beta_given_q(beta_h, beta_m,
        #                              p_betas_prior=self.p_betas_prior, q_pairs=[q2], q_pair=q2)
        p_beta_d = prob_beta_pair(p_q2=p_q2_d, q_pairs=q_pairs)  # this calls p_beta_q internally

        'future state prediction'
        p_traj, state_list = traj_prob(curr_state_h, curr_state_m, beta_h, beta_m, dt=self.dt)
        p_q2 = np.array(p_q2_d).tolist()
        which_q2 = p_q2.index(max(p_q2))
        print('p_traj:', p_traj)
        marginal_state = marginal_state_prob(p_traj=p_traj, p_q2=p_q2_d, which_q2=which_q2)

        'recording prior'
        self.q2_prior = p_q2_d
        self.p_betas_prior = p_beta_d
        'getting best predicted betas'
        beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
        print("best betas at time {0}".format(self.frame), beta_pair_id)
        #beta_id = p_beta_d.argmax()
        best_beta_h = self.betas[beta_pair_id[0]]
        best_beta_m = self.betas[beta_pair_id[1]]
        self.past_beta.append([best_beta_h, best_beta_m])

        p_actions = action_prob(curr_state_h, curr_state_m, best_beta_h, best_beta_h)

        predicted_actions = []
        for p_a in p_actions:
            p_a = np.array(p_a).tolist()
            id = p_a.index(max(p_a))
            predicted_actions.append(self.action_set[id])
        "getting marginal prob for beta_h or beta_m:"
        p_beta_d_h = marginal_joint_intent(id=0, _p_beta_d=p_beta_d)
        p_beta_d_m = marginal_joint_intent(id=1, _p_beta_d=p_beta_d)
        # TODO: implement proposed:
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde

        # return p_theta_prime, suited_lambdas <- predicted_intent other
        # p_betas: [8 x 8] = [BH x BM]

        return {'predicted_states_other': (marginal_state, state_list),
                'predicted_actions_other': predicted_actions[0],
                'predicted_intent_other': [p_beta_d_h, beta_h],
                'predicted_intent_self': [p_beta_d_m, beta_m]}  # TODO: do we need to process p_beta_d?
        #TODO: modify so that this works for both agents??

        # return {'predicted_intent_other': joint_probability,
        #         'predicted_states_other': marginal_state,
        #         'predicted_actions_other': predicted_action}

        "---------------------------"
        "end of placeholder for NFSP"
        "---------------------------"

        # def h_action_prob(self, state, _lambda_h, q_values_h):
        #     # TODO: documentation
        #     """
        #     Calculate H's action probability based on the Q values from Nash Equilibrium (QH,QM)eq
        #     :param self:
        #     :return: H's action probabilities correspond to each action available at state s
        #     """
        #     # TODO: consider how q value from nash equilibrium is imported
        #     #Pseudo code, need confirmation:
        #     q = self.q_h_eq #get q values for H from nash equilibrium
        #
        #     #reusing q variable to save storage
        #     "p_action_h = exp(q*lambda)/sum(exp(q*lambda))"
        #     np.multiply(q, _lambda_h, out = q)
        #     np.exp(q, out = q)
        #     normalize(q, norm = 'l1', copy = False)
        #
        #     # TODO: CONSIDER if there are more than 1 set of Qs: maybe just implement them in outer scope
        #     pass
        #     return q
        #
        # def m_action_prob(self, state, _lambda_m, q_values_m):
        #     # TODO: documentation
        #     """
        #     Calculate M's action probability based on the Q values obtained from Nash Equilibrium (QH,QM)eq
        #     :param self:
        #     :return: M's action probabilities correspond to each action available at state s
        #     """
        #     # Pseudo code, need confirmation:
        #     #TODO: consider how q value from nash equilibrium is imported
        #
        #     q = self.q_m_eq  # get q values for H from nash equilibrium
        #     # reusing q variable to save storage
        #     "p_action_h = exp(q*lambda)/sum(exp(q*lambda))"
        #     np.multiply(q, _lambda_m, out=q)
        #     np.exp(q, out=q)
        #     normalize(q, norm='l1', copy=False)
        #
        #     pass
        #     return q
        #
        # def belief_resample(self, priors, epsilon):
        #     """
        #     Equation 3
        #     Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
        #     :return: resampled belief P(k-1) on lambda and theta
        #     """
        #     initial_belief = np.ones(len(priors)) / len(priors)
        #     resampled_priors = (1 - epsilon)*priors + epsilon * initial_belief
        #     return resampled_priors
        #
        # def q_pair_prob(self, prior ,q_pairs, state, lambda_h):
        #     #TODO: documentation
        #     """
        #     Equation 6
        #     Calculates Q function pairs probabilities for use in beta pair probabilities calculation,
        #     since each beta pair may map to multiple Q function/value pair.
        #
        #     :requires: p_action_h, p_pair_prob(k-1) or prior, q_pairs
        #     :param:
        #     self:
        #     q_pairs: all Q function pairs (QH, QM)
        #     p_action_h
        #     p_q2
        #
        #     :return:
        #     """
        #     #TODO: code is still in work
        #     if prior is None:
        #         prior = np.ones(q_pairs)/len(q_pairs) #TODO:: assuming uniform prior?
        #
        #     "resample from initial/uniform distribution"
        #     prior = self.belief_resample(prior, epsilon=0.05)
        #
        #     p_action_h = h_action_prob(state, lambda_h)
        #     p_q2 = np.empty(q_pairs)
        #
        #     #TODO:: I'm assuming a 2D array of q pairs is imported, needs confirmation!!!
        #     #TODO: also I need to confirm if this calculation is correct
        #     "Calculating probability of each Q pair: Equation 6"
        #     for i,q_h in enumerate(q_pairs): #rows
        #         for j,q_m in enumerate(q_h): #cols
        #             p_q2[i, j] = p_action_h[i] *prior[i, j] #action prob corresponds to H's Q so it's "i"
        #
        #     p_q2 /= sum(p_q2) #normalize
        #     assert sum(p_q2) == 1 #check if properly normalized
        #     pass
        #     return p_q2
        #
        # def prob_beta_given_q(beta_D_prior, beta_H, beta_M):
        #     #TODO: this is a placeholder, needs to be implemented
        #     """
        #     Equation 8: using Bayes rule
        #     Calculates probability of beta pair (Bh, BM_hat) given Q pair (QH, QM): P(Bh, BM_hat | QH, QM),
        #     for beta_pair_prob formula.
        #     :param self:
        #     :return:
        #     """
        #     # TODO: code is still in work
        #     #TODO: check betas
        #     if beta_D_prior is None:
        #         prior = np.ones([beta_H, beta_M]) / len([beta_H, beta_M])  # TODO:: assuming uniform prior?
        #     "import prob of beta pair given D(k-1) from Equation 7"
        #     p_betas_prev = beta_pair_prob()
        #     "calculate prob of beta pair given Q pair"
        #
        #     pass
        #
        # def beta_pair_prob(self, beta_H, beta_M, q_pairs):
        #     """
        #     Equation 7
        #     Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
        #     :param self:
        #     :return:
        #     """
        #     #TODO: This code is still in work
        #     "importing prob of Q pair given observation D(k)"
        #     p_q2 = q_pair_prob() #p_q2_d
        #     "importing prob of beta pair given Q pair"
        #     #TODO: this is a placeholder!
        #     p_betas_q = prob_beta_given_q() #does not return anything yet
        #     "Calculate prob of beta pair given D(k) by summing over Q pair"
        #
        #     # TODO: resample from initial belief! HOW??? (no prior is used!)
        #     "resample from initial/uniform distribution"
        #
        #     p_betas_d = np.zeros(p_betas_q)
        #     r = len(q_pairs)
        #     c = len(q_pairs[0])#for iterating through 2D array p_q_pairs
        #     for p in range(len(p_betas_d)): #iterating through 2D array p_betas
        #         for q in range(len(p_betas_d)[0]):
        #             p_beta_q = p_betas_q[p, q]
        #             for i in range(r): #iterate through 2D array p_q2 or p_q2_d
        #                 for j in range(c):
        #                     #if p_betas[p,q] == p_beta:
        #                     if (i, j) == (0,0): #first element
        #                         p_betas_d[p, q] = p_beta_q * p_q2[0, 0]
        #                     else:
        #                         p_betas_d[p, q] += p_beta_q * p_q2[i,j]
        #     pass
        #     return p_betas_d
        #
        # def get_state_list_h(self, T):
        #     """
        #     calculate an array of state (T x S at depth T)
        #     shape: t0[x(k)  ]
        #            t1[x(k+1)], for 1 step look ahead
        #     :param self:
        #     :return:
        #     """
        #     # TODO: Check this modification so that action probability is calculated for states within a time horizon
        #     # ---Code: append all states reachable within time T----
        #     states = self.states  # TODO: import a state to start from
        #     actions = self.action_set
        #     #T = self.T  # this should be the time horizon/look ahead: not using predefined T to generalize for usage
        #     dt = self.sim.dt
        #
        #     # TODO: MODIFY for calculating state list for agent M
        #     def get_s_prime(_states, _actions):
        #         _s_prime = []
        #
        #         def calc_state(x, u, dt):
        #             sx, sy, vx, vy = x[0], x[1], x[2], x[3]
        #             vx_new = vx + u * dt * vx / (np.linalg.norm([vx, vy]) + 1e-12)
        #             vy_new = vy + u * dt * vy / (np.linalg.norm([vx, vy]) + 1e-12)
        #             sx_new = sx + (vx + vx_new) * dt * 0.5
        #             sy_new = sy + (vy + vy_new) * dt * 0.5
        #
        #             return sx_new, sy_new, vx_new, vy_new
        #
        #         for s in _states:
        #             for a in _actions:
        #                 _s_prime.append(calc_state(s, a, dt))
        #         return _s_prime
        #
        #     # TODO: Check the state!
        #     states = self.initial_state  # or use current state
        #     i = 0  # row counter
        #     state_list = np.zeros([T, actions * T])
        #     for t in range(0, T):
        #         s_prime = get_s_prime(states, actions)  # separate pos and speed!
        #         state_list[i] = s_prime
        #         state = s_prime  # get s prime for the new states
        #         i += 1  # move onto next row
        #     return state_list
        #     # -------end of code--------
        # def get_state_list_m(self, T):
        #     """
        #     calculate an array of state (T x S at depth T)
        #     shape: t0[x(k)  ]
        #            t1[x(k+1)], for 1 step look ahead
        #     :param self:
        #     :return:
        #     """
        #     # TODO: Check this modification so that action probability is calculated for states within a time horizon
        #     # ---Code: append all states reachable within time T----
        #     states = self.states  # TODO: import a state to start from
        #     actions = self.action_set
        #     #T = self.T  # this should be the time horizon/look ahead: not using predefined T to generalize for usage
        #     dt = self.sim.dt
        #     #TODO: MODIFY for calculating state list for agent M
        #     def get_s_prime(_states, _actions):
        #         _s_prime = []
        #
        #         def calc_state(x, u, dt):
        #             sx, sy, vx, vy = x[0], x[1], x[2], x[3]
        #             vx_new = vx + u * dt * vx / (np.linalg.norm([vx, vy]) + 1e-12)
        #             vy_new = vy + u * dt * vy / (np.linalg.norm([vx, vy]) + 1e-12)
        #             sx_new = sx + (vx + vx_new) * dt * 0.5
        #             sy_new = sy + (vy + vy_new) * dt * 0.5
        #
        #             return sx_new, sy_new, vx_new, vy_new
        #
        #         for s in _states:
        #             for a in _actions:
        #                 _s_prime.append(calc_state(s, a, dt))
        #         return _s_prime
        #
        #     #TODO: Check the state! As well as the state info for M (state[:,1]?? or simply self.h_state?)!
        #     states = self.initial_state  # or use current state
        #     i = 0  # row counter
        #     state_list = np.zeros([T, actions * T])
        #     for t in range(0, T):
        #         s_prime = get_s_prime(states, actions)  # separate pos and speed!
        #         state_list[i] = s_prime
        #         state = s_prime  # get s prime for the new states
        #         i += 1  # move onto next row
        #     return state_list
        #     # -------end of code--------
        #
        # def get_state_list(self):
        #     """
        #
        #     :return:
        #     """
        #     h_states = self.get_state_list_h(self.T)
        #     h_states = h_states[1, :] #extract states from t = k+1, which is in the 2nd row
        #     m_states = self.get_state_list_m(self.T)
        #     m_states = m_states[1, :] #extract states from t = k+1, which is in the 2nd row
        #     states = np.empty([len(h_states), len(m_states)])
        #     for i in range(len(h_states)):
        #         for j in range(len(m_states)):
        #             states[i, j] = (h_states[i], m_states[j]) #TODO: Check states data type in sim
        #     return states #h_state by m_state matrix
        #
        # def state_prob(self, curr_state,Qh,Qm):
        #     """
        #     Equation 11 (includes 9 and 10)
        #     Calculates state probabilities given past observation D(k)
        #     :param self:
        #     :return:
        #     """
        #     state_list = get_state_list(self.T) #CHECK Universal Time Horizon!!!!
        #     def p_action_pair(state, _lambda):
        #         """
        #         This implementation only considers current given state! Implement outside for iterating through states
        #         Equation 10: p(uh,um|x(k);Qh,Qm)
        #         Computes joint probabilities of H and M's action prob
        #         :return:
        #         """
        #         #TODO: consider if the state contains 2 agents information!
        #         p_action_h = self.h_action_prob(state, _lambda,q_values_h=q_value_h) #1D arrays
        #         p_action_m = self.m_action_prob(state, _lambda,q_values_m=q_value_m)
        #         p_a2 = np.zeros([p_action_h,p_action_m])
        #         for ah in p_action_h:
        #             for am in p_action_m:
        #                 p_a2[ah,am] = p_action_h[ah]*p_action_m[am]
        #         return p_a2
        #         #pass
        #     def state_prob_q(prior = None):
        #         """
        #         Equation 9
        #         :return:
        #         """
        #         "import p(uh,um|x(k);Qh,Qm)"
        #         p_a2 = p_action_pair()
        #         "import prior p(x(k)|Qh,Qm)"
        #         #TODO: do we need to check if prior is available?
        #         #p_state_prev = state_prob_q() #previous time step
        #         "In the case where transition is deterministic, and prior P(x(k)) is 1 as it is already observed:"
        #         "And time horizon T = 1"
        #         return p_a2 #P(x(k+1)|Q2) is solely determined by p(uh,um|x(k),Q2)
        #         #pass
        #     #p_actions_list =
        #     "Import Q pair prob from Equation 6"
        #     p_q2 = q_pair_prob()
        #     "Call the function: state prob q"
        #     p_state_q = state_prob_q()
        #     "Equation 11: calculate state probabilities given past observation D(k)"
        #     #p_s = np.multiply(p_q2,p_state_q) <- incorrect, we are calculating the marginal
        #
        #     #TODO: check the size of state_list and p_q2
        #     p_s = np.zeros(state_list) #p_s should have the same size as s_list
        #     for p in range(len(state_list)): #assuming state_list is 2D
        #         for q in range(len(state_list[0])):
        #             p_state_q_i = p_state_q[p, q] #extract an element for multiplication
        #             for i in range(len(p_q2)): #assuming p_q2 is a 2D array
        #                 for j in range(len(p_q2[0])):
        #                     if (i, j) == (0,0): #for the first element
        #                         p_s[p, q] = p_state_q_i * p_q2[0,0]
        #                     else:
        #                         p_s[p, q] += p_state_q_i * p_q2[i, j]
        #
        #
        #     return p_s #size of uH x uM
        #


    @staticmethod
    def less_inference():
        # implement Bobu et al. "LESS is More:
        # Rethinking Probabilistic Models of Human Behavior"
        pass

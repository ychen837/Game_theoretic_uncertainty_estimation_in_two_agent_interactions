import os
from typing import List
import pygame as pg
import datetime
import pickle
import torch as t
import numpy as np
import csv
from inference_model import InferenceModel
from decision_model import DecisionModel
from autonomous_vehicle import AutonomousVehicle
from sim_draw import VisUtils
from models import constants as C  # for terminal state check (car length)
import pdb


class Simulation:

    def __init__(self, env, duration, n_agents, inference_type, decision_type, sim_dt, sim_lr, sim_par, sim_nepochs, belief_weight):

        self.duration = duration
        self.n_agents = n_agents
        self.dt = sim_dt
        self.running = True
        self.paused = False
        self.end = False
        self.clock = pg.time.Clock()
        self.frame = 0
        self.time = 0
        self.time_stamp = []
        self.decision_type = decision_type
        self.decision_type_h = decision_type[0]
        self.decision_type_m = decision_type[1]
        self.inference_type = inference_type
        self.env = env
        self.agents = []
        self.sharing_belief = True  # True for empathetic inferences
        self.theta_priors = None  # For test_baseline and baseline inference
        self.drawing_prob = True  # if function for displaying future states are enabled

        if self.inference_type[1] == 'none':
            self.drawing_intent = False
        else:
            self.drawing_intent = True

        if env.name == 'bvp_intersection':  # don't draw future states
            self.drawing_prob = False

        # define simulation
        car_parameter = self.env.car_par
        "theta and lambda pairs (betas):"
        self.theta_list = sim_par["theta"]
        self.lambda_list = sim_par["lambda"]
        self.action_set = sim_par["action_set"]
        # self.action_set_combo = [[-0.05,8], [0,-8], [0.05,-8], [-0.05,-4], [0,-4],
        #                          [0.05,-4], [-0.05,0], [0, 0], [0.05,0], [-0.05,4],
        #                          [0,4], [0.05,4], [-0.05, 8], [0,8], [0.05,8]]  # merging case actions
        self.action_set_combo = [[-0.05, -4], [0.05, -4], [0, 0], [0.05, 4], [-0.05, 0.4]]# CHANGE THIS LATER SUNNY

        if self.env.name == 'merger':
            self.action_set = self.action_set_combo

        # ----------------------------------------------------------------------------------------
        # beta: [theta1, lambda1], [theta1, lambda2], ... [theta2, lambda4] (2x4 = 8 set of betas)
        # betas: [ [theta1, lambda1], [theta1, lambda2], [theta1, lambda3], [theta1, lambda4],
        #          [theta2, lambda1], [theta2, lambda2], [theta2, lambda3], [theta2, lambda4] ]
        # ----------------------------------------------------------------------------------------
        self.beta_set = []
        '1D version of beta'
        for i, theta in enumerate(self.theta_list):
            for j, _lambda in enumerate(self.lambda_list):
                self.beta_set.append([theta, _lambda])

        'getting ground truth betas'
        self.true_params = []
        self.true_params_id = []
        for i, par_i in enumerate(self.env.car_par):
            self.true_params.append(par_i["par"])
            self.true_params_id.append(self.beta_set.index(par_i["par"]))
        'getting initial beliefs of others'
        self.belief_params = []
        for i, par_i in enumerate(self.env.car_par):
            self.belief_params.append(par_i["belief"])

        self.action_distri_1 = []
        self.action_distri_2 = []
        self.belief_weight = belief_weight
        self.initial_belief = self.get_initial_belief(self.env.car_par[1]['belief'][0],  # theta_h
                                                      self.env.car_par[0]['belief'][0],  # theta_m
                                                      self.env.car_par[1]['belief'][1],  # lambda_h
                                                      self.env.car_par[0]['belief'][1],  # lambda_m
                                                      weight=self.belief_weight)  # note: use params from the other agent's belief
        "getting initial b_i"
        if self.inference_type[1] == 'bd_non_empathetic':  # separate table for 2 agents
            self.initial_bi = np.empty(self.n_agents)
            self.initial_bi[0] = self.initial_belief[1][0][2] + self.initial_belief[1][0][3]
            self.initial_bi[1] = self.initial_belief[0][1][2] + self.initial_belief[0][1][3]
        elif self.inference_type[1] == 'bd_empathetic':  # sharing a common table
            self.initial_bi = np.empty(self.n_agents)
            self.initial_bi[0] = self.initial_belief[0][2] + self.initial_belief[0][3]
            self.initial_bi[1] = self.initial_belief[1][2] + self.initial_belief[1][3]

        self.past_loss1 = []  # for storing loss of simulation
        self.past_loss2 = []
        self.policy_correctness = [[], []]  # for the two agents

        if self.n_agents == 2:
            # simulations with 2 cars
            # Note that variable annotation is not supported in python 3.5

            inference_model: List[InferenceModel] = [InferenceModel(inference_type[i], self) for i in range(n_agents)]
            decision_model: List[DecisionModel] = [DecisionModel(decision_type[i], self) for i in range(n_agents)]

            # define agents
            self.agents = [AutonomousVehicle(sim=self, env=self.env, par=car_parameter[i],
                                             inference_model=inference_model[i],
                                             decision_model=decision_model[i],
                                             i=i) for i in range(len(car_parameter))]

        self.draw = True  # visualization during sim
        self.capture = False  # save images during visualization
        # DISPLAY
        if self.draw:

            self.vis = VisUtils(self)  # initialize visualization
            # self.vis.draw_frame()
            # if self.capture:
            #     output_name = datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s")
            #     os.makedirs("./sim_outputs/%s" % output_name)
            #     self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")
            #     self.output_dir = "./sim_outputs/%s/video/" % output_name
            #     os.makedirs(self.output_dir)

    def snapshot(self):
        # take a snapshot of the current system state
        return self.agents.copy()

    def run(self):
        while self.running:
            # Update model here
            if not self.paused:
                for agent in self.agents:
                    agent.update(self)  # Run simulation
            self.calc_loss()
            self.time_stamp.append(self.time)
            # termination criteria

            if self.draw:
                self.vis.draw_frame()  # Draw frame
                # if self.capture:
                #     pg.image.save(v.screen, "%simg%03d.jpeg" % (self.output_dir, self.frame))

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        self.running = False
                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_p:
                            self.paused = not self.paused
                        if event.key == pg.K_q:
                            pg.quit()
                            self.running = False
                        # if event.key == pg.K_d:
                        #     self.car_num_display = ~self.car_num_display

                # Keep fps

            if self.frame >= 35:  # for dt=0.05
                print('Simulation ended with duration exceeded limit')
                break
            # pdb.set_trace()
            x_H = self.agents[0].state[self.frame][0]  # sy_H ??
            x_M = self.agents[1].state[self.frame][0]  # sx_M
            y_H = self.agents[0].state[self.frame][1]  # sy_H
            y_M = self.agents[1].state[self.frame][1]  # sy_M
            if self.env.name == "merger":
                if y_H >= 50 and y_M > 50:
                    print("terminating on vehicle merger:")
                    break
            elif self.env.name == 'bvp_intersection':
                if y_H >= 38 and x_M >= 38:
                    print("terminating on vehicle passed intersection:", y_H, x_M)
                    break
            else:
                if y_H >= 5 and x_M <= -5:
                    # road width = 2.0 m
                    # if crossed the intersection, done or max time reached
                    # if (x_ego >= 0.5 * C.CONSTANTS.CAR_LENGTH + 10. and x_other <= -0.5 * C.CONSTANTS.CAR_LENGTH - 10.):
                    print("terminating on vehicle passed intersection:", x_H, x_M)
                    break

            # draw stuff after each iteration
            # if self.draw:
            #     self.vis.draw_frame()  # Draw frame
            #     # if self.capture:
            #     #     pg.image.save(v.screen, "%simg%03d.jpeg" % (self.output_dir, self.frame))
            #
            #     for event in pg.event.get():
            #         if event.type == pg.QUIT:
            #             pg.quit()
            #             self.running = False
            #         elif event.type == pg.KEYDOWN:
            #             if event.key == pg.K_p:
            #                 self.paused = not self.paused
            #             if event.key == pg.K_q:
            #                 pg.quit()
            #                 self.running = False
            #             # if event.key == pg.K_d:
            #             #     self.car_num_display = ~self.car_num_display
            #
            #     # Keep fps

            if not self.paused:
                self.frame += 1
                self.time += self.dt

        pg.quit()
        if self.env.name == 'bvp_intersection':
            self.write_loss()
            print('writing to cvs file')
            # if self.inference_type[1] == 'bvp_empathetic':
            #     self.write_intent_predict()
        "drawing results"
        self.vis.draw_dist()
        if self.drawing_intent:
            self.vis.draw_intent()
        if self.env.name == 'bvp_intersection':
            self.vis.plot_loss()
        print("-------Simulation results:-------")
        print("inference types:", self.inference_type)
        print("decision types:", self.decision_type)
        print("initial intents:", self.env.car_par[0]['par'], self.env.car_par[1]['par'])
        print("Frames:", self.frame)
        print("len of action and states:", len(self.agents[0].action), len(self.agents[0].state))
        print("action distribution", self.action_distri_1)
        print("Initial belief:", self.initial_belief)
        # print("states of H:", self.agents[0].state)
        # print("states of H predicted by M:", self.agents[1].predicted_states_other)
        print("Action taken by P1:", self.agents[0].action)
        print("Action of P1 predicted by P2:", self.agents[1].predicted_actions_other)
        print("Action taken by P2:", self.agents[1].action)
        # print("lambda prob of P1:", self.vis.lambda_distri_h)
        # print("lambda prob of P2:", self.vis.lambda_distri_m)
        print("Loss of H (p1):", self.past_loss1)
        print("Loss of M (p2):", self.past_loss2)
        if self.inference_type[1] == 'bvp_empathetic':
            print("Count of each belief:", self.agents[1].belief_count[-1])
            policy_count_1, policy_count_2 = self.calc_policy_choice()
            self.write_policy_predict()
            print("Policy correctness:", self.policy_correctness)
            print("Policy correctness for P1:", policy_count_1)
            print("Policy correctness for P2:", policy_count_2)
        loss_1 = np.sum(self.past_loss1) * self.dt
        loss_2 = np.sum(self.past_loss2) * self.dt
        print("agent 1's loss:", loss_1)
        print("sum of 2 agent's loss:", loss_1 + loss_2)
        # print(loss_1 + loss_2)

    def get_initial_belief(self, theta_h, theta_m, lambda_h, lambda_m, weight):
        """
        Obtain initial belief of the params
        :param theta_h:
        :param theta_m:
        :param lambda_h:
        :param lambda_m:
        :param weight:
        :return:
        """
        # given weights for certain param, calculate the joint distribution (p(theta_1), p(lambda_1) = 0.8, ...)
        theta_list = self.theta_list
        lambda_list = self.lambda_list
        beta_list = self.beta_set
        true_params = self.true_params

        # index is 1 because table is updated through agent 1
        if self.inference_type[1] == 'empathetic' or self.inference_type[1] == 'bvp_empathetic':
            # beta_list = beta_list.flatten()
            belief = np.ones((len(beta_list), len(beta_list)))
            for i, beta_h in enumerate(beta_list):  # H: the rows
                for j, beta_m in enumerate(beta_list):  # M: the columns
                    if beta_h[0] == theta_h:  # check lambda
                        belief[i][j] *= weight
                        if beta_h[1] == lambda_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                        if beta_h[1] == lambda_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)

                    if beta_m[0] == theta_m:  # check lambda
                        belief[i][j] *= weight
                        if beta_m[1] == lambda_m:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                        if beta_m[1] == lambda_m:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)

                    # if beta_h == [lambda_h, theta_h] and beta_m == [lambda_m, theta_m]:
                    #     belief[i][j] = weight
                    # else:
                    #     belief[i][j] = 1
            assert round(np.sum(belief)) == 1

        # TODO: should BOTH agent perform inference???
        # TODO: in the case of NE, beta of self should be true param (known beta of self)??
        elif self.inference_type[1] == 'bd_non_empathetic':
            # if agent 0 is non empathetic, it keeps its own version of belief table (fixing)
            belief = ()
            belief_0 = np.ones((self.n_agents, len(beta_list)))
            belief_1 = np.ones((self.n_agents, len(beta_list)))
            "agent 0's initial belief"
            for j, beta0 in enumerate(beta_list):  # col are betas
                if beta0[0] == true_params[0][0]:  # check theta: if its the same as self true param
                    belief_0[0][j] *= weight
                    if beta0[1] == true_params[0][1]:  # check lambda
                        belief_0[0][j] *= weight
                    else:
                        belief_0[0][j] *= (1 - weight) / (len(lambda_list) - 1)
                else:
                    belief_0[0][j] *= (1 - weight) / (len(theta_list) - 1)
                    if beta0[1] == true_params[0][1]:  # check lambda
                        belief_0[0][j] *= weight
                    else:
                        belief_0[0][j] *= (1 - weight) / (len(lambda_list) - 1)

            for j, beta1 in enumerate(beta_list):  # col are betas
                if beta1[0] == theta_m:  # check theta
                    belief_0[1][j] *= weight
                    if beta1[1] == lambda_m:  # check lambda
                        belief_0[1][j] *= weight
                    else:
                        belief_0[1][j] *= (1 - weight) / (len(lambda_list) - 1)
                else:
                    belief_0[1][j] *= (1 - weight) / (len(theta_list) - 1)
                    if beta1[1] == lambda_m:  # check lambda
                        belief_0[1][j] *= weight
                    else:
                        belief_0[1][j] *= (1 - weight) / (len(lambda_list) - 1)

            "agent 1's initial belief"
            for j, beta0 in enumerate(beta_list):  # col are betas
                if beta0[0] == theta_h:  # check theta: if its the same as self true param
                    belief_1[0][j] *= weight
                    if beta0[1] == lambda_h:  # check lambda
                        belief_1[0][j] *= weight
                    else:
                        belief_1[0][j] *= (1 - weight) / (len(lambda_list) - 1)
                else:
                    belief_1[0][j] *= (1 - weight) / (len(theta_list) - 1)
                    if beta0[1] == lambda_h:  # check lambda
                        belief_1[0][j] *= weight
                    else:
                        belief_1[0][j] *= (1 - weight) / (len(lambda_list) - 1)

            for j, beta1 in enumerate(beta_list):  # col are betas
                if beta1[0] == true_params[1][0]:  # check theta
                    belief_1[1][j] *= weight
                    if beta1[1] == true_params[1][1]:  # check lambda
                        belief_1[1][j] *= weight
                    else:
                        belief_1[1][j] *= (1 - weight) / (len(lambda_list) - 1)
                else:
                    belief_1[1][j] *= (1 - weight) / (len(theta_list) - 1)
                    if beta1[1] == true_params[1][1]:  # check lambda
                        belief_1[1][j] *= weight
                    else:
                        belief_1[1][j] *= (1 - weight) / (len(lambda_list) - 1)
            belief = (belief_0, belief_1)
            assert np.sum(belief[0][0]) == 1

            # TODO: do NE and E have different initial belief???

        elif self.inference_type[1] == 'bd_empathetic':  # belief distribution, sharing belief
            belief = np.ones((self.n_agents, len(beta_list)))
            for i in range(self.n_agents):
                for j, beta in enumerate(beta_list):
                    if i == 0:
                        if beta[0] == theta_h:  # check lambda
                            belief[i][j] *= weight
                            if beta[1] == lambda_h:  # check theta
                                belief[i][j] *= weight
                            else:
                                belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                            if beta[1] == lambda_h:  # check theta
                                belief[i][j] *= weight
                            else:
                                belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    if i == 1:
                        if beta[0] == theta_m:  # check lambda
                            belief[i][j] *= weight
                            if beta[1] == lambda_m:  # check theta
                                belief[i][j] *= weight
                            else:
                                belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                            if beta[1] == lambda_m:  # check theta
                                belief[i][j] *= weight
                            else:
                                belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
            assert np.sum(belief[0]) == 1
            assert np.sum(belief[1]) == 1

        elif self.inference_type[1] == 'none':
            belief = 1  # no inference

        # not in use! we only use the game theoretic inference
        else:  # get belief on H agent only
            belief = np.ones((len(lambda_list), len(theta_list)))
            for i, lamb in enumerate(lambda_list):
                for j, theta in enumerate(theta_list):
                    if lamb == lambda_h:  # check lambda
                        belief[i][j] *= weight
                        if theta == theta_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                        if theta == theta_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
        # THIS SHOULD NOT NEED TO BE NORMALIZED!
        # print(belief, np.sum(belief))
        # assert round(np.sum(belief)) == 1
        return belief

    def reset(self):
        # reset the simulation
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0

    def postprocess(self):
        # import matplotlib.pyplot as plt
        # import numpy as np
        # car_1_theta = np.empty((0, 2))
        # car_2_theta = np.empty((0, 2))
        # for t in range(self.frame):
        #     car_1_theta = np.append(car_1_theta, np.expand_dims(self.sim_data.car2_theta_probability[t], axis=0), axis=0)
        #     car_2_theta = np.append(car_2_theta, np.expand_dims(self.sim_data.car1_theta_probability[t], axis=0), axis=0)
        # plt.subplot(2, 1, 1)
        # plt.plot(range(1,self.frame+1), car_1_theta[:,0], range(1,self.frame+1), car_1_theta[:,1])
        # plt.subplot(2, 1, 2)
        # plt.plot(range(1,self.frame+1), car_2_theta[:,0], range(1,self.frame+1), car_2_theta[:,1])
        # plt.show()
        # # pickle.dump(self.sim_data, self.sim_out, pickle.HIGHEST_PROTOCOL)
        # print('Output pickled and dumped.')
        # if self.capture:
        #     # Compile to video
        #     # os.system("ffmpeg -f image2 -framerate 1 -i %simg%%03d.jpeg %s/output_video.mp4 " % (self.output_dir, self.output_dir))
        #     img_list = [self.output_dir+"img"+str(i).zfill(3)+".jpeg" for i in range(self.frame)]
        #     import imageio
        #     images = []
        #     for filename in img_list:
        #         images.append(imageio.imread(filename))
        #     imageio.mimsave(self.output_dir+'movie.gif', images)
        #     #
        #     # # Delete images
        #     # [os.remove(self.output_dir + file) for file in os.listdir(self.output_dir) if ".jpeg" in file]
        #     # print("Simulation video output saved to %s." % self.output_dir)
        # print("Simulation ended.")
        pass

    def calc_loss(self):
        """
        Calculate loss function after each step
        :return:
        """
        state_h = self.agents[0].state[self.frame]
        state_m = self.agents[1].state[self.frame]
        xh = t.tensor(state_h[1], requires_grad=True, dtype=t.float32)
        xm = t.tensor(state_m[0], requires_grad=True, dtype=t.float32)
        theta1 = self.true_params[0][0]
        theta2 = self.true_params[1][0]
        R1 = 70
        R2 = 70
        W1 = 1.5
        W2 = 1.5
        L1 = 3
        L2 = 3
        beta = 10000.
        x1_in = (xh - R1 / 2 + theta1 * W2 / 2) * 10
        x1_out = -(xh - R1 / 2 - W2 / 2 - L1) * 10
        x2_in = (xm - R2 / 2 + theta2 * W1 / 2) * 10
        x2_out = -(xm - R2 / 2 - W1 / 2 - L2) * 10

        Collision_F_x = beta * t.sigmoid(x1_in) * t.sigmoid(x1_out) * \
                        t.sigmoid(x2_in) * t.sigmoid(x2_out)
        U1 = self.agents[0].action[self.frame]
        U2 = self.agents[1].action[self.frame]
        L1 = U1 ** 2 + Collision_F_x.detach().numpy()
        L2 = U2 ** 2 + Collision_F_x.detach().numpy()
        # L1 = 1*Collision_F_x.detach().numpy()
        # L2 = 1*Collision_F_x.detach().numpy()
        self.past_loss1.append(L1)
        self.past_loss2.append(L2)
        return

    def write_loss(self):
        states_1 = self.agents[0].state
        states_2 = self.agents[1].state
        x1 = []
        x2 = []
        time_stamp = self.time_stamp
        for i in range(len(states_1) - 1):
            x1.append(states_1[i][1])
            x2.append(states_2[i][0])
        assert len(x1) == len(self.past_loss1)
        # assert len(time_stamp) == len(self.past_loss1)

        'writing to csv file'

        filename = 'experiment/' + 'traj_loss' + str(x1[0]) + str(x2[0])+'_'\
                   + str(self.env.car_par[0]['par'][0]) + str(self.env.car_par[1]['par'][0])+'.csv'
        with open(filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(time_stamp)
            csv_writer.writerow(self.past_loss1)  # loss1 should be same as loss2
            csv_writer.writerow(self.past_loss2)
            csv_writer.writerow(x1)
            csv_writer.writerow(x2)

        return

    def calc_policy_choice(self):  # calculate if the correct param is predicted for each agent
        half = round(len(self.beta_set) / 2)
        policy_choice = self.agents[1].policy_choice[-1]
        for i in range(self.n_agents):
            if self.true_params_id[i-1] < half:  # meaning true param of the other is NA
                for choice in policy_choice[i]:
                    if choice < half:  # same as true param
                        self.policy_correctness[i].append(1)  # 1 for true
                    else:
                        self.policy_correctness[i].append(0)  # 0 for false
            else:  # true param of the other is A
                for choice in policy_choice[i]:
                    if choice > half:  # same as true param
                        self.policy_correctness[i].append(1)  # 1 for true
                    else:
                        self.policy_correctness[i].append(0)  # 0 for false
        assert len(self.policy_correctness[0]) == len(policy_choice[0])
        count_1 = {0: self.policy_correctness[0].count(0), 1: self.policy_correctness[0].count(1)}
        count_2 = {0: self.policy_correctness[1].count(0), 1: self.policy_correctness[1].count(1)}
        return count_1, count_2

    def write_policy_predict(self):
        states_1 = self.agents[0].state
        states_2 = self.agents[1].state
        x1 = []
        x2 = []
        time_stamp = self.time_stamp
        "getting probability of ground true intent"
        # true_theta_1 = self.true_params[0][0]
        # true_theta_2 = self.true_params[1][0]
        # true_id_1 = self.theta_list.index(true_theta_1)
        # true_id_2 = self.theta_list.index(true_theta_2)
        policy_choice_1 = self.policy_correctness[0]
        policy_choice_2 = self.policy_correctness[1]
        for i in range(len(states_1) - 1):  # inference will be behind
            x1.append(states_1[i][1])
            x2.append(states_2[i][0])
        # assert len(x1) == len(intent_prob_1)
        # assert len(time_stamp) == len(self.past_loss1)

        'writing to csv file'
        filename = 'experiment/' + 'traj_policy_choice' + str(x1[0]) + str(x2[0]) + '_' \
                   + str(self.env.car_par[0]['par'][0]) + str(self.env.car_par[1]['par'][0]) + '.csv'
        with open(filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(time_stamp)
            csv_writer.writerow(policy_choice_1)  # loss1 should be same as loss2
            csv_writer.writerow(policy_choice_2)
            csv_writer.writerow(x1)
            csv_writer.writerow(x2)

    def write_intent_predict(self):  # not in use
        states_1 = self.agents[0].state
        states_2 = self.agents[1].state
        x1 = []
        x2 = []
        time_stamp = self.time_stamp
        "getting probability of ground true intent"
        true_theta_1 = self.true_params[0][0]
        true_theta_2 = self.true_params[1][0]
        true_id_1 = self.theta_list.index(true_theta_1)
        true_id_2 = self.theta_list.index(true_theta_2)
        # intent_prob_1 = self.vis.intent_distri_h[true_id_1]
        # intent_prob_2 = self.vis.intent_distri_m[true_id_2]
        intent_prob_1 = self.vis.true_intent_prob_h
        intent_prob_2 = self.vis.true_intent_prob_m
        for i in range(len(states_1) - 1):  # inference will be behind
            x1.append(states_1[i][1])
            x2.append(states_2[i][0])
        # assert len(x1) == len(intent_prob_1)
        # assert len(time_stamp) == len(self.past_loss1)

        'writing to csv file'
        filename = 'experiment/' + 'traj_intent_prob' + str(x1[0]) + str(x2[0]) + '_' \
                   + str(self.env.car_par[0]['par'][0]) + str(self.env.car_par[1]['par'][0]) + '.csv'
        with open(filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(time_stamp)
            csv_writer.writerow(intent_prob_1)  # loss1 should be same as loss2
            csv_writer.writerow(intent_prob_2)
            csv_writer.writerow(x1)
            csv_writer.writerow(x2)



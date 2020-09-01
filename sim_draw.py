# TODO: add uncertainty visualization

import pygame as pg
import pygame.gfxdraw
import numpy as np
from matplotlib import pyplot
import math
from inference_model import InferenceModel
from autonomous_vehicle import AutonomousVehicle
import time

LIGHT_GREY = (230, 230, 230)


class VisUtils:

    def __init__(self, sim):
        "for drawing state distribution"

        self.sim = sim
        self.env = sim.env
        self.drawing_prob = sim.drawing_prob
        if self.drawing_prob:
            self.p_state_H = sim.agents[1].predicted_states_other  # get the last prediction
            self.past_state_h = sim.agents[1].state[-1]
            self.past_state_m = sim.agents[0].state[-1]
            self.intent_h = []
            self.intent_m = []
            self.intent_distri_h = [[], []]  # theta1, theta2
            self.intent_distri_m = [[], []]  # theta1, theta2
        self.frame = sim.frame
        self.dist = []


        if sim.decision_type == 'baseline':

            self.screen_width = 10  # 50
            self.screen_height = 10  # 50
            self.coordinate_scale = 80
            self.zoom = 0.16
            self.asset_location = "assets/"
            self.fps = 24  # max framework

            img_width = int(self.env.car_width * self.coordinate_scale * self.zoom)
            img_height = int(self.env.car_length * self.coordinate_scale * self.zoom)

            "initialize pygame"
            pg.init()
            self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                               self.screen_height * self.coordinate_scale))
            # self.car1_image = pg.transform.rotate(pg.image.load(self.asset_location + self.sim.agents[0].car_par["sprite"]),
            #                                       -self.sim.agents[0].car_par["orientation"])
            #
            # self.car2_image = pg.transform.rotate(pg.image.load(self.asset_location+ self.sim.agents[1].car_par["sprite"]),
            #                                       -self.sim.agents[1].car_par["orientation"])
            self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
                                                                                   + self.sim.agents[i].car_par["sprite"]),
                                                                     (img_width, img_height)),
                                                  -self.sim.agents[i].car_par["orientation"])
                              for i in range(self.sim.n_agents)]

            #self.origin = np.array([-15.0, 15.0])
            self.origin = np.array([0, 0])

        else:
            self.screen_width = 5
            self.screen_height = 5
            self.asset_location = "assets/"
            self.fps = 24  # max framework
            self.coordinate_scale = 100
            self.zoom = 0.3

            "initialize pygame"
            pg.init()
            self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                               self.screen_height * self.coordinate_scale))

            img_width = int(self.env.car_width * self.coordinate_scale * self.zoom)
            img_height = int(self.env.car_length * self.coordinate_scale * self.zoom)

            "loading car image into pygame"
            self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location
                                                                                   + self.sim.agents[i].car_par[
                                                                                       "sprite"]),
                                                                     (img_width, img_height)),
                                                  -self.sim.agents[i].car_par["orientation"])
                              for i in range(self.sim.n_agents)]

            self.origin = np.array([1.0, -1.0])

        "Draw Axis Lines"
        self.screen.fill((255, 255, 255))
        self.draw_axes() #calling draw axis function
        pg.display.flip()
        pg.display.update()

    def draw_frame(self):
        # Draw the current frame
        frame = self.sim.frame

        # render 10 times for each step
        steps = 20

        "--dummy data--"
        black = (0, 0, 0)
        #red = (255, 0, 0)
        #r = pg.Color.r
        p_state = (0.25, [0, 0, 0, 0]) #[p_state, (sx, sy, vx, vy)]
        sx = p_state[1][0]
        sy = p_state[1][1]
        #pos1 = (sx, sy)
        pos2 = self.c2p((sx, sy))
        # def draw_circle( pos, color, radius):
        #     pg.draw.circle(self.screen, color, pos, radius) #surface,  color, (x, y),radius>=1
        #draw_circle(pos1, red, 10)
        #print('IMPORTED p state: ', self.p_state_H[-1])
        "--end of dummy data--"
        if frame == 0:
            for i in range(self.sim.n_agents):
                pos = np.array(self.sim.agents[i].state[frame][:2])  # get 0 and 1 element (not include 2)
                "smooth out the movement between each step"
                #pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
                "transform pos"
                pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()
                self.screen.blit(self.car_image[i],
                                (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
            "drawing the map of state distribution"
            pg.draw.circle(self.screen, (255, 255, 255), pos2, 10)  # surface,  color, (x, y),radius>=1
            if self.drawing_prob:
                self.draw_prob()  # calling function to draw with data from inference

            # Annotations
            # font = pg.font.SysFont("Arial", 30)
            font = pg.font.SysFont("Arial", 15)
            screen_w, screen_h = self.screen.get_size()
            label_x = screen_w - 550
            label_y = 260
            label_y_offset = 30
            #TODO: length of state/action is mismatched from frame: frame behind by 1
            pos_h, speed_h = self.sim.agents[0].state[-1][1], self.sim.agents[0].state[-1][3]
            label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
                                (0, 0, 0))
            self.screen.blit(label, (label_x, label_y))
            pos_m, speed_m = self.sim.agents[1].state[-1][0], self.sim.agents[1].state[-1][2]
            label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
                                (0, 0, 0))
            self.screen.blit(label, (label_x, label_y+label_y_offset))
            action1, action2 = self.sim.agents[0].action[-1], self.sim.agents[1].action[-1]
            label = font.render("Car 1 action: (%5.4f)" % action1, 1, (0, 0, 0))
            self.screen.blit(label, (label_x, label_y + 2*label_y_offset))
            label = font.render("Car 2 action: (%5.4f)" % action2, 1, (0, 0, 0))
            self.screen.blit(label, (label_x, label_y + 3*label_y_offset))
            label = font.render("Frame: %i" % self.sim.frame, 1, (0, 0, 0))
            self.screen.blit(label, (10, 10))
            pg.display.flip()
            pg.display.update()
        else:
            for k in range(1, steps + 1):
                self.screen.fill((255, 255, 255))
                self.draw_axes()
                # Draw Images
                for i in range(self.sim.n_agents):
                    "getting pos of agent"
                    pos_old = np.array(self.sim.agents[i].state[frame - 1][:2])
                    pos_new = np.array(self.sim.agents[i].state[frame][:2])  # get 0 and 1 element (not include 2)
                    "smooth out the movement between each step"
                    pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
                    "transform pos"
                    pixel_pos_car = self.c2p(pos)
                    size_car = self.car_image[i].get_size()
                    self.screen.blit(self.car_image[i],
                                     (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
                    if self.sim.decision_type == "baseline":
                        time.sleep(0.05)
                # Annotations
                # font = pg.font.SysFont("Arial", 30)
                font = pg.font.SysFont("Arial", 15)
                screen_w, screen_h = self.screen.get_size()
                label_x = screen_w - 555
                label_y = 260
                label_y_offset = 30
                pos_h, speed_h = self.sim.agents[0].state[-1][1], self.sim.agents[0].state[-1][3]
                label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
                                    (0, 0, 0))
                self.screen.blit(label, (label_x, label_y))
                pos_m, speed_m = self.sim.agents[1].state[-1][0], self.sim.agents[1].state[-1][2]
                label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
                                    (0, 0, 0))
                self.screen.blit(label, (label_x, label_y+ label_y_offset))
                action1, action2 = self.sim.agents[0].action[-1], self.sim.agents[1].action[-1]
                label = font.render("Car 1 action: (%5.4f)" % action1, 1, (0, 0, 0))
                self.screen.blit(label, (label_x, label_y + 2*label_y_offset))
                label = font.render("Car 2 action: (%5.4f)" % action2, 1, (0, 0, 0))
                self.screen.blit(label, (label_x, label_y + 3*label_y_offset))
                label = font.render("Frame: %i" % self.sim.frame, 1, (0, 0, 0))
                self.screen.blit(label, (10, 10))

                "drawing the map of state distribution"
                #pg.draw.circle(self.screen, (255, 255, 255), pos2, 10)  # surface,  color, (x, y),radius>=1  # test
                if self.drawing_prob:
                    self.draw_prob() #calling function to draw with data from inference

                pg.display.flip()
                pg.display.update()

        self.calc_dist()
        if self.drawing_prob:
            self.calc_intent()

    def draw_axes(self):
        # draw lanes based on environment TODO: lanes are defined as bounds of agent state spaces, need to generalize
        #pg.draw.line(self.screen, LIGHT_GREY, self.c2p((-10, 10)), self.c2p((10, -10)), 1)  # testing
        for a in self.env.bounds:
            bound_x, bound_y = a[0], a[1]
            if bound_x:
                b_min, b_max = bound_x[0], bound_x[1]
                _bound1 = self.c2p((b_min, 0))
                _bound2 = self.c2p((b_max, 0))
                bounds = np.array([_bound1[0], _bound2[0]])
                pg.draw.line(self.screen, LIGHT_GREY, ((bounds[1] + bounds[0])/2, 1),
                             ((bounds[1] + bounds[0])/2, self.screen_height * self.coordinate_scale,
                              ), bounds[1] - bounds[0])
            if bound_y:
                b_min, b_max = bound_y[0], bound_y[1]
                _bound1 = self.c2p((0, b_min))
                _bound2 = self.c2p((0, b_max))
                bounds = np.array([_bound1[1], _bound2[1]])
                pg.draw.line(self.screen, LIGHT_GREY, (1, (bounds[1] + bounds[0]) / 2),
                             (self.screen_width * self.coordinate_scale,
                              (bounds[1] + bounds[0]) / 2), bounds[0] - bounds[1])

    def draw_dist(self):
        #TODO: implement plotting of distance between cars over time
        # pyplot.plot(self.dist)
        # pyplot.plot(self.sim.agents[0].action)
        fig1, (ax1, ax2, ax3) = pyplot.subplots(3) #3 rows
        fig1.suptitle('Euclidean distance and Agent Actions')
        ax1.plot(self.dist, label='car dist')
        ax1.legend()
        ax1.set(xlabel='time', ylabel='distance')

        #fig1, (ax2) = pyplot.subplots(1)
        #fig1.suptitle('Actions of H at each time')
        ax2.plot(self.sim.agents[0].action, label='actual')
        ax2.plot(self.sim.agents[1].predicted_actions_other, label='predicted', linestyle='--')
        ax2.set_ylim([-10, 10])
        ax2.set_yticks([-8, -4, 0, 4, 8])
        ax2.legend()
        ax2.set(xlabel='time', ylabel='H actions')

        #fig1, (ax3) = pyplot.subplots(1)
        #fig1.suptitle('Actions of M at each time')
        ax3.plot(self.sim.agents[1].action, label='actual')
        ax3.set_ylim([-10, 10])
        ax3.set_yticks([-8, -4, 0, 4, 8])
        ax3.legend()
        ax3.set(xlabel='time', ylabel='M actions')
        # pyplot.ylabel("distance")
        # pyplot.xlabel("time")

        pyplot.show()

    def calc_dist(self):
        past_state_h = self.sim.agents[0].state[-1]
        past_state_m = self.sim.agents[1].state[-1]
        dist_h = past_state_h[1]
        dist_m = past_state_m[0]
        dist = np.sqrt(dist_h * dist_h + dist_m * dist_m)
        self.dist.append(dist)

    def calc_intent(self):
        joint_infer_m = self.sim.agents[0].predicted_intent_other
        # TODO: assign list of theta and lambda somewhere in sim
        theta_list = [1, 1000]
        lambda_list = [0.05, 0.1, 1, 10]
        if not len(joint_infer_m) == 0:
            p_joint_h, lambda_h = self.sim.agents[1].predicted_intent_other
            p_joint_m, lambda_m = joint_infer_m
            sum_h = p_joint_h.sum(axis=0)
            sum_h = np.ndarray.tolist(sum_h)
            sum_m = p_joint_m.sum(axis=0)
            sum_m = np.ndarray.tolist(sum_m) # [theta1, theta2]
            #TODO: add sum to list
            idx_h = sum_h.index(max(sum_h))
            idx_m = sum_m.index(max(sum_m))
            for i in range(len(sum_h)):
                if not len(self.intent_distri_h) == len(sum_h):  # create 2D array
                    j = 0
                    while j in range(len(sum_h)):
                        self.intent_distri_h.append([])
                        self.intent_distri_m.append([])
                self.intent_distri_h[i].append(sum_h[i])
                self.intent_distri_m[i].append(sum_m[i])
            H_intent = theta_list[idx_h]
            M_intent = theta_list[idx_m]
            self.intent_h.append(H_intent)
            self.intent_m.append(M_intent)
        else:
            p_joint_h, lambda_h = self.sim.agents[1].predicted_intent_other[-1]
            #print("-draw- p_joint_h: ", p_joint_h)
            sum_h = p_joint_h.sum(axis=0)
            sum_h = np.ndarray.tolist(sum_h)
            for i in range(len(sum_h)):
                if not len(self.intent_distri_h) == len(sum_h):  # create 2D array
                    j = 0
                    while j in range(len(sum_h)):
                        self.intent_distri_h.append([])
                self.intent_distri_h[i].append(sum_h[i])

            #print('sum of theta prob:', sum_h)
            idx_h = sum_h.index(max(sum_h))
            # TODO: assign list of theta and lambda somewhere in sim
            H_intent = theta_list[idx_h]
            print('probability of thetas H:', sum_h, 'H intent:', H_intent)
            self.intent_h.append(H_intent)

    def draw_intent(self):
        joint_infer_m = self.sim.agents[0].predicted_intent_other
        # TODO: assign list of theta and lambda somewhere in sim
        if not len(joint_infer_m) == 0:
            fig2, (ax1, ax2) = pyplot.subplots(2)
            fig2.suptitle('Predicted intent of other agent')
            ax1.plot(self.intent_h, label='predicted H intent')
            ax1.legend()
            ax1.set_yticks([1, 1000])
            ax1.set_yticklabels(['na', 'a'])

            ax2.plot(self.intent_m, label='predicted M intent')
            ax2.legend()
            ax2.set_yticks([1, 1000])
            ax2.set_yticklabels(['na', 'a'])

        else:
            print(self.intent_h)
            print(self.sim.agents[1].predicted_intent_other)
            fig2, (ax1, ax2) = pyplot.subplots(2)
            fig2.suptitle('Predicted intent of other agent')
            ax1.plot(self.intent_h, label='predicted H intent')
            ax1.legend()
            #TODO: get actual intent from decision model/ autonomous vehicle
            ax1.set_yticks([1, 1000])
            ax1.set_yticklabels(['na', 'a'])

            w = 0.15
            #TODO: generalize for more than two thetas
            x = list(range(0, len(self.intent_h)))
            x1 = [i-w for i in x]
            x2 = [i+w for i in x]
            ax2.bar(x1, self.intent_distri_h[0], width=0.15, label='theta 1')
            ax2.bar(x2, self.intent_distri_h[1], width=0.15, label='theta 2')
            ax2.legend()
            ax2.set_yticks([0.25, 0.5, 0.75])

        #TODO: plot actual distributions
        pyplot.show()

    def draw_prob(self):
        """
        drawing probability distribution of future state on pygame surface
        :params:

        :return:
        """

        "colors"
        red = (255, 0, 0)
        orange = (255, 165, 0)
        yellow = (255, 255, 51)
        green = (204, 255, 153)
        blue = (100, 178, 255)
        purple = (0, 100, 255)

        "get state distribution"
        p_state1 = (0.25, [0, 0, 0, 0])  # [p_state, (sx, sy, vx, vy)]
        #print(self.p_state_H[-1])
        p_state_D, state_list  = self.p_state_H[-1]
        #print("PLOTTING: ", state_list, "and ", p_state_D)

        "unpacking the info"
        for k in range(len(state_list)): #time steps
            states_k = state_list[k]
            p_state_Dk = p_state_D[k]

            for i in range(len(state_list[0])):
                x, y = states_k[i][0], states_k[i][1]
                print("X, Y: ", x, y)
                nx, ny = self.c2p((x, y))
                p_s = p_state_Dk[i]
                #TODO: change the range of color! (we will have different distribution)
                #TODO: continuous distribution of color?
                "plot different colors base on their probabilities"
                if p_s > 0.22:
                    pg.draw.circle(self.screen, red, (nx, ny), 6) #(surface, color, pos, radius)
                elif 0.21 < p_s <= 0.22:
                    pg.draw.circle(self.screen, orange, (nx, ny), 6)
                elif 0.20 < p_s <= 0.21:
                    pg.draw.circle(self.screen, yellow, (nx, ny), 6)
                elif 0.19 < p_s <= 0.2:
                    pg.draw.circle(self.screen, green, (nx, ny), 6)
                elif 0.18 < p_s <= 0.19:
                    pg.draw.circle(self.screen, blue, (nx, ny), 6)
                else:
                    pg.draw.circle(self.screen, purple, (nx, ny), 6)

        #pg.draw.circle()

    def c2p(self, coordinates):
        x = self.coordinate_scale * (coordinates[0] - self.origin[0] + self.screen_width / 2)
        y = self.coordinate_scale * (- coordinates[1] + self.origin[1] + self.screen_height / 2)
        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int(
            (y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_height * self.coordinate_scale * 0.5)
        return np.array([x, y])

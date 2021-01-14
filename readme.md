# Game theoretic belief approximation in two agent interactions

Contributors: Yi Chen, Lei Zhang, Tanner Merry, Sunny Amatya, Yi Ren, Wenlong Zhang

##TO BE EDITED
## Table of Contents
1. [Overview](#overview)
2. [From Honks to Socially Adept Motions](#directions)
3. [Constitutive Model of Socially Adept Decision Making](#motionplanning)
    1.  [The influence of driving strategy on interaction dynamics](#drivingstrategy)
    2.  [The importance of empathy in intent decoding](#empathy)
4. [Driver Simulation Setup](#simulation)
5. [Instruction of reproduce the results](#instruction)

## Overview <a name="overview"></a>

The past decade has witnessed rapid advancement of autonomous driving since [the DARPA urban 
challenge](http://archive.darpa.mil/grandchallenge/). 
Till 2017, there were [46 large corporates](https://www.cbinsights.com/research/autonomous-driverless-vehicles-corporations-list/)
 across the world developing autonomous vehicles (
AVs). In the U.S., California and Arizona have allowed tests of AVs on public roads without human drivers. 
There are currently more than 600 AVs on the public roads in Arizona, including Alphabet's Waymo, 
who has started its [paid road-sharing services](https://www.azcentral.com/story/money/business/tech/2018/01/30/waymo-start-driverless-ride-sharing-phoenix-area-year/1078466001/)
 in Phoenix. Undoubtedly, AVs have the potential to enable 
faster and safer commute, improve road and fuel efficiency, and increase the accessibility of 
transportation for seniors and people with disabilities. They have become a key component for 
future smart cities. However, there is still a long way before AVs can reliably interact with 
pedestrians and other vehicles in a safe, efficient, and socially-graceful manner.

Incidents have been reported where AVs stopped indefinitely at traffic intersections, 
due to misunderstanding of the intent of [pedestrians](https://gizmodo.com/a-cyclists-track-stand-totally-befuddled-one-of-googles-1727024285)
 and [human drivers](https://jalopnik.com/google-is-training-self-driving-cars-to-mimic-human-dri-1733624021). 
 [The first claim](https://www.mercurynews.com/2018/01/23/motorcyclist-hit-by-self-driving-car-in-s-f-sues-general-motors/) against 
an AV manufacturer has been filed against General Motors in 2018, following a collision between 
a Cruise Automation 2016 Chevrolet Bolt and a motorcyclist, who alleged that the car swerved 
into his path without his agreement. Questions have also been raised regarding whether AV 
can reactive to and learn from actions such as 
[honking from other vehicles](http://theconversation.com/driverless-cars-might-follow-the-rules-of-the-road-but-what-about-the-language-of-driving-88824).

The common cause of these incidents is the lack of a mechanism for AVs to understand the 
"social norms" of driving. Such norms include the driving style (e.g., how much space between 
vehicles is considered as acceptable), the driving strategy (e.g., whether one should 
consider others' intents and driving styles during its motion planning), and tacit driving 
rules (e.g., [the "Pittsburgh left"](https://triblive.com/x/pittsburghtrib/sports/s_457936.html#axzz2SAfy5vRK): 
taking a left turn just as the light turns 
green). These norms are often region- and culture-dependent, and may 
change along time. We hypothesize that being cognizant about unique social norms in 
new environments will allow an AV to correctly model interactions, infer intents 
of other agents, and derive socially adept motions (e.g., graceful and legible). 

## From Honks to Socially Adept Motions <a name="directions"></a>
 
The objective of this project is to answer the following scientific questions 
towards the learning of social norms in driving: 

1. **Constitutive model of socially adept decision making?**: What formalism of decision making 
is capable of creating socially adept motions, provided that the model incorporates 
the correct social norms? 

2. **Artificial general intelligence?**: What learning mechanisms can effectively 
update this decision making model based on social feedback such as 
honking from other cars? 

To answer these questions, we build our 
investigation upon decision models that mechanistically derive vehicle 
motions through game-theoretic control, where the control games are 
formulated based on both real-time observations and learned social 
norms. We will then propose a hierarchical learning framework that updates 
the AV's game formulations, and thus its understanding of social norms, 
through semantic reasoning and data-driven learning.

This repo serves as the base for all codes, data, meta-data we create 
during the investigation.

## Constitutive model of socially adept decision making <a name="motionplanning"></a>

To answer the first question, we investigate a formalism of motion planning that enables 
empathy and social awareness of an AV agent. Empathy is defined as the agent's ability 
to answer the question: ``What do others' think about me?'', or In other words, 
to infer others' intents in conjunction with their estimation of the agent's self intent. 
We show that empathy is critical to the correct inference of others' intents. 
Social awareness is defined as the agent's ability to measure the discrepancy 
between what it does and what others may want it to do, and the actions wanted 
by others are those that allow others to achieve high-payoff equilibrium in 
an interaction game. Thus social awareness is built upon empathy. We show 
that incorporating social awareness into motion planning enriches the 
sophisticacy of the agent, allowing it to create passive aggressive motions 
that are more socially graceful than proactive ones.

Details of the motion planning algorithms can be found in our [draft paper][maxpaper].

### The influence of driving strategy on interaction dynamics <a name="drivingstrategy"></a>
In the following, we demonstrate interactions between two agents (M and H) in a simple 
intersection scenario. Details of these interactions can be found in the [draft paper][maxpaper].

The two agents are equipped with three 
different driving strategies, namely, reactive, proactive, and socially-aware:

* A *reactive* agent M plans based on the inferred distribution of future motions of the other agent H. 

* A *proactive* M exploits the fact that H's future motion are dependent on M's. 
In the intersection case, M will act as if it does not care about safety and force H to yield. 
However, when facing an aggressive H who does care less about safety, being proactive can 
cause M to brake abruptly when it realizes H's aggressiveness. 

* A *socially-aware* M is built upon the proactive model, with an additional goal of reducing 
the lack of gracefulness.

#### Case 1: Non-aggressive reactive M vs. Non-aggressive reactive H
Both agents take exactly the same actions at every time step since they share the same settings.
Both switch between fast and slow movements along time, since their observation of the other's slow (fast)
movement lead to their reaction of a fast (slow) movement. 

<img src="./sim_outputs/reactive_reactive_c0c0.gif" alt="Drawing" style="height: 100px;"/> 

#### Case 2: Non-aggressive reactive M vs. Aggressive reactive H
An aggressive H moves at the highest speed all time, forcing M to yield.

<img src="./sim_outputs/reactive_reactive_c0c9.gif" alt="Drawing" style="height: 100px;"/>

#### Case 3: Non-aggressive proactive M vs. Non-aggressive reactive H
Proactive M moves at a high speed in order to make H believe that it is aggressive.

<img src="./sim_outputs/proactive_reactive_c0c0.gif" alt="Drawing" style="height: 100px;"/> 

#### Case 4: Non-aggressive proactive M vs. Aggressive reactive H
M starts by pretending to be aggressive. However, as it approaches the intersection zone,
it realizes that H is aggressive. Thus M stopped immediately and started to back off.

<img src="./sim_outputs/proactive_reactive_c0c9.gif" alt="Drawing" style="height: 100px;"/>

#### Case 5: Non-aggressive socially-aware M vs. Non-aggressive reactive H
Socially-aware M moves in a way that gently force H to yield, leading to less "lack of gracefulness".

<img src="./sim_outputs/social_reactive_c0c0.gif" alt="Drawing" style="height: 100px;"/> 

#### Case 6: Non-aggressive socially-aware M vs. Aggressive reactive H
When H is aggressive, a socially-aware M can avoid the emergency stop as it naturally allows H to pass first.

<img src="./sim_outputs/social_reactive_c0c9.gif" alt="Drawing" style="height: 100px;"/>

### The importance of empathy in intent decoding <a name="empathy"></a>
We investigated the importance of empathy in intent inference using a similar experimental setting 
as above. We show that if M is modeled to believe that H already knows its true intent, 
then M will create false estimation of H's intent. The following videos compare 
the empathetic and non-empathetic cases.

#### Empathetic non-aggressive reactive M vs. Empathetic mildly-aggressive reactive H
M realizes early that H is driving more aggressively than itself, thus decides to yield before
it enters the interaction zone.

<img src="./sim_outputs/empathetic.gif" alt="Drawing" style="height: 100px;"/>

#### Non-empathetic non-aggressive reactive M vs. Empathetic mildly-aggressive reactive H
By fixing its belief of what H thinks about M, M excludes plausible explanations of H's behavior and
fails to correctly infer H's intent. As a result, M only realizes that H is not being non-aggressive after
it encounters H in the interaction zone, and is forced to back off.

<img src="./sim_outputs/non_empathetic.gif" alt="Drawing" style="height: 100px;"/>

### Driving Simulator Setup <a name="simulation"></a>

#### Overview <a name="overview"></a>

A Logitech G29 racing wheel is adopted to build a driving simulator which can be applied to record human drivng behavior.
controlled_vehicle.py contains the dynamic model of human controlled vehicle.

#### Simulator Setup
<img src="./sim_outputs/IMG_0455.jpg" alt="Drawing" style="height: 500px;"/> 

#### Example Demo
Red car was controlled by human.
<img src="./sim_outputs/output_video.gif" alt="Drawing" style="height: 50px;"/> 

[maxpaper]: ./draft.pdf
 
## Instruction of reproducing the results <a name="instruction"></a>
In general, the simulation can be conducted by running main.py. The motion planning methods can be edited by changing $loss\_style$ on line 34 and 38 in main.py. Aggressiveness ($\theta$) of agents are determined on line 129 and 141 of constants.py; acceleration ability ($\alpha$) of agents on line 132 and 144, while $\hat{\alpha}$ on line 133 and 145 of the same file. Courtesy weight ($\beta$) is on line 60 in constants.py. All tables and figures 1-9 of the paper can be reproduced by changing these parameters and run main.py in the root folder. Figure 10 can be reproduced by main.py in /turning\_scene.
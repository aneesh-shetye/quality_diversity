
import torch
import random
import numpy as np 
from typing import List

from sentence_transformers import SentenceTransformer

prefix = """
#Build an agent to solve an environment.
The agent should perform better than the agent given below
#The  environment has a hopper - a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts.

#The input to this agent  is the current state of the environment. Its output should be an action of the form (float, float, float) where each value ranges from -1 to 1.

#This output would represent torques applied on rotors such that: 
#action[0] = torque applied on the thigh rotor
#action[1] = torque applied on the leg rotor
#action[2] = torque applied on teh foot rotor

#The function you are designing would take in a "state" argument which is a 11 dimensional vector:
#state[0] gives the z-coordinate of the torso (height of the hopper),
#state[1] gives the angle of the torso, 
#state[2] gives the angle of the thigh joint, 
#state[3] gives the angle of the foot joint and, 
#state[4] gives the velocity of the x-coordinate (height) of the torso
#state[5] gives the velocity of the x-coordinate of the torso 
#state[6] gives the velocity of the z-coordinate of the torso 
#state[7] gives the angular velocity of the angle of the torso 
#state[8] gives the angular velocity of the thigh hinge 
#state [9] gives the angular velocity of the leg hinge 
#state[10] gives the angular velocity of the foot hinge 

                """
prefix_cartpole = """
#Build an agent to solve an environment.
#The agent should perform better than the agent given below
#The  environment has a single pole on top of a cart which you have to write a control policy for  

#The input to this agent  is the current state of the environment. Its output should be an action of the form an integer which takes values {0, 1} 

#This output would represent the direction of force the cart is pushed with. 
0: Push the cart to the left
1: Push the cart to the right

#The function you are designing would take in a "state" argument which is a 4 dimensional ndarray:
#state[0] gives the cart position 
#state[1] gives the cart velocity
#state[2] gives the pole angle 
#state[3] gives the pole angular velocity  
"""

model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

class Island: 

    def __init__(self, 
                 code: str | list[str], 
                 score: float | int | list[float] | list[int], behaviour: dict | list[dict]): 
        if type(code) == list: 
            self.codes = code 
        elif type(code) == str: 
            self.codes = [code]
        else: 
            raise ValueError("code must be of type List[str] or string") 
        assert type(self.codes[0]) == str,f"code must  be of type List[str] or string instead got code of type {type(self.codes[0])}"


        if type(score) == list: 
            self.scores = score 
        elif type(score) == float or type(score) == int: 
            self.scores = [score]
        else: 
            raise ValueError("score must  be of type List[float | int] or float or int") 
        assert type(self.scores[0]) == int or type(self.scores[0]) == float,f"score must be of type List[float | int] or float or int, got score of type: {type(self.scores[0])}, e.g: {self.scores[0]}"

        if type(behaviour) == list: 
            self.behaviours = behaviour 
        elif type(behaviour) == dict:
            self.behaviours = [behaviour]
        else: 
            raise ValueError("behaviour must  be of type List[dict] or dict") 
        assert type(self.behaviours[0]) == dict ,f"behaviour must be of type List[dict] or dict but got behaviour of type: {type(self.behaviours[0])} instead, e.g: {self.behaviours[0]}"

        assert len(self.codes) == len(self.scores), "codes and scores must be of same length" 

        assert len(self.codes) == len(self.behaviours), "codes and scores must be of same length" 

        self.best_score = max(self.scores) 
        self.best_code = self.codes[np.argmax(self.scores)] 
        self.best_behaviour =  self.behaviours[np.argmax(self.scores)]

        self.codes_embeddings, self.median_embedding = self.calc_median(self.codes) 

    def calc_median(self, codes: list[str]): 
        with torch.no_grad():
            codes_embeddings = model.encode(self.codes) 
        median_embedding = np.median(codes_embeddings, axis=0)

        return codes_embeddings, median_embedding 
        

    def add_code(self, code: str, score: int | float): 
        self.codes.append(code) 
        self.scores.append(score)

        if score > self.best_score: 
            self.best_score = score 
            self.best_code = code 

        self.codes_embeddings, self.median_embedding = self.calc_median(self.codes)

    def get_prompt(self): 

        return prefix + random.sample(self.codes, 1)[0]



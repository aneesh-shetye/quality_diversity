import random 
import json
import logging
import os
import pathlib
import pickle
import time
#import psutils 
import matplotlib.pyplot as plt 
import seaborn as sns 

import heapq
from typing import List, Optional
import click
# import llm
import numpy as np 
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
#from llama import Dialog, Llama
from dotenv import load_dotenv

from programs_database import Island
from evaluator_mod import evaluate_agent
from scoring import kl_divergence, wasserstein, wasserstein_with_weights

import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
#from joblib import Parallel, delayed

lock = threading.Lock()
database_lock = threading.Lock()

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import wandb 
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--output_path', default='data/', type= click.Path(file_okay=False), help='path for logs and data') 
parser.add_argument('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')

#embedding_model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

wandb.init(project='quality_diversity', 
         settings=wandb.Settings(start_method="fork"), 
         reinit=True)

model_id = "meta-llama/Llama-3.3-70B-Instruct"
#model_id = "upiter/TinyCodeLM-400M"
  
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    batch_size=4, 
    padding=True, 
    truncation=True
  )
#tokenizer = AutoTokenizer.from_pretrained(model_id) 
#pipeline.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto') 
#print(f"Memory usage after loading checkpoint: {process.memory_info().rss / (1024**2):.2f} MB")


class Sampler: 
    def __init__(self, model_id: str = "meta-llama/Llama-3.3-70B-Instruct"): 
        #self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
        #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto') 
        #self.device = self.pipeline.device
        #self.pipeline = pipeline
        pass
    
    def generate(self, input_text: str | List[str]): 
        #print(input_text)
        '''
        input_ids = tokenizer(input_text, return_tensors='pt', 
                                   padding=True, truncation=True).to(self.device)
        
        return model.generate(**input_ids, max_length=1024) 
        '''
        response = pipeline(input_text, max_new_tokens=1024)
        #print(response) 
        return [t[0]['generated_text'] for t in response] 

#initial_template: str 
initial_code_hopper = """
Build an agent to solve an environment.  
edit the previous agent `agent_v0`. Name the new agent as `agent_v1` and define it within the `Agent` class. 
The  environment has a hopper - a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts. 

The input to this agent  is the current state of the environment. Its output should be an action of the form (float, float, float) where each value ranges from -1 to 1.

This output would represent torques applied on rotors such that: 
action[0] = torque applied on the thigh rotor
action[1] = torque applied on the leg rotor 
action[2] = torque applied on the foot rotor

The function you are designing would take in a "state" argument which is a 11 dimensional vector:
state[0] gives the z-coordinate of the torso (height of the hopper),
state[1] gives the angle of the torso, 
state[2] gives the angle of the thigh joint, 
state[3] gives the angle of the foot joint and, 
state[4] gives the velocity of the x-coordinate (height) of the torso
state[5] gives the velocity of the x-coordinate of the torso 
state[6] gives the velocity of the z-coordinate of the torso 
state[7] gives the angular velocity of the angle of the torso 
state[8] gives the angular velocity of the thigh hinge 
state [9] gives the angular velocity of the leg hinge 
state[10] gives the angular velocity of the foot hinge 

```python
class Agent(): 
    def __init__(self): 
        pass 
    
    def agent_v0(self,state) -> tuple[float, float, float]:
      #state[0] gives the z-coordinate of the torso (height of the hopper),
      #state[1] gives the angle of the torso, 
      #state[2] gives the angle of the thigh joint, 
      #state[3] gives the angle of the foot joint and, 
      #state[4] gives the velocity of the x-coordinate (height) of the torso
      #state[5] gives the velocity of the x-coordinate of the torso 
      #state[6] gives the velocity of the z-coordinate of the torso 
      #state[7] gives the angular velocity of the angle of the torso 
      #state[8] gives the angular velocity of the thigh hinge #state [9] gives the angular velocity of the leg hinge 
      #state[10] gives the angular velocity of the foot hinge 
      #Given the state output actions that would carry the object to the required position using the robotic arm.
      return (0.0, 0.0, 0.0) 

    #########################################
    #your agent here

    #########################################
```
                """
prefix_prompt_hopper =""" 
Build an agent to solve an environment.  
edit the previous agent `agent_v0`. Name the new agent as `agent_v1` and define it within the `Agent` class. 
The  environment has a hopper - a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts. 

The input to this agent  is the current state of the environment. Its output should be an action of the form (float, float, float) where each value ranges from -1 to 1.

This output would represent torques applied on rotors such that: 
action[0] = torque applied on the thigh rotor
action[1] = torque applied on the leg rotor 
action[2] = torque applied on the foot rotor

The function you are designing would take in a "state" argument which is a 11 dimensional vector:
state[0] gives the z-coordinate of the torso (height of the hopper),
state[1] gives the angle of the torso, 
state[2] gives the angle of the thigh joint, 
state[3] gives the angle of the foot joint and, 
state[4] gives the velocity of the x-coordinate (height) of the torso
state[5] gives the velocity of the x-coordinate of the torso 
state[6] gives the velocity of the z-coordinate of the torso 
state[7] gives the angular velocity of the angle of the torso 
state[8] gives the angular velocity of the thigh hinge 
state [9] gives the angular velocity of the leg hinge 
state[10] gives the angular velocity of the foot hinge 

```python
"""


initial_code_cartpole = """
#Build an agent to solve an environment.  
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

def agent_v0(state) -> int:
"""

initial_code_ip = """
#Change the actions made by the agent. Use the policy-gradient method to solve the environment.  The environment is a cart with an inverted pendulum on it and the agent's task is to keep it upright and not let it fall by pushing the cart by applying force to it.  
#The input to this agent  is the current state of gym's inverted pendulum v-4 environment. This input is a 4 dimensional vector with  state[0] gives the position of the cart, state[1] gives the vertical angle of the pole on the cart, state[2] gives the linear velocity of the cart and state[3] gives the angular velocity of the pole on the cart. 
#Given this state, the agent returns an action which is the force applied to the cart which is a list with a single element, e.g. [0] where the element would range from [-3, 3]
#Improvement is measured by a "score" that is given by running your model in the environemnt. You do not have to implement. Look at the changes you made in the past and try to make  positive changes to increase this score. 
#Make only small changes.
#Try to make the code short.

def agent_v0(state) -> int:
"""

initial_code_dm = """
#Build an agent to solve an environment.  
#This environment is a dm control finger turn easy enviroment.

# Input - is an OrderedDict with 'position', 'velocity', 'touch', 'target_position', 'dist_to_target' keys
# example - OrderedDict([('position', array([ 1.33423858,  0.81376991, -0.02043611, -0.12838367])), ('velocity', array([-2.76861475,  0.72014295,  0.        ])), ('touch', array([0., 0.])), ('target_position', array([ 0.08670474, -0.09686221])), ('dist_to_target', 0.04168152605095278)])
# The input to this function is the current state of gym's finger turn-easy environment. 
# state['position'] is an array of length 3 containing the position of the fingertips in the world frame
# state['velocity'] is an array of length 3 containing the velocities of the fingertips in the world frame
# state['touch'] is an array of length 2 containing the touch sensors of the fingertips
# state['target_position'] is an array of length 3 containing the position of the target in the world frame
# state['dist_to_target'] is the distance between the fingertips and the target
# output is a BoundedArray(shape=(2,), dtype=dtype('float64'), name=None, minimum=[-1. -1.], maximum=[1. 1.])
# example - [0.81395884 0.81576687]


#Complete the code for the agent that solves the environment. 
#do not use neural networks or deep RL techniques to do so

def agent_v0(state) -> int: 
"""


#INITIALIZE  DATABASE with 0 islands  
global database
database = {}
initial_code = """
class Agent(): 
    def __init__(self): 
        pass 
    
    def agent_v0(self,state) -> tuple[float, float, float]:
      #state[0] gives the z-coordinate of the torso (height of the hopper),
      #state[1] gives the angle of the torso, 
      #state[2] gives the angle of the thigh joint, 
      #state[3] gives the angle of the foot joint and, 
      #state[4] gives the velocity of the x-coordinate (height) of the torso
      #state[5] gives the velocity of the x-coordinate of the torso 
      #state[6] gives the velocity of the z-coordinate of the torso 
      #state[7] gives the angular velocity of the angle of the torso 
      #state[8] gives the angular velocity of the thigh hinge #state [9] gives the angular velocity of the leg hinge 
      #state[10] gives the angular velocity of the foot hinge 
      #Given the state output actions that would carry the object to the required position using the robotic arm.
      return (0.0, 0.0, 0.0) 
"""
for i in range(10): 
    database[i] = initial_code

time_step = 0 
if __name__ == "__main__": 

    for t in range(int(1e4)): 


        print(database)
        prompts = [prefix_prompt_hopper + database[i] + '\n```'  for i in range(10)] 
        start_time = time.time()
        #print('prompting LLM') 
        #print(f'prompts: {prompts}') 
        #code = sampler.generate(prompts)
        response = pipeline(prompts, max_new_tokens=1024)
        print("########################################")
        print(response)
        print("########################################")
        codes = [t[0]['generated_text'] for t in response]

        start_time = time.time()

        for i, c in enumerate(codes): 
            start_time = time.time()
            result = evaluate_agent(c)
            #result: [code, total_reward, normalized_distribution, quantized action]

            if result[1]: 
                database[i] = result[0] 
                wandb.log({f'score_island_{i}': result[1]})
                wandb.log({f'program_{i}':wandb.Html(f'<pre>{result[0]}</pre>')})




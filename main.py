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
from evaluator import evaluate_agent
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

model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "upiter/TinyCodeLM-400M"
  
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    batch_size=4
  )
#tokenizer = AutoTokenizer.from_pretrained(model_id) 
pipeline.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
        print(response) 
        return [t[0]['generated_text'] for t in response] 

#initial_template: str 
initial_code_hopper = """
#Build an agent to solve an environment.  
#The  environment has a hopper - a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts. 

#The input to this agent  is the current state of the environment. Its output should be an action of the form (float, float, float) where each value ranges from -1 to 1.

#This output would represent torques applied on rotors such that: 
#action[0] = torque applied on the thigh rotor
#action[1] = torque applied on the leg rotor 
#action[2] = torque applied on the foot rotor

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

def agent_v0(state) -> tuple[float, float, float]:
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
#database is a dictionary with island_id as keys and Islands as values
#Islands: 
#attributes:
    #codes: List[str] 
    #scores:List[float] 
    #best_performing_code:str
    #best_score:float
    #median_code_embedding: torch.tensor (median(embedding_function(codes)) 
#methods: 
    #add_code -> adds code to codes 
    #update_best_performing_code 
    #calculate_median_code_embedding

#INITIALZE SAMPLER
#sampler is a simple LLM (pipline.generate) 
#sampler = Sampler()

#INITIALIZE EVALUATOR 
#evaluator: 
#gets text that is corresponds to our function 
#makes it into a runnable program
#runs the function in the environment in a sandbox say 10 times  
#returns a score  

#INITIALIZE A DISTANCE FUNCTION e.g. DBSCAN 

##process = psutils.Process(os.getpid())

'''
def monitor_memory(interval=0.5):
    process = psutils.Process(os.getpid())
    while True:
        rss_mb = process.memory_info().rss / (1024 ** 2)
        print(f"[Memory Monitor] RSS: {rss_mb:.2f} MB")
        time.sleep(interval)

# Start the memory monitor thread
monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
monitor_thread.start()
'''


def process_pair(score, code, behaviour): 
    if score is None: 
        return 

    #with lock: 
    if len(database.keys())==0: 
        database[0] = Island(code=[code], score=[score], behaviour=[behaviour])
        wandb.log({f'score_island_{len(database.keys())}': score})
        wandb.log({f'program_{len(database.keys())}':wandb.Html(f'score: {score}<pre>{code}</pre>')})
    else: 
        update_database(code, score, behaviour, threshold=1.5)


def update_database(candidate_code, candidate_score, candidate_behaviour, threshold):
    """
    Update the global dataset with a candidate agent.
    
    - candidate_behaviour: A dictionary representing the probability distribution over actions.
    - threshold: A KL divergence threshold used to decide if the candidate is too dissimilar.
    
    If the candidate's behaviour (as measured by KL divergence) is not similar enough to any
    existing island, a new Island is created. Otherwise, if the candidate is similar and its score
    is higher than the best in that island, the candidate is appended to that island and the best values are updated.
    """
    #print(candidate_behaviour)
    with database_lock:
        min_kl = float('inf')
        closest_index = None
        # Loop over islands and compute KL divergence.
        for idx, island in database.items():
            #print(f'candidate_behaviour: {candidate_behaviour}') 
            #print(f'island.best_behaviour: {island.best_behaviour}') 
            #divergence = kl_divergence(candidate_behaviour, island.best_behaviour)
            divergence = wasserstein_with_weights(np.array([np.array(t) for t in candidate_behaviour.keys()]), np.array([np.array(t) for t in island.best_behaviour.keys()]), np.array([np.array(t) for t in candidate_behaviour.values()]), np.array([np.array(t) for t in island.best_behaviour.values()]))
            if divergence < min_kl:
                min_kl = divergence
                closest_index = idx

        # If no island is similar enough (KL divergence above threshold) or no island exists:
        if min_kl > threshold or closest_index is None:
            new_index = len(database)
            database[new_index] = Island(candidate_code, candidate_score, candidate_behaviour)
            wandb.log({f'score_island_{new_index}': candidate_score})
            wandb.log({f'program_{new_index}':wandb.Html(f'score: {candidate_score}<pre>{candidate_code}</pre>')})
            #print(f"Created new Island {new_index} (score: {candidate_score}, KL divergence: {min_kl:.4f})")
        else:
            # Candidate is similar to an existing island.
            island = database[closest_index]
            if candidate_score > island.best_score:
                # Append the new candidate and update best values.
                island.codes.append(candidate_code)
                island.scores.append(candidate_score)
                island.best_score = candidate_score
                island.best_code = candidate_code
                wandb.log({f'score_island_{closest_index}': candidate_score})
                wandb.log({f'program_{closest_index}':wandb.Html(f'score: {candidate_score}<pre>{candidate_code}</pre>')})
                #print(f"Updated Island {closest_index} with a higher score: {candidate_score} (prev: {island.best_score})")
            #else:
                #print(f"Candidate rejected for Island {closest_index} (score: {candidate_score} <= best: {island.best_score})")



time_step = 0 
if __name__ == "__main__": 

    '''
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn') 
    '''

    for t in range(int(1e4)): 
        #print("Entered the for loop")
        #code = sampler.generate(get_prompt_from_existing_islands) 
        #print(database) 
        #print(f'database.keys: {database.keys()}')

        if len(database.keys()) < 10: 
            prompt = initial_code_hopper
            prompts = [initial_code_hopper]*10
        else: 
            prompt = database[random.sample(list(database.keys()), 1)[0]].get_prompt()
            prompts = [i.get_prompt() for i in database.values()]

        start_time = time.time()
        #print('prompting LLM') 
        print(f'prompts: {prompts}') 
        #code = sampler.generate(prompts)
        response = pipeline(prompts, max_new_tokens=1024)
        codes = [t[0]['generated_text'] for t in response]

        #codes = tokenizer.batch_decode(code)  
        #codes = code
        #print(f'sample in sampler: {codes}')
        #print(f"time taken to generate code: {time.time() - start_time}")

        start_time = time.time()
        '''
        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(evaluate_agent, codes))
        '''
        for c in codes: 
            start_time = time.time()
            result = evaluate_agent(c)
            #print(f"time taken to evaluate code: {time.time() - start_time}") 
            end_time = time.time()
            process_pair(result[1], result[0], result[2])
            #print(f"time taken to put evaluated code into islands:{time.time() - end_time}")

        '''

        end_time = time.time()

        print(f"time taken to evaluate code: {end_time - start_time}") 
        
        #print(results)

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submitting tasks for each (score, code) pair.
            futures = [executor.submit(process_pair, result[1], result[0], result[2]) for result in results]
            # Optionally, wait for all futures to complete.
            for future in futures:
                future.result()

        print(f"time taken to put evaluated code into islands:{time.time() - end_time}")
        '''
        start_time = time.time()

        #delete some islands 
        if len(database.keys()) > 10: 
            new_database_indices = heapq.nlargest(10, database, key=lambda database_key: database[database_key].best_score)
            new_database = { k: database[new_database_indices[k]] for k in range(len(new_database_indices))} 
            database = new_database


        '''
        results =  Parallel(n_jobs=-1, backend="multiprocessing")(delayed(wasserstein_with_weights)(np.array([np.array(t) for t in database[i].best_behaviour.keys()]), np.array([np.array(t) for t in database[j].best_behaviour.keys()]), np.array([np.array(t) for t in database[i].best_behaviour.values()]), np.array([np.array(t) for t in database[j].best_behaviour.values()]))
                                                                  for i in database.keys()
                                                                  for j in database.keys())
        '''

        #wasserstein_matrix = np.array(results).reshape(len(database.keys()), -1)

        #print(f"time taken to draw the wasserstein matrix: {time.time() - start_time}")
        wasserstein_matrix = np.zeros((len(database.keys()), len(database.keys())))

        for i in database.keys(): 

            plt.figure(figsize=(8,4))
            cand_behaviour = database[i].best_behaviour
            labels = [str(key) for key in cand_behaviour.keys()]
            plt.bar(labels, cand_behaviour.values())
            plt.xlabel("Actions")
            plt.ylabel("P(a)")
            plt.title(f"action distribution for island-{i}")
            plt.xticks(rotation=90)
            plt.savefig("action_plot.png")
            wandb.log({f"action_distribution_Island_{i}":wandb.Image("action_plot.png")})
            for j in database.keys():

                wasserstein_matrix[i][j] = wasserstein_with_weights(np.array([np.array(t) for t in database[i].best_behaviour.keys()]), np.array([np.array(t) for t in database[j].best_behaviour.keys()]), np.array([np.array(t) for t in database[i].best_behaviour.values()]), np.array([np.array(t) for t in database[j].best_behaviour.values()]))

        plt.figure(figsize=(8, 6))
        sns.heatmap(wasserstein_matrix, annot=True, cmap="viridis", xticklabels=[f"D{i}" for i in range(len(database.keys()))], yticklabels=[f"D{i}" for i in range(len(database.keys()))])
        plt.title("Pairwise Wasserstein Distance")

        wandb.log({"wasserstein_distance_between_island_(best)_behaviours":wandb.Image(plt)})



        '''

        with torch.no_grad():
            code_embedding = embedding_model.encode(agent)  
        embedding = [t.median_embedding for t in database.values()]
        with torch.no_grad():
            similarity = embedding_model.similarity(code_embedding, embedding)
        print(f"similarity between agent's codes: {similarity}")

        distance_from_islands = (1 - similarity).clamp(min=0)

        if float(min(distance_from_islands)[0]) > 0.4: 
            database[len(database.keys())+1] = Island(code=[agent], score=[score])
            #database[len(islands)+1] = Island(codes=[code], scores=[score]) 
            wandb.log({f'best_score_island_{len(database.keys())+1}': score})
            wandb.log({f'program_{len(database.keys())+1}':wandb.Html(f'score: {score}<pre>{agent}</pre>')})

        elif database[int(np.argmin(distance_from_islands))].best_score < score: 
            database[int(np.argmin(distance_from_islands))].add_code(agent, score)
            wandb.log({f'best_score_island_{int(np.argmin(distance_from_islands))}': score})
            wandb.log({f'program_{int(np.argmin(distance_from_islands))}':wandb.Html(f'score: {score}<pre>{agent}</pre>')})
        '''

wandb.finish()







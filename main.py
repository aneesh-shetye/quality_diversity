import random 
import json
import logging
import os
import pathlib
import pickle
import time

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
from evaluator import find_agent, extract_agent, run_agent

import wandb 

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--output_path', default='data/', type= click.Path(file_okay=False), help='path for logs and data') 
parser.add_argument('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')

embedding_model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()


model_id = "meta-llama/Llama-3.3-70B-Instruct"
#model_id = "upiter/TinyCodeLM-400M"
  
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
  )

#initial_template: str 
initial_code_hopper = """
#Build an agent to solve an environment.  
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

#INITIALIZE  DATABASE with 0 islands  
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
sampler = pipeline

#INITIALIZE EVALUATOR 
#evaluator: 
#gets text that is corresponds to our function 
#makes it into a runnable program
#runs the function in the environment in a sandbox say 10 times  
#returns a score  

#INITIALIZE A DISTANCE FUNCTION e.g. DBSCAN 

time_step = 0 
def main(): 
    for t in range(int(1e4)): 
        print("Entered the for loop")
        #code = sampler.generate(get_prompt_from_existing_islands) 
        print(f'databse.keys: {database.keys()}')
        if len(database.keys()) < 10: 
            prompt = initial_code_hopper
            prompts = [initial_code_hopper]
        else: 
            prompt = database[random.sample(list(database.keys()), 1)[0]].get_prompt()
            prompts = [i.get_prompt for i in database.values()]

        print('prompting sampler')
        print(f'prompt: {prompt}') 
        code = sampler(prompt, max_new_tokens=4096)  
        code = code[0]['generated_text']
        print(f'sample in sampler: {code}')

        agent = extract_agent(find_agent(code))
        print(f'agent in sampler: {agent}')
        score = run_agent(agent) 
        print(f'score in sampler: {score}')
        
        if score is None: 
            continue

        if len(database.keys()) == 0:
            database[0] = Island(code=[agent], score=[score])
            wandb.log({f'score_island_0': score})
            wandb.log({f'program_0':wandb.Html(f'score: {score}<pre>{agent}</pre>')})

            continue

        elif len(database.keys()) < 10: 
            database[len(database.keys())] = Island(code=[agent], score=[score])
            wandb.log({f'score_island_{len(database.keys())}': score})
            wandb.log({f'program_{len(database.keys())}':wandb.Html(f'score: {score}<pre>{agent}</pre>')})

            continue

        with torch.no_grad():
            code_embedding = embedding_model.encode(agent)  
        embedding = [t.median_embedding for t in database.values()]
        with torch.no_grad():
            similarity = embedding_model.similarity(code_embedding, embedding)
        print(f"similarity between agent's codes: {similarity}")

        distance_from_islands = (1 - similarity).clamp(min=0)



        if float(min(distance_from_islands)[0]) > 0.1: 
            database[len(database.keys())+1] = Island(code=[agent], score=[score])
            #database[len(islands)+1] = Island(codes=[code], scores=[score]) 
            wandb.log({f'best_score_island_{len(database.keys())+1}': score})
            wandb.log({f'program_{len(database.keys())+1}':wandb.Html(f'score: {score}<pre>{agent}</pre>')})

        elif database[int(np.argmin(distance_from_islands))].best_score < score: 
            database[int(np.argmin(distance_from_islands))].add_code(agent, score)
            wandb.log({f'best_score_island_{int(np.argmin(distance_from_islands))}': score})
            wandb.log({f'program_{int(np.argmin(distance_from_islands))}':wandb.Html(f'score: {score}<pre>{agent}</pre>')})

        time_step += 1 

        #delete some islands 
        if time_step%1e3 == 0: 
            if len(databse.keys()) > 10: 
                database = heapq.nlargest(10, database, key=lambda island: island.best_score)




if __name__ == "__main__": 
    wandb.init(project='quality_diversity', 
         settings=wandb.Settings(start_method="fork"), 
         reinit=True)
    main()
    wandb.finish()

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

from collections import defaultdict

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
from tools import load_files_to_dict

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

parser.add_argument('--island_directory', default="islands/", help='Directory with intial code for all the islands')

args = parser.parse_args()

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

prefix_prompt = """
Form a function named `agent_vX` (where X is an integer not used so far in other function names) to solve gym's hopper environment. The  environment has a hopper - a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts. 

The input to this agent  is the current state of gym's inverted pendulum v-4 environment. 
The function you are designing would take in a "state" argument which is a tuple consisting of 2 elements- qpos nad qvel. 
qpos is a 5 dimensional vector where:  
qpos[0] gives the z-coordinate of the torso (height of the hopper,
qpos[1] gives the angle of the torso, 
qpos[2] gives the angle of the thigh joint, 
qpos[3] gives the angle of the foot joint and, 
qpos[4] gives the velocity of the x-coordinate (height) of the torso

qvel is a 6 dimensional vector where: 
qvel[0] gives the velocity of the x-coordinate of the torso 
qvel[1] gives the velocity of the z-coordinate of the torso 
qvel[2] gives the angular velocity of the angle of the torso 
qvel[3] gives the angular velocity of the thigh hinge 
qvel [4] gives the angular velocity of the leg hinge 
qvel[5] gives the angular velocity of the foot hinge 

Use the following set of functions to solve the gym hopper environemnt.
You could use any or none of the given functions.

"""

#INITIALIZE  DATABASE with 0 islands  

files_dict = load_files_to_dict(directory=args.island_directory)

time_step = 0 
if __name__ == "__main__": 

    for t in range(int(1e4)): 

        for i, c in zip(files_dict.keys(), files_dict.values()): 
            prompt = prefix_prompt + c 

            response = pipeline(prompt, max_new_tokens=4096)
            code = response[0]['generated_text']
            result = evaluate_agent(code, support_code=c)
            print("###########################")
            print(f'result:{result}')
            print("###########################")
            #result: [code, total_reward, normalized_distribution, quantized action]

            if result[1]: 

                path = os.path.join(args.island_directory, i) 
                with open(path, 'w') as f: 
                    f.write(result[0])

                wandb.log({f'score_island_{i}': result[1]})
                wandb.log({f'program_{i}':wandb.Html(f'<pre>{result[0]}</pre>')})



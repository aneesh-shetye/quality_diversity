import time

import re 
import ast
import gymnasium as gym 
import traceback
import numpy as np 
import torch 

#pattern = r"def agent_v\d+\([^)]*\)\s*(?:->\s*[^:]+)?:\n(?:    .+\n)*"

pattern = re.compile(r"def agent_v\d+\([^)]*\)\s*(?:->\s*[^:]+)?:\n(?:\s+.*(?:\n|$))*")

pattern = re.compile(r"(def agent_\w+\(.*?\)(?:\s*->\s*[^:]*?)?:\n(?:\s+.*?\n)*?\s+return.*?(?:\n|$))")
# Find all matching functions
METHOD_MATCHER = re.compile(r"def agent_v\d+\(.*?\) -> [a-zA-Z\[\]_,.\s]+:(?:\s*(?:[ \t]*(?!def|#).*?(?:\n|$)))+") 

def find_agent(generated_text: str): 
# Find all matching functions
  matches = METHOD_MATCHER.findall(generated_text)#, re.DOTALL)
  #print(f'matches in evaluator: {matches}')
  last_match = matches[-1] if matches else None
  if len(matches) > 3: 
    last_match = matches[-2] 
  return last_match

def extract_agent(generated_text: str): 
    # Parse the text into an AST
    try: 
      tree = ast.parse(generated_text)
    except: 
      return None
    extracted_functions = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("agent_"):
            # Convert function back to source code
            last_line = None
            for inner_node in ast.walk(node):
                if isinstance(inner_node, ast.Return):
                    last_line = inner_node.lineno

            if last_line:
                # Filter out everything after the last return statement
                function_lines = generated_text.splitlines()[node.lineno - 1:last_line]
                extracted_functions.append("\n".join(function_lines))

    return extracted_functions[-1] if len(extracted_functions)!=0 else None


def run_agent(code: str, env_name: str=None): 

  if code is None: 
    return None

  match_name = re.search(r"def (agent_v\d+)\(", code)
  agent_name = match_name.group(1) if match_name else None

  if not agent_name: 
    return None

  if code is None: 
    return None

  sandbox = {"__builtins__": __builtins__}
  #print(f'code in evaluator: {code}')
  try:
    exec(code,sandbox) 
  except: 
    return None

  if not env_name: 
    env = gym.make("Hopper-v4")

  else: 
    env = gym.make(env_name) 


  try:
    total_reward = 0
    observation, info = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        action = sandbox[agent_name](observation)
        #print(len(env.step(action)))
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward

    return int(total_reward)
  except Exception as e:
    print(f"Error: {traceback.format_exc()}")
    return None 



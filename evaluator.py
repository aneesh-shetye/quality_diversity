import random
import time
import re 
import ast
import gymnasium as gym 
import traceback
import numpy as np 
import torch 

from collections import Counter

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

def single_process_extractor(generated_text:str): 
  candidate = find_agent(generated_text) 
  if not candidate: 
    return None
  return extract_agent(candidate)


def run_agent(candidate_code: str, support_code: str,  env_name: str="Hopper-v4"): 

  if candidate_code is None: 
    return None, None, None, None

  code = single_process_extractor(candidate_code)
  code = support_code + candidate_code
  print("########################################") 
  print(f"code in evaluator: {code}")
  print("########################################") 

  if code is None: 
    return None, None, None, None

  match_name = re.search(r"def (agent_v\d+)\(", code)
  agent_name = match_name.group(1) if match_name else None

  if agent_name is None: 
    return None, None, None, None

  if code is None: 
    return None, None, None, None

  sandbox = {"__builtins__": __builtins__, "np": np, "random": random}
  #print(f'code in evaluator: {code}')
  try:
    exec(code,sandbox) 
  except: 
    return None, None, None, None

  env = gym.make(env_name) 


  try:
    total_total_reward = 0 
    total_reward = 0 
    total_actions_taken = []
    actions_taken = []
    '''
    for _ in range(5):
      total_reward = 0
      actions_taken = []
      observation, info = env.reset()
      done = False
      truncated = False
    '''
    observation, info = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        action = sandbox[agent_name](observation)
        #print(len(env.step(action)))
        total_actions_taken.append(list(action))
        actions_taken.append(list(action) + list(observation))
        total_actions_taken.append(list(action) + list(observation))
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward

    total_total_reward += total_reward
    #total_reward = total_total_reward//5
    env.close()

    if isinstance(env.action_space, gym.spaces.Box):
        quantized_actions = [tuple(np.round(np.array(a), 1)) for a in actions_taken]
    else:
        quantized_actions = actions_taken

    action_counts = Counter(quantized_actions)
    total_actions = sum(action_counts.values())
    normalized_distribution = {action: count / total_actions for action, count in action_counts.items()}

    #print(f'actions taken: {quantized_actions}')
    return code, int(total_reward), normalized_distribution, np.array(quantized_actions)

  except Exception as e:
    print(f"Error: {traceback.format_exc()}")
    return None, None, None, None 

def evaluate_agent(text: str, support_code: str):
    return run_agent(text, support_code, "Hopper-v4")

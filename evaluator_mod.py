import ast 
import re 
import gymnasium as gym 
import traceback 

import numpy as np
import random 

from collections import Counter


def evaluate_agent(generated_output): 
  code_match = re.search(r"```python\n(.*?)```", generated_output, re.DOTALL)
  print(code_match)

  if not code_match: 
    return None, None, None, None

  else: 
    generated_class_code = code_match.group(1) 

    parsed_code = ast.parse(generated_class_code) 

    last_agent_name = None 

    for node in ast.walk(parsed_code): 

      if isinstance(node, ast.FunctionDef) and node.name.startswith("agent_v"): 
        last_agent_name = node.name 

    if last_agent_name is None: 
      print("No agent_vX function found in the class definition")
      return None, None, None, None 

    sandbox = {"__builtins__": __builtins__, "np": np, "random": random}

    try: 
      exec(generated_class_code, sandbox) 
      agent_instance = sandbox['Agent']()

      env = gym.make("Hopper-v4")
      obs, info = env.reset() 
      total_reward = 0 
      actions_taken = []
      done = False
      truncated = False

      while not done and not truncated: 
        try: 
          action = getattr(agent_instance, last_agent_name)(obs) 
          if not isinstance(action, (tuple, list)): 
            action = [float(action)]*env.action_space.shape[0]
          actions_taken.append(action)
          obs, reward, done, truncated,  _ = env.step(action)
          total_reward += reward

        except Exception as e: 
          print(f"Error during env step: {e}")

          break

      env.close()

      if isinstance(env.action_space, gym.spaces.Box):
        quantized_actions = [tuple(np.round(np.array(a), 1)) for a in actions_taken]
      else:
        quantized_actions = actions_taken

      action_counts = Counter(quantized_actions)
      total_actions = sum(action_counts.values())
      normalized_distribution = {action: count / total_actions for action, count 
                                 in action_counts.items()}

      return code_match.group(1), int(total_reward), normalized_distribution, np.array(quantized_actions) 
    except Exception as e: 
      print(f"Error: {traceback.format_exc()}")
      return None, None, None, None 


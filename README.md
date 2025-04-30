# Quality Diversity with Agent History 

## Algorithm 
```python 

database = {island_id: code}

for island_id, code in database:
	prompt = env_description + code
	response = LLM(prompt) 
	result = evaluator(response) 

	if result: 
		database[island_id] <- response

```

## How to run the code? 
- simply run the `main.py` file using `torchrun`
- `prefix_prompt` within `main.py` is the environment description 
- `/island` directory has initializations of all islands. This is also where the results are stored. 

## Things that need mending 
- `evaluator.py`
- prompt tuning is required

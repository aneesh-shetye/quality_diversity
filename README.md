## Quality Diversity Algorithm for RL code discovery 

Motivated by Quality Diversity Algorithms [[more on this link](https://github.com/DanieleGravina/divergence-and-quality-diversity)] and funsearch [[link to blog](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)], this repository aims to solve RL environments by bootstraping LLMs onto a genetic algorithm. 

There are two branches to this repository. The main branch has a singular process for prompting an LLM, evaluating the generated code and saving the results. The batched branch as the name suggests uses batched processing. Choose wisely depending on your available resources 🙃

Here is the sketch of the architecture which this repository uses (An image is worth ...) 
<img width="687" alt="image" src="https://github.com/user-attachments/assets/29181b2e-30ba-4a85-8599-cfe01207f5d0" />

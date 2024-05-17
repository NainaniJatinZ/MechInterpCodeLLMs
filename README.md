# MechInterpCodeLLMs

A general outline for the structure of this repository is as follows:

- [circuits/codellama](circuits/codellama) contains our discovered circuits with head and neuron indices within a JSON file. The title of each file should indicate the kind of prompt that the circuit was run on, as well as the threshold hyperparameter for mean ablation.
- [cluster](cluster) contains the bash scripts and error logs from running our experiments on the Unity Cluster.
- [data](data) contains all of our datasets. As specified by the final report, many iterations of the datasets are uploaded, as many experiments had to be run to find the best prompt. The final datasets which were actually used to discover circuits were [instructed_trial4_NL](data/info_retrieval/instructed_trial4_NL.json) for the NL task and [instructed_trial3](data/info_retrieval/instructed_trial3.json) for the structured task. 
- [experiments](experiments) contains a lot of early experiments used to decide what the best course of action was for larger, more computationally expensive activation and attribution patching experiments. [AttributionPatching.ipynb](experiments/InfoRetrieval/AttributionPatching.ipynb) contains some early experiments which guided our final decisions.
- [plots](plots) contains all relevant plots found within the final report
- [src](src) contains [data_gen.py](src/data_gen.py) which was used to generate some prompts that were tested but ultimately not used in our final experiments, and [performanceTest.py](src/performanceTest.py) which was used to run baselines on how well LLaMa and CodeLLaMa performed on the NL and Structured tasks.
- [transformer_lens](transformer_lens) is a custom transformer_lens package which allowed us many additional functionalities which were crucial to running activation and attribution patching experiments. In any attempts to replicate these experiments, be sure that this transformer_lens package is the one which is imported and used
- [utils_patch](utils_patch) contains some small helper files
- [seqAttPatching.py](seqAttPatching.py) is the main file that is run on the Unity cluster to get our results. These results are later visualized using some of the techniques in the experiments folder.

import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('experiments/old_trials/finetuneMI/data/objects.csv')

def generate_prompts(num_prompts=40):
    # operators = ['+', '-', '*', '**', '%', '<', '>', '<=', '>=', '=='] #['+', '-', '*', '**', '%']
    functions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    prompts = []
    
    for _ in range(num_prompts):
        sampled_df = df.sample(n=4)
        selected_operators = list(sampled_df.object_name)  #random.sample(operators, 4) #
        selected_functions = random.sample(functions, 4)
        prompt = ""
        # prompt = "#For this set of variables, return the variable name containing the specified item."
        
        examples = []
        op_to_func_mapping = {}

        for op, func in zip(selected_operators, selected_functions):
            prompt += f" {func} = '{op}' "
            # prompt += f"    return a {op} b\n"
            # prompt += f"def {func}(a, b):\n"
            # prompt += f"    return a {op} b\n"
            examples.append([func,op])
            op_to_func_mapping[op] = func
        
        for i in range(4):
            chosen_op = selected_operators[i] #random.sample(selected_operators, 1)[0]
            prompt2add = prompt+ f" The name of the key that has the value '{chosen_op}' is "
            # prompt2add = prompt+f"Question: What is the name of the variable containing '{chosen_op}'? Answer: "
            prompts.append({"prompt": prompt2add, "output": op_to_func_mapping[chosen_op]})
    
    print(prompts[0]['prompt'])
    print(prompts[0]['output'])
    with open("data/info_retrieval/instructed_trial2.json", "w") as f:
        json.dump(prompts, f, indent=4)

generate_prompts()
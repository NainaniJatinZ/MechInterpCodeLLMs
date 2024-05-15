import json
import re

# Function to transform the prompt using regex
def transform_prompt(original_prompt):
    # Extract key-value pairs from the original prompt
    pairs = re.findall(r"(\w) = '([^']+)'", original_prompt)
    
    # Extract the target item from the original prompt
    target_item = re.search(r"value '([^']+)'", original_prompt).group(1)
    
    # Create a dictionary from the pairs for easier lookup
    pairs_dict = {key: value for key, value in pairs}
    
    # Create the new format for the prompt
    new_prompt = " ".join([f"The {value} is in Box {key}." for key, value in pairs]) + f" What is the letter of the Box that the {target_item} is in? Box "
    
    return new_prompt

# Read the JSON file
with open('data\info_retrieval\instructed_trial2.json', 'r') as infile:
    data = json.load(infile)

# Process each prompt-output pair
new_data = []
for pair in data:
    original_prompt = pair['prompt']
    transformed_prompt = transform_prompt(original_prompt)
    new_pair = {"prompt": transformed_prompt, "output": pair['output']}
    new_data.append(new_pair)

# Write the transformed pairs to a new JSON file
with open('output.json', 'w') as outfile:
    json.dump(new_data, outfile, indent=4)

print("Transformation complete. Check the output.json file for results.")

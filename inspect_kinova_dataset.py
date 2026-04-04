#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds

# Load a small sample from the kinova dataset to inspect language instructions
dataset_name = "kinova_coke_pick_same_start"
data_dir = "/home/freddie/tensorflow_datasets"

# Create dataset builder
builder = tfds.builder(dataset_name, data_dir=data_dir)
ds = builder.as_dataset(split="train", shuffle_files=False)

print("Inspecting first 5 episodes for language instructions...")
print("-" * 50)

episode_count = 0
seen_instructions = set()

for episode in ds.take(5):
    episode_count += 1
    
    # Get the first step of the episode
    first_step = next(iter(episode['steps']))
    
    # Extract language instruction
    lang_instruction = first_step['language_instruction'].numpy().decode('utf-8')
    seen_instructions.add(lang_instruction)
    
    print(f"Episode {episode_count}:")
    print(f"  Language instruction: '{lang_instruction}'")
    print(f"  Episode length: {len(episode['steps'])}")
    print()

print(f"Unique language instructions found: {len(seen_instructions)}")
for i, instruction in enumerate(sorted(seen_instructions), 1):
    print(f"{i}. '{instruction}'")
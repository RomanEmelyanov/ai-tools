#!/usr/bin/env python3

# Local model location: ~/.cache/huggingface/hub/

from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

while(True):
	prompt = input("Prompt: ")
	messages = [ {"role": "system", "content": "You are a friendly chatbot."}, 
				 {"role": "user",   "content": prompt}, ]
	input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
	prompt = tokenizer.decode(input_ids)
	response = generate(model, tokenizer, prompt=prompt)
	print("Bot: ", response)

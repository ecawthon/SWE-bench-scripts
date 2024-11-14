import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import openai
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import requests
import json
import os
import difflib
import time
import traceback

begin_time = time.time()
# model_name = "Llama 3.1"
model_name = "gpt-4o"
timeout = 240
max_workers = 4

formatting = "Please output only the diff and no additional text. No context, "+\
            "no explanations, no markdown formatting. Just something I can feed "+\
            "into patch without errors. Your response must start with 'diff "+\
            "--git'."

def get_response(instance_id, prompt, model_name=model_name):
    start_time = time.time()
    patch = ""
    if model_name == "Llama 3.1":
        llm = Ollama(model="llama3.1:latest", request_timeout=timeout)
        messages = [
                    ChatMessage(
                        role="system",
                        content="You are an expert software engineer."
                    ),
                    ChatMessage(
                        role="user",
                        content="Create a diff to resolve this github issue. "+\
                                formatting +"\nGithub Issue:\n"+ prompt
                    )
                ]
        try:
            patch = llm.chat(messages).message.content
        except Exception as e:
            print(f"Request {instance_id} probably timed out: {e}")

    elif model_name == "gpt-4o":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        retries = 3
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Set it as an environment variable 'OPENAI_API_KEY'.")
        for attempt in range(retries):
            try:
                response = openai.chat.completions.create(
                        model=model_name,
                        messages =[
                            {
                                "role": "system",
                                "content": "You are an expert software engineer."

                            },
                            {
                                "role": "user",
                                "content": "Create a diff to resolve this github issue. "+\
                                        formatting +"\nGithub Issue:\n"+ prompt
                            }
                        ],
                        )
                patch = response.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                print(f"Rate limit exceeded {e}. Retrying in 10 seconds")
                time.sleep(10)
            except openai.OpenAIError as e:
                print(f"OpenAI API Error: {e}")
                break
    prediction = {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": patch
            }
    end_time = time.time()
    print(f"Instance {instance_id} completed in {end_time - start_time:.2f} seconds", flush=True)
    return prediction


dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

predictions = []
count = 0
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(get_response, instance['instance_id'],
                               instance['problem_statement']): instance for instance in dataset}
    for future in as_completed(futures):
        try:
            result = future.result(timeout=timeout)
### serial code commented out ###
# for task in dataset:
    # prompt = task['problem_statement']
    # result = get_response(task['instance_id'], prompt, model_name)
            predictions.append(result)
            count += 1
            print(count, "/", len(dataset))
        except Exception as e:
            print(f"Error processing instance {futures[future]['instance_id']}: {e}.")
            print(traceback.format_exc())

with open("predictions-"+ model_name+"."+time.strftime('%Y-%m-%d %H:%M:%S')+".jsonl", "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred)+"\n")
stop_time = time.time()
print(f"That took {stop_time - begin_time:.2f} seconds total.")

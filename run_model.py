import argparse
import sys
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA or OpenAI model queries with optional timeout and threading")
    parser.add_argument("model_name", help="Name of the model to use (Llama3.1 or gpt-4o)")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of threads to use (default = 4)")
    parser.add_argument("--timeout", type=int, default=240, help="Timeout (seconds) (default 240)")
    parser.add_argument("--output_file", type=str, default="predictions."+time.strftime('%Y-%m-%d %H:%M:%S')+".jsonl")
    if len(sys.argv) == 1:
        parser.print_usage()
        sys.exit()
    return parser.parse_args()

args = parse_args()
model_name = args.model_name
max_workers = args.max_workers
timeout = args.timeout
output_file = args.output_file

formatting = "Please output only the diff and no additional text. No context, "+\
            "no explanations, no markdown formatting. Just something I can feed "+\
            "into patch without errors. Your response must start with 'diff "+\
            "--git'."
role = "You are an expert software engineer."
prompt1 = "Create a diff to resolve this GitHub issue. "+ formatting +\
        "\nGitHub Issue:\n"

def get_response(instance_id, prompt, model_name=model_name):
    start_time = time.time()
    patch = ""
    if model_name == "Llama3.1":
        llm = Ollama(model="llama3.1:latest", request_timeout=timeout)
        messages = [
                    ChatMessage(
                        role="system",
                        content=role
                    ),
                    ChatMessage(
                        role="user",
                        content=prompt1 + prompt
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
                                "content": role

                            },
                            {
                                "role": "user",
                                "content": prompt1 + prompt
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

with open(output_file, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred)+"\n")
stop_time = time.time()
print(f"That took {stop_time - begin_time:.2f} seconds total.")

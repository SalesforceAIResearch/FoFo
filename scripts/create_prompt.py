# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
# Or, if you want to use later versions of OpenAI packages, uncomment line 9-12,
# uncomment line 27-37 and comment out line 15-25.
import os 
import csv
import json
import random
import openai
from tqdm import tqdm
# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

def gpt_call(sys_prompt, prompt):
    """ Get GPT4 response given the input prompt """
    ret = openai.ChatCompletion.create(
      model="gpt-4-1106-preview", # gpt-4
      messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    response = ret["choices"][0]['message']['content']
    return response

# def gpt_call(sys_prompt, prompt):
#     """ Get GPT4 response given the input prompt """
#     ret = client.chat.completions.create(
#       model="gpt-4-1106-preview",
#       messages=[
#             {"role": "system", "content": sys_prompt},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     response = ret.choices[0].message.content
#     return response

def generate_domain_specific_data_format(input_file_path, output_file_path):
    """Generates domain specific formats according to pre-defined domain, subdomains.

    Parameters
    ----------
    input_file_path: string
        Path of the json file that contains the domain, subdomain information for format generation.
        Each item in the json file should be a dict with "domain", "sub_domain_list" as keys.
        The default input file path is "../data/domain_subdomain_list.json". 
    output_file_path: string
        Path to save the output data. 
        Each item in the output data is dict like this: {"domain": domain, 
        "sub_domain": sub_domain, "response": response}. Among them, the "response" is the generated formats
        according to the domain and subdomain information.
    """
    sys_prompt = "You are a helpful agent."
    output_data = []
    # load input data on domains
    domain_data = json.load(open(input_file_path))
    for domain in domain_data:
        domain_name = domain["domain"]
        sub_domain_list = domain["sub_domain_list"][0].keys()
        for sub_domain_name in sub_domain_list:
            while True:
                try:
                    prompt = "Please give 5 human-understandable text data formats that a AI agent in the domain of " + domain_name + " -> " + sub_domain_name + \
                    ", would likely encounter as its required output formats during its interaction with humans. Note that only text data format should be provided. " + \
                    "The data formats should also be as domain-specific as possible. Generic formats such as txt, csv, json, xml, etc, shouldn't be included. "  + \
                    "An example of a piece of data of the specific format should be provided after the name of each format. " 
                    response = gpt_call(sys_prompt, prompt)
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``")
                    print(prompt)
                    print(response)
                    output_data.append({"domain": domain_name, "sub_domain": sub_domain_name, "response": response})
                    break
                except openai.APIConnectionError as e:
                    print("The server could not be reached")
                    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
                except openai.RateLimitError as e:
                    print("A 429 status code was received; we should back off a bit.")
                except openai.APIStatusError as e:
                    print("Another non-200-range status code was received")
                    print(e.status_code)
                    print(e.response)
    json.dump(output_data, open(output_file_path, 'w'), indent=4)


def generate_format_prompt(input_file_path, output_file_path):
    """Generates test prompts according to pre-defined domain, subdomain and format information.

    Parameters
    ----------
    input_file_path: string
        Path of the json file that contains the domain, subdomain and format information for prompt generation.
        Each item in the json file should be a dict with "domain", "sub_domain", and "format" as keys.
        The default input file path is "../data/format_list.json". 
        If you want to generate your own prompts, you can create your own format_list.json accordingly.
    output_file_path: string
        Path to save the output data. 
        Each item in the output data is dict like this: {"format": format, "domain": domain, 
        "sub_domain": sub_domain, "instruction": response}. Among them, the "instruction" is the generated test prompt
        according to the domain, subdomain and format information.
    """
    data_list = json.load(open(input_file_path))
    output_data = []
    sys_prompt = "You are a helpful agent."
    for i, data in enumerate(data_list):
        domain = data['domain']
        sub_domain = data['sub_domain']
        format = data['format']
        prompt = "Please write a prompt for an AI agent in the domain of " + domain  + " -> " + sub_domain + \
        ". The task of the prompt should be detailed and complicated content generation in the given domain. " + \
        "The task should require the output to to strictly adhere to a '" + format +  "' format with specific configurations. " + \
        "If the format name is not specific enough, please give concrete illustrations on the specific format requirements to follow. " + \
        "Please try your best to give detailed dummy context/data required in the prompt when necessary. If you can't give all necessary dummy data, please mention in the prompt that the AI agent is allowed to make up data required and improvise on ungiven details." + \
        "Your response should only contain the prompt or question, without any preliminary or concluding statements."
        while True:
            try:
                response = gpt_call(sys_prompt, prompt)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``")
                print(prompt)
                print(response)
                output_data.append({"format": format, "domain": domain, "sub_domain": sub_domain, "instruction": response})
                break
            except openai.APIConnectionError as e:
                print("The server could not be reached")
                print(e.__cause__)  # an underlying Exception, likely raised within httpx.
            except openai.RateLimitError as e:
                print("A 429 status code was received; we should back off a bit.")
            except openai.APIStatusError as e:
                print("Another non-200-range status code was received")
                print(e.status_code)
                print(e.response)
    json.dump(output_data, open(output_file_path, 'w'), indent=4)


def main():
    """
    # generate domain specific data formats
    input_file_path = '../data/agent_domain_sub_list.json'
    output_file_path = '../data/domain_specific_data_format.json'
    generate_domain_specific_data_format(input_file_path, output_file_path)
    """
    # generate test prompts with domain specific data formats
    input_file_path = '../data/format_list.json'
    output_file_path = '../data/test_prompts.json'
    generate_format_prompt(input_file_path, output_file_path)


if __name__ == "__main__":
    main()

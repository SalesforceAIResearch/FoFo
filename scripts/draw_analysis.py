# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
#from openai import OpenAI
import json
import csv
import random
from tqdm import tqdm
import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None


from collections import defaultdict

def save_json(data, file_path):
    with open(file_path, "w") as tf:
        json.dump(data, tf, indent=2)


def load_json(file_path):
    data = json.load(open(file_path))
    #print(data)
    return data


from collections import defaultdict

def domain_analysis(instruct_addr, annotation_addr):
    instruction_data_list = load_json(instruct_addr)
    # instruction-domain dict
    instruction_domain_dict = {}
    for instruction_data in instruction_data_list:
        instruction_domain_dict[instruction_data["instruction"]] = {"domain": instruction_data["domain"],
        "sub_domain": instruction_data["sub_domain"], "format_type": instruction_data["format_type"]}

    annotation_data_list = load_json(annotation_addr)
    format_acc_dict = defaultdict(list)
    domain_acc_dict = defaultdict(list)
    for test_case in annotation_data_list:
        instruction = test_case["instruction"]
        annotation = test_case["annotation"]
        format_type = instruction_domain_dict[instruction]["format_type"]
        domain = instruction_domain_dict[instruction]["domain"]
        format_acc_dict[format_type].append(annotation)
        domain_acc_dict[domain].append(annotation)

    print(domain_acc_dict)
    total_num_correct = 0
    total_num_samples = 0
    # present results
    for (format_type_name, annotation_list) in format_acc_dict.items():
        print(format_type_name)
        filtered_annotation_list = [x for x in annotation_list if x is not None]
        acc = sum(filtered_annotation_list)/len(filtered_annotation_list)
        print(acc)

    output_dict = {}
    for (domain_type_name, annotation_list) in domain_acc_dict.items():
        print(domain_type_name)
        #if (domain_type_name != "Customer Relationship Management (CRM)")&(domain_type_name != "Arts and Entertainment"):
        filtered_annotation_list = [x for x in annotation_list if x is not None]
        total_num_correct += sum(filtered_annotation_list)
        total_num_samples += len(filtered_annotation_list)
        acc = sum(filtered_annotation_list)/len(filtered_annotation_list)
        print(acc)
        output_dict[domain_type_name] = acc

    print(total_num_correct/total_num_samples)
    return output_dict


def format_analysis(instruct_addr, annotation_addr):
    data_format_set = {'json': 'JSON', 'csv': 'CSV', 'xml': 'XML', 'yaml':'YAML', 'markdown':'Markdown'}
    instruction_data_list = load_json(instruct_addr)
    # instruction-domain dict
    instruction_domain_dict = {}
    for instruction_data in instruction_data_list:
        instruction_domain_dict[instruction_data["instruction"]] = {"domain": instruction_data["domain"],
        "sub_domain": instruction_data["sub_domain"], "format": instruction_data["format"]}
        #print(instruction_domain_dict[instruction_data["instruction"]])
    annotation_data_list = load_json(annotation_addr)
    format_acc_dict = defaultdict(list)
    domain_acc_dict = defaultdict(list)
    for test_case in annotation_data_list:
        instruction = test_case["instruction"]
        annotation = test_case["annotation"]
        if instruction in instruction_domain_dict:
            format_type = instruction_domain_dict[instruction]["format"]
            domain = instruction_domain_dict[instruction]["domain"]
        #print(format_type)
        if format_type in data_format_set:
            format_acc_dict[data_format_set[format_type]].append(annotation)
        domain_acc_dict[domain].append(annotation)

    #print(format_acc_dict)
    total_num_correct = 0
    total_num_samples = 0
    # present results
    output_dict = {}
    for (format_type_name, annotation_list) in format_acc_dict.items():
        #print(format_type_name)
        filtered_annotation_list = [x for x in annotation_list if x is not None]
        acc = sum(filtered_annotation_list)/len(filtered_annotation_list)
        total_num_correct += sum(filtered_annotation_list)
        total_num_samples += len(filtered_annotation_list)
        #print(acc)
        output_dict[format_type_name] = acc

    #print(total_num_correct/total_num_samples)
    return output_dict

def draw(scores_all, category_list = None):
    #print(scores_all)
    modified_scores_all = []
    for score in scores_all:
        if score['category'] == 'Customer Relationship Management (CRM)':
            score['category'] = 'CRM'
        if score['category'] == 'Scientific Research and Development':
            score['category'] = 'Scientific R&D'
        if score['category'] == 'Commerce and Manufacturing':
            score['category'] = 'Manufacturing'
        if score['category'] == 'Arts and Entertainment':
            score['category'] = 'Entertainment'
        modified_scores_all.append(score)
    if category_list == None:
        category_list = []
        for score in modified_scores_all[:5]:
            category_list.append(score["category"])


    # target_models = ["Llama 7B Chat", "Mistral 7B V0.1"]#, "wlm13b", "zephyr"]
    target_models = ["WizardLM 13B V1.2", "Zephyr 7B Beta"]
    # target_models = ["Gemini Pro", "GPT-3.5"]
    scores_all = sorted(scores_all, key=lambda x: target_models.index(x["Models"]), reverse=True)

    df_score = pd.DataFrame(scores_all)

    print(df_score)
    df_score.iloc[0], df_score.iloc[2] =  df_score.iloc[2], df_score.iloc[0].copy()
    df_score.iloc[10], df_score.iloc[12] =  df_score.iloc[12], df_score.iloc[10].copy()
    print(df_score)

    print("----")
    # print(category_list)
    category_list = ['Technology and Software',  'Finance' ,'Healthcare', 'Manufacturing', 'CRM']
    fig = px.line_polar(df_score, r = 'score', theta = 'category', line_close = True, category_orders = {"category": category_list},
                        color = 'Models', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel)

    #fig.show()
    fig.update_layout(
        font=dict(
            size=20,  # Set the font size here
     ),
        showlegend=False,
       margin=dict(l=10, r=35, t=35, b=35)
    )

    fig.write_image("figs/fig2b.pdf", engine="kaleido")


def format_draw():
    scores_all = []
    your_root_addr = "../../"
    instruction_addr = f"{your_root_addr}/eval_benchmark/data/long_sequence_format_following_benchmark.json"
    llama2_7b_annotation_addr = f"{your_root_addr}/eval_benchmark/results/llama-2-7b/annotations.json"
    mistral_annotation_addr = f"{your_root_addr}/eval_benchmark/results/mistral_7b_instruct_v0.1/annotations.json"
    wlm13b_annotation_addr = f"{your_root_addr}/eval_benchmark/results/wizardlm-13b-v1.2/annotations.json"
    zephyr_annotation_addr = f"{your_root_addr}/eval_benchmark/results/zephyr-7b-beta/annotations.json"
    gemini_annotation_addr = f"{your_root_addr}/eval_benchmark/results/gemini-pro/annotations.json"
    chatgpt_annotation_addr = f"{your_root_addr}/eval_benchmark/results/chatgpt/annotations.json"

    llama = format_analysis(instruction_addr, llama2_7b_annotation_addr)
    mistral = format_analysis(instruction_addr, mistral_annotation_addr)
    gemini = format_analysis(instruction_addr, gemini_annotation_addr)
    chatgpt = format_analysis(instruction_addr, chatgpt_annotation_addr)

    # figure 3 (b):
    # for c in ['JSON', 'CSV', 'XML', 'YAML', 'Markdown']:
    #     scores_all.append({"Models": "Gemini Pro", "category": c, "score": gemini[c]})
    # for c in ['JSON', 'CSV', 'XML', 'YAML', 'Markdown']:
    #     scores_all.append({"Models": "GPT-3.5", "category": c, "score": chatgpt[c]})
    # print(scores_all)
    # draw(scores_all, category_list=['JSON', 'CSV', 'XML', 'YAML', 'Markdown'])

    # # figure 3(a)

    for c in ['JSON', 'CSV', 'XML', 'YAML', 'Markdown']:
        scores_all.append({"Models": "Mistral 7B V0.1", "category": c, "score": mistral[c]})
    for c in ['JSON', 'CSV', 'XML', 'YAML', 'Markdown']:
        scores_all.append({"Models": "Llama 7B Chat", "category": c, "score": llama[c]})
    print(scores_all)
    draw(scores_all, category_list=['JSON', 'CSV', 'XML', 'YAML', 'Markdown'])

def domain_draw():
    scores_all = []
    your_root_addr = "../../"
    instruction_addr = f"{your_root_addr}/eval_benchmark/data/long_sequence_format_following_benchmark.json"
    llama2_7b_annotation_addr = f"{your_root_addr}/eval_benchmark/results/llama-2-7b/annotations.json"
    mistral_annotation_addr = f"{your_root_addr}/eval_benchmark/results/mistral_7b_instruct_v0.1/annotations.json"
    wlm13b_annotation_addr = f"{your_root_addr}/eval_benchmark/results/wizardlm-13b-v1.2/annotations.json"
    zephyr_annotation_addr = f"{your_root_addr}/eval_benchmark/results/zephyr-7b-beta/annotations.json"
    gemini_annotation_addr = f"{your_root_addr}/eval_benchmark/results/gemini-pro/annotations.json"
    chatgpt_annotation_addr = f"{your_root_addr}/eval_benchmark/results/chatgpt/annotations.json"

    llama = domain_analysis(instruction_addr, llama2_7b_annotation_addr)
    mistral = domain_analysis(instruction_addr, mistral_annotation_addr)
    wlm13b = domain_analysis(instruction_addr, wlm13b_annotation_addr)
    zephyr = domain_analysis(instruction_addr, zephyr_annotation_addr)

    # figure 2 (a):
    # for (c,a) in llama.items():
    #     scores_all.append({"Models": "Llama 7B Chat", "category": c, "score": a})
    # for (c,a) in mistral.items():
    #     scores_all.append({"Models": "Mistral 7B V0.1", "category": c, "score": a})
    # draw(scores_all)


    # figure 2(b)

    for (c,a) in wlm13b.items():
        scores_all.append({"Models": "WizardLM 13B V1.2", "category": c, "score": a})
    for (c,a) in zephyr.items():
        scores_all.append({"Models": "Zephyr 7B Beta", "category": c, "score": a})
    draw(scores_all)


def plot_legends(name_and_colors={}, legend_name='legend.pdf'):
    import numpy as np
    from matplotlib import pyplot as plt
    x = np.linspace(1, 100, 1000)
    y = np.log(x)
    y1 = np.sin(x)
    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot")
    ax = fig.add_subplot(111)
    lines = []
    names = []
    for key in name_and_colors:
        line, = ax.plot(x, y, c=name_and_colors[key], lw=2, linestyle="-", marker='.')
        lines.append(line)
        names.append(key)

    legendFig.legend(lines, names, loc='center', frameon=False, ncol=len(lines))
    legendFig.savefig(legend_name, bbox_inches='tight')

if __name__ == "__main__":

    # figure 2:
    domain_draw()

    # figure 3:
    # format_draw()
    name_and_colors = {
        "Mistral 7B V0.1": "#66C5CC",
        "Llama 7B Chat": "#F6CF71"

    }
    legend_name = 'figs/legend_2a.pdf'

    # name_and_colors = {
    #     "Zephyr 7B Beta": "#66C5CC",
    #     "WizardLM 13B V1.2": "#F6CF71"

    # }
    # legend_name = 'legend_2b.pdf'

    # name_and_colors = {
    #     "Mistral 7B V0.1": "#66C5CC",
    #     "Llama 7B Chat": "#F6CF71"
    # }
    # legend_name = 'legend_3a.pdf'


    # name_and_colors = {
    #     "GPT-3.5": "#66C5CC",
    #     "Gemini Pro": "#F6CF71"
    # }
    # legend_name = 'legend_3b.pdf'


    plot_legends(name_and_colors, legend_name)

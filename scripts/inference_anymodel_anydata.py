import sys, os, json
import transformers
import torch
import time
import pdb
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

PROMPT_ALPACA = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
)
PROMPT_VICUNA = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: {instruction} "
        "ASSISTANT:"
)
PROMPT_WIZARDLM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: {instruction} "
    "ASSISTANT:"
)
PROMPT_ZEPHYR = (
    "<|system|>\n You are a helpful agent. </s>\n "
    "<|user|> \n {instruction}</s>\n "
    "<|assistant|>"
)

PROMPT_OPENCHAT = (
    "GPT4 Correct User: Hello<|end_of_turn|>"
    "GPT4 Correct Assistant: Hi<|end_of_turn|>"
    "GPT4 Correct User: {instruction}<|end_of_turn|>"
    "GPT4 Correct Assistant:"
)



class Generator(object):
    """Generator class to generate responses given prompts and model configs.

    Parameters
    ----------
    args : dict
        Arguments that will be passed saved in the Generator for controlling 
        generation. Details of args are in parse_args().
    """
    def __init__(self, args) -> None:
        self.max_seq_length = args.max_seq_length
        self.eos_token_id = 0
        self.args = args
        self.model_name = args.output_file_path.split("/")[-1].split(".json")[0].replace("output_", "ft_")
        self.prompt_style = args.prompt_style

    def save_json(self, data, file_path):
        print(f"Saving file {file_path} ...")
        with open(file_path, "w") as tf:
            json.dump(data, tf, indent=2)


    def generate(self):
        """Function to setup the model and generate responses given test prompts.
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            model_max_length=self.max_seq_length
        )
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model=None
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            torch_dtype=torch.bfloat16, 
            #attn_implementation="flash_attention_2"
        ).cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("model loaded successfully!")

        # load dataset
        dataset = load_dataset('json', data_files=self.args.input_file_path)['train']
        output_data = []

        progress_bar = tqdm(range(len(dataset)))
        with open(self.args.output_file_path, 'w') as outfile:
            with torch.no_grad():
                for batch_idx, batch in (enumerate(dataset)): 

                    if self.prompt_style == "alpaca":
                        input_seq = [PROMPT_ALPACA.format(instruction=batch["instruction"])]

                    elif self.prompt_style == "wizardlm" or self.prompt_style == "vicuna":
                        input_seq = [PROMPT_VICUNA.format(instruction=batch["instruction"])]
                    elif self.prompt_style == "mistral":
                        input_seq = ["[INST] {instruction} [/INST]".format(instruction=batch["instruction"])]
                    elif self.prompt_style == "zephyr":
                        input_seq = [PROMPT_ZEPHYR.format(instruction=batch["instruction"])]
                    elif self.prompt_style == "openchat":
                        input_seq = [PROMPT_OPENCHAT.format(instruction=batch["instruction"])]
                        self.eos_token_id = tokenizer.convert_tokens_to_ids("<|end_of_turn|>")
                    else:
                        input_seq = [batch["instruction"]]
                    inputs = tokenizer(input_seq, 
                                        return_tensors='pt', padding=False, 
                                        truncation=False,
                                        )
                    inputs = inputs.to(device) 
                    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
                    output = model.generate(input_ids=input_ids, 
                                            max_new_tokens=self.max_seq_length, 
                                            do_sample=True, 
                                            temperature=0.7,
                                            top_p=1,
                                            # eos_token_id=tokenizer.eos_token_id
                                            )
                    # Convert output to text and update the dataset
                    output_text = tokenizer.batch_decode(output, skip_special_tokens=True, eos_token_id=tokenizer.eos_token_id)
                    if self.prompt_style == "zephyr":
                        extract_output = output_text[0].split("<|assistant|>")[-1]
                    else:
                        extract_output = output_text[0].split(input_seq[0])[-1]
                    output_data.append({"instruction": batch["instruction"], "output": extract_output, "generator": self.args.model_name_or_path})
                    print("=======================================")
                    print(output_data[-1]["instruction"])
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print(output_data[-1]["output"])

                    json.dump(output_data[-1], outfile)
                    outfile.write('\n')
                    progress_bar.update(1)

        self.save_json(output_data, self.args.output_file_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model.")
    parser.add_argument(
        "--input_file_path",
        type=str,
        default="./data/fofo_test_prompts.json",
        help="The name of the test file.",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="./result/output_alapaca.json",
        help="The path to save the output file with both instructions and responses.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="The path to model.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default= "alpaca",
        help="The name of the model for deciding the prompt style",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=5120,
        help="",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    gen = Generator(args)
    gen.generate()

if __name__ == '__main__':
    main()

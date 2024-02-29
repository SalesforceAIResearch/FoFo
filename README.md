## Setup Enviroment
```
conda create --prefix ./envs/ python=3.10
conda init
conda activate envs
cd scripts
pip install -r requirements.txt
cd ..
cd alpaca_eval
pip install -e .[all]
pip install openai==0.27.0
```

## Setup Openai Account
```
export OPENAI_API_KEY=<your_api_key>
export OPENAI_ORGANIZATION_IDS=<your_organization_id> # Optional; if not set, this will be your default org id.
```

## Model Evaluation
```
data_path='data/fofo_test_prompts.json'
output_path='results'
```

### 1. Get model outputs
```
CUDA_VISIBLE_DEVICES='0' python scripts/inference_anymodel_anydata.py --input_file_path $data_path --output_file_path $output_path/wizardlm-13b-v1.2/model_outputs.json --model_name_or_path WizardLM/WizardLM-13B-V1.2 --prompt_style wizardlm --max_seq_length 5120```
```
### 2. Evaluate models' performance based on the outputs
```
alpaca_eval --annotators_config gpt4_format_correctness --model_outputs $output_path/chatgpt/reference_outputs.json --output_path $output_path/wizardlm-13b-v1.2/
```

# Script for setting up the environment and installing dependencies

import os

def install_dependencies():
    os.system('pip install unsloth')
    os.system('pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git')
    # os.system('pip install -U xformers --index-url https://download.pytorch.org/whl/cu124') # in case of error triton: https://github.com/invoke-ai/InvokeAI/issues/2611#issuecomment-2508871772
    # os.system('pip install unsloth_zoo')
    # os.system('pip install transformers')
    # os.system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124') # check you cuda version
    # os.system('pip install trl')
    # os.system('pip install peft')
    # os.system('pip install hf_transfer')
    # os.system('pip install databases')
    # os.system('pip install -r requirements.txt')

if __name__ == "__main__":
    install_dependencies()

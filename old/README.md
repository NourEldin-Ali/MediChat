# MediChat: An AI-Powered Medical Consultation Assistant

## Project Overview
We're building an intelligent medical consultation chatbot by specializing Llama3.1:8B for healthcare conversations. This project bridges the gap between advanced language models and practical medical assistance.

### Docker install
  ` docker build -t medichat .

### Install dependencies & set environment variables
* **Clone** the project in your preferred IDE (e.g. VScode)
* **Prepare the environment**:
    1. Create a virtual environment for dependencies called `env` using the following command: 
        ```sh 
          python -m venv env
        ```
    2. Activate the created `env` by running:
        * **Windows**: 
        ```sh 
          env\Scripts\activate.bat
        ```
        * **Unix/MacOS**:
        ```sh
          env/bin/activate
        ```
    3. Install the required libraries listed in `requirements.txt` by running:
        ```sh
          pip install -r requirements.txt
        ```
* **Set Up environment variables**:  
   - Open the file named `.env-example`.
   - Replace the placeholder `your-huggingface-token` with your actual Hugging face token.
   - Save and close the file.
   - Rename the file to `.env`.

## File structure
- app: contains the files used to execute streamlit and access the models and inteact with the user.
- data: contains the dataset used for fine-tune the mode
- models: to save the checkpoints after each epoc
- notebooks: 
  - EDA.ipynb: the notebook to load the initial dataset and select 1000 row
  - model-dev.ipynb: the notebook to load model and start fine-tune
- results: 
  - /init: contains results for the inital model
  - /fine-tuned: contains results from the fine_tuned model
- src: contains the code that help us in loading the model and and data

## Training model
To train the model you can open the `model-dev.ipynb` notebook, then install the requirement, and follow the step of the training.
Within the same notebook, you can find in the same notebook BLUE metric to evaluate the overlap between the generated text and ground truth text.
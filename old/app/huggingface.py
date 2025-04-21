import os
from dotenv import load_dotenv
from huggingface_hub import login

def login_huggingface(dotenv_path = '.env'):
    # Specify the path to your .env file
    

    # Load the .env file
    load_dotenv(dotenv_path)

    # Replace 'your_token_here' with your Hugging Face token
    huggingface_token = os.getenv('HUGGING_FACE_TOKEN')
    load_dotenv()
    # print(huggingface_token)
    # Log in to Hugging Face
    login(huggingface_token)

    print("Logged in to Hugging Face successfully!")
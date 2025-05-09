import streamlit as st
import os
import sys 


# Get the current working directory
current_directory = os.getcwd()
# print(current_directory)

# # Navigate to the parent directory
# project_root = os.path.abspath(os.path.join(current_directory, ".."))
project_root = os.path.abspath(current_directory)
sys.path.append(project_root)

from src.models.model import get_response
from src.models.model import load_model
from huggingface import login_huggingface
login_huggingface(dotenv_path=current_directory + '/.env')

# App title
st.set_page_config(page_title="🦙💬 Medical Chatbot")
llm_model = None
tokenizer = None
# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Medical Chatbot')
    
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama3.1-8B', 'FineTuned-Llama3.1'], key='selected_model')
    if selected_model == 'Llama3.1-8B':
        llm = "Llama3.1-8B"
        llm_model, tokenizer = load_model("unsloth/Meta-Llama-3.1-8B")
    else:
        llm = "medichat_model" #local path to fine-tuned model
        llm_model, tokenizer = load_model("NourEldin-Ali/medichat_model")
    
    # temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.01, step=0.01)
    # top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    # max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=4096, step=8)
    
    st.markdown('📖 Learn how to fine tune Llama in this [repository]()!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    # print("preparing to generate response")
    # inputs = tokenizer(f"{string_dialogue} {prompt_input} Assistant: ", return_tensors="pt")
    # outputs = llm_model.generate(inputs["input_ids"], temperature=temperature, top_p=top_p, max_length=max_length)
    # print("generate response")

    generated_text = get_response(llm_model, tokenizer, prompt_input)
    return generated_text

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
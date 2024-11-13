import ollama
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



llm=Ollama(model="llama3.2:1b", temperature=1,request_timeout=180.0)
Settings.llm=llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")



#openai.api_key = st.secrets.openai_key
st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about de energía comprimida"}
    ]

##Cargar datos 
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Cargadno e indexando los archivos , esto pued tardarar varios minutos !."):
        reader = SimpleDirectoryReader(input_files=["carpeta_archivos/Aire_comprimido.pdf"], recursive=True)
        docs = reader.load_data()
        #service_context = ServiceContext.from_defaults(llm=llm, temperature=0.5, system_prompt="You are an expert on the compressed energy ")                                                
        index = VectorStoreIndex.from_documents(docs, settings=Settings)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history




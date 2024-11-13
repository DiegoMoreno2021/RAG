#Primero importamos las librerias
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.tools import Tool

# Cargamos el api_key de open AI
with open('C:/ProyectoChatOpenAI/Scripts/Api_key.txt') as f:
    api_key = f.read().strip()

# Ahora se configura el modelo con temperatura cero para que no alucine
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)

# Pasamos ahora a Configurar memoria y base de datos vectorial
memory = ConversationBufferMemory(memory_key="chat_history")  # Memoria con clave "chat_history"
funcion_embedding = OpenAIEmbeddings(openai_api_key=api_key)
persist_path = "C:/ProyectoChatOpenAI/RAG/Vectorial_db"
vector_store_connection = SKLearnVectorStore(embedding=funcion_embedding, persist_path=persist_path, serializer="parquet")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vector_store_connection.as_retriever())

# Ahora se crea una herramienta de consulta interna y envolverla en un objeto Tool
def consulta_interna_func(text: str) -> str:
    '''Retorna respuestas sobre la historia de Colombia.'''
    compressed_docs = compression_retriever.invoke(text)
    if compressed_docs:
        return compressed_docs[0].page_content
    else:
        return "No se encontraron resultados relevantes para su consulta."

consulta_interna = Tool(
    func=consulta_interna_func,
    name="ConsultaInterna",
    description="Herramienta que consulta información sobre la historia de Colombia en una base de datos vectorial."
)

# Cargamos las herramientas adicionales
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
tools.append(consulta_interna)

# Cambiamos el tipo de agente a ZERO_SHOT_REACT_DESCRIPTION para mayor flexibilidad
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Cambiado a tipo que acepta respuestas más flexibles
    memory=memory,
    verbose=True
)

# Pasamos a la configuración Streamlit
st.title("Asistente de Historia de Colombia")
st.write("Pregunta sobre la historia de Colombia y obtén respuestas de la base de datos.")

# Capturamos de la pregunta del usuario
user_input = st.text_input("Introduce tu pregunta:", "")

# Aqui se procesa la pregunta si hay entrada del usuario
if user_input:
    # Invocarmos el agente con la pregunta y manejar errores de análisis de salida
    try:
        response = agent.invoke(user_input)
        st.write("**Respuesta:**", response)
    except ValueError as e:
        st.write("Error al procesar la respuesta del agente:", e)














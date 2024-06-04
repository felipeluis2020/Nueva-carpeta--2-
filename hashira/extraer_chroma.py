from flask import Flask, request, render_template
import openai
import os
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import RetrievalQA


# Configuración de OpenAI
openai.api_type = "azure"
openai.api_base = "https://dascd2024.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
openai.api_key = "fe15959cc2e9499c9c8fa35413abab4a"
chat_model = "ChatDASCD"

# Directorio donde están los archivos PDF
pdf_directory = r'C:\Users\lbravo\CHAT_MY\Nueva carpeta (2)\documentos'

# Cargar documentos desde el directorio
ml_papers = []

for filename in os.listdir(pdf_directory):
    if filename.endswith('.pdf'):
        filepath = os.path.join(pdf_directory, filename)
        loader = PyPDFLoader(filepath)
        data = loader.load()
        ml_papers.extend(data)
        print(f'Cargado {filename}')

# Imprimir información básica sobre los documentos cargados
print('Contenido de ml_papers:')
print(type(ml_papers), len(ml_papers))

# Dividir los documentos en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8600,
    chunk_overlap=4300,
    length_function=len
)

chunks = text_splitter.split_documents(ml_papers)
print('Número de fragmentos:', len(chunks))

# Crear embeddings e indexar en una base de datos vectorial
embeddings = OpenAIEmbeddings(
    deployment_id="Embedding",
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base
)

#Guardar en la base de datos vectorial ( 1. Nombre de la base de datos 2. iniciar el vectorstore  3. configurar el vectorestore 4. agregar la persistencia)

NOMBRE_INDICE_CHROMA="persistencia"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=NOMBRE_INDICE_CHROMA
)

vectorstore.persist()
vectorstore = Chroma(
    persist_directory=NOMBRE_INDICE_CHROMA,
    embedding_function=embeddings
)
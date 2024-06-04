from flask import Flask, request, render_template
import openai
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import requests

# 0.27 -- version openai

# Configuración de OpenAI
openai.api_type = "azure"
openai.api_base = "https://dascd2024.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
openai.api_key = "fe15959cc2e9499c9c8fa35413abab4a"
chat_model = "ChatDASCD"


app = Flask(__name__)

# Descargar documentos
urls = [
    'https://sideap.serviciocivil.gov.co/sideap/sideapdoc/Manuales/E-GCO-IN-005%20INSTRUCTIVO_PARA_EL_DILIGENCIAMIENTO_DEL_FOR_DE_BYR_SIDEAP_%20V6.pdf',
    'https://www.serviciocivil.gov.co/sites/default/files/2023-04/M_ITHD_IN_006%20INSTRUCTIVO_PARA_EL_DILIGENCIAMIENTO_DE_LA_HOJA_DE_VIDA_SIDEAP%20.pdf',
]

ml_papers = []

for i, url in enumerate(urls):
    response = requests.get(url)
    filename = f'paper{i+1}.pdf'
    with open(filename, 'wb') as f:
        f.write(response.content)
        print(f'Descargado {filename}')

        loader = PyPDFLoader(filename)
        data = loader.load()
        ml_papers.extend(data)

# Imprimir información básica sobre los documentos cargados
print('Contenido de ml_papers:')
print(type(ml_papers), len(ml_papers))

# Dividir los documentos en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
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

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Configurar el modelo de chat y la cadena de consulta
chat = AzureChatOpenAI(
    deployment_name=chat_model,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)


@app.route('/', methods=['GET', 'POST'])
def index():
    response = ""
    if request.method == 'POST':
        query = request.form['query']
        response = qa_chain.run(query)
        print(f"Respuesta: {response}\n")
    return render_template('index.html', response=response)
    

if __name__ == '__main__':
    app.run(port=5000)
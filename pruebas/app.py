from flask import Flask, request, render_template
import openai
import os
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import requests
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate
from utils import DocsJSONLLoader, get_file_path, get_openai_api_key, get_query_from_user


# Configuración de OpenAI
openai.api_type = "azure"
openai.api_base = "https://dascd2024.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
openai.api_key = "fe15959cc2e9499c9c8fa35413abab4a"
chat_model = "ChatDASCD"

# se activa la consola de visualización
console = Console()

#Acontinuación se le dice si es necesario hacer el chromadb o no (false)
recreate_chroma_db = False

#Se asigna el tipo de chat, en este caso es un chat de memoria 
chat_type = "memory_chat"

# Directorio donde están los archivos PDF o Jsonl
pdf_directory = r'C:\Users\lbravo\CHAT_MY\Nueva carpeta (2)\documentos'

#Se iniciar la definición o función de lectura de documentos

def load_documents(pdf_directory: str):
    #Buscae un archivo jsonl
    loader = DocsJSONLLoader(pdf_directory)
    # La información encontrada se guarda en la variable data
    data = loader.load()
    # Se inicia la variable text_splitter, que está ligada a la herramienta RecursiveCharacterTextSplitter, esta divide los textos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        length_function=len,
        chunk_overlap=160
    )
    # A continuación imprime la variable text_splitter
    print('Número de fragmentos:',text_splitter )
    # la variable text_splitter tiene una función o un objeto que se llama split_documents, que ayuda a dividir datos, en este
    # caso es la variable data (donde esta guardada la información), la configuracion del text_splitter se hace en el 
    # RecursiveCharacterTextSplitter
    return text_splitter.split_documents(data)

# se inicia la función o la definición para crear el chromadb, debe iniciar con embeddings, documents, path
def get_chroma_db(embeddings, documents, path):
# si recreate_chroma_db = a false, se crea un chromadb
    if recreate_chroma_db:
        # se muestra en consola la palabra RECREANDO CHROMA DB
        console.print("RECREANDO CHROMA DB")
        #retorna o ejecuta la funcion o el objeto, from_documents que quiere decir crear chroma de documentos desde...
        return Chroma.from_documents(
            documents=documents, embedding=embeddings, persist_directory=path
        )
    # Si recreate_chroma_db = true, carga la chromadb existente
    else:
        console.print("CARGANDO CHROMA EXISTENTE")
        return Chroma(persist_directory=path, embedding_function=embeddings)

# definición o función de Inicio de conversación, chat_type puede ser igual a "qa" o "memory_chat"
def run_conversation(vectorstore, chat_type, llm):

    console.print(
        "\n[blue]IA:[/blue] Hola soy CHAT PAO quieres preguntarme sobre SIDEAP en general?"
    )
    #Si es "qa" ejecutará el siguiente comando
    if chat_type == "qa":
        console.print(
            "\n[green]EstÃ¡s utilizando el chatbot en modo de preguntas y respuestas. Este chatbot genera respuestas basÃ¡ndose puramente en la consulta actual sin considerar el historial de la conversaciÃ³n.[/green]"
        )
    # Si es "memory_chat" ejecutará el siguiente comando
    elif chat_type == "memory_chat":
        console.print(
            "\n[green]Estás utilizando el chatbot en modo de memoria. Este chatbot genera respuestas basandose en el historial de la conversaciÃ³n y en la consulta actual.[/green]"
        )
    # A continuación se configura la variable retriver, si k es mayor, el proceso de respuesta puede ser más amplio
    # Quiere decir que busca mas vectores para dar una respuesta más inteligente.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    #Se inicia la variable chat_history, que en este caso es una matriz
    chat_history = []
    
    # Lo siguiente se ejecutará siempre que no se escriba "salir"
    while True:
        #Se muestra en la consola con color azul el 
        console.print("\n[blue]usuario SIDEAP:[/blue]")
        # Se inicia la variable query y esta nos lleva al archivo utils, con get_query_from_user()
        # esta función me trae el "input" del usuario
        query = get_query_from_user()
        # Si se escribe salir se termina se sale del while
        if query.lower() == "salir":
            break
        # Si es "qa" ejecuta el siguiente codigo
        if chat_type == "qa":
            # En la siguiente linea se inicia la var response que a su vez llama a la 
            # función o definición process_qa_query, quien para ejecutarse, necesita un 
            # Query o input, un retriver y un llm
            response = process_qa_query(query=query, retriever=retriever, llm=llm, chat_history=chat_history)
        # Si es "memory_chat" ejecuta el siguiente codigo
        elif chat_type == "memory_chat":
            # En la siguiente linea se inicia la var response que a su vez llama a la 
            # función o definición process_memory_query, quien para ejecutarse, necesita un 
            # Query o input, un retriver y un llm
            response = process_memory_query(
                query=query, 
                retriever=retriever, 
                llm=llm, 
                chat_history=chat_history
            )
        # Imprime en rojo la variable response
        console.print(f"[red]Chat PAO:[/red] {response}")


def process_qa_query(query, retriever, llm, chat_history):
    prompt = ChatPromptTemplate.from_messages(
      [
         ("placeholder", "{chat_history}"),
         ("user", "{query}"),
         (
             "user",
             "Dada la conversación anterior, genere una consulta de búsqueda para buscar información relevante de SIDEAP para la conversación.",
         ),
      ]   
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages(
      [
         (
             "system",
             "Solo puedes responder preguntas de SIDEAP:\n\n{context}",
         ),
         ("placeholder", "{chat_history}"),
         ("user", "{query}"),
      ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(
        retriever_chain,
        document_chain
    )
    console.print("[yellow]La IA estÃ¡ pensando...[/yellow]")
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]

def process_memory_query(query, retriever, llm, chat_history):
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        verbose=True
    )
    console.print("[yellow]La IA estÃ¡ pensando...[/yellow]")
    print(f"La historia antes de esta respuesta es: {chat_history}")
    result = conversation({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]

def main():

    documents = load_documents(get_file_path())
    get_openai_api_key()
    embeddings = OpenAIEmbeddings(
    deployment_id="Embedding",
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base
   )

    vectorstore_chroma = get_chroma_db(embeddings, documents, "chroma_docs")

    console.print(f"[green]Documentos {len(documents)} cargados.[/green]")

    llm = AzureChatOpenAI(
    deployment_name=chat_model,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base
    )

    run_conversation(vectorstore_chroma, chat_type, llm)


if __name__ == "__main__":
    main()
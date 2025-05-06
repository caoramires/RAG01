"""
Exemplo de RAG utilizando Streamlit, DeepSeek local com Ollama.

O Streamlit faz uma UI

O programa faz o processo resumido:

1. Carga com PDFlumberLoader
2. Divide em Chunks com SemanticChunker(HuggingFaceEmbeddings()) e text_splitter.split_documents(docs)
3. Cria embeddings com  HuggingFaceEmbeddings()
4. Vetoriza os Stores, com FAISS
5.
"""

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# 1. Streamlit UI
st.title("📄 RAG System with DeepSeek R1 & Ollama")

"""
2. Carrega o PDF
"""

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    """
    3. Chunking Semântico
    SemanticChunker do Langchain Experimental, utilizando embeddings do Hugging Face (HuggingFaceEmbeddings), 
    para dividir o texto do PDF em chunks semanticamente relevantes. A ideia por trás do chunking semântico é agrupar 
    frases e parágrafos que compartilham um significado similar, o que pode melhorar a relevância 
    do contexto recuperado."""

    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    """
    4. Criação do Vetor Store
    Utiliza HuggingFaceEmbeddings para gerar embeddings para cada chunk de texto.
    Cria um índice FAISS (uma biblioteca para busca de similaridade eficiente) a partir dos embeddings e dos 
    documentos (chunks). Este índice permitirá a busca rápida de chunks relevantes para uma dada pergunta."""

    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)

    """
    5. Criação do Retriever
    Cria um retriever (vector.as_retriever) a partir do índice FAISS.
    Configura a busca para ser por similaridade (search_type="similarity") e para retornar os 3 chunks mais 
    relevantes (search_kwargs={"k": 3})."""

    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    """
    6. Configuração do LLM
    Inicializa um modelo de linguagem grande (llm) utilizando Ollama, especificamente o 
    modelo deepseek-r1:1.5b. Ollama permite executar modelos LLM localmente."""

    llm = Ollama(model="deepseek-r1:1.5b")

    """
    7. Criação do Prompt
    Define um template de prompt (PromptTemplate) que será usado para instruir o LLM a responder à 
    pergunta do usuário com base no contexto recuperado. O prompt inclui as variáveis {context} (onde os chunks
    relevantes serão inseridos) e {question} (a pergunta do usuário)."""

    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""

    QA_PROMPT = PromptTemplate.from_template(prompt)

    """
    8. Criação das Chains
    LLMChain: Cria uma chain que combina o LLM (Ollama) com o prompt.
    StuffDocumentsChain: Cria uma chain para combinar os múltiplos documentos (chunks) recuperados em uma única 
    string de contexto que será inserida no prompt do LLM. A estratégia "stuff" simplesmente junta todos os documentos 
    em um único contexto.
    RetrievalQA: Cria a chain principal de RAG, que orquestra a recuperação dos documentos relevantes (via retriever) 
    e a geração da resposta pelo LLM (via combine_documents_chain).
     """

    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    """
    9. Interface de Pergunta e Resposta
    Fornece um campo de texto (st.text_input) para o usuário inserir sua pergunta sobre o documento.
    Quando o usuário faz uma pergunta:

    A pergunta é passada para a chain qa.
    A resposta gerada pelo LLM é extraída do resultado (["result"]).
    A resposta é exibida na interface Streamlit."""

    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        response = qa(user_input)["result"]
        st.write("**Response:**")
        st.write(response)

Código experimental de RAG - Retreavel-Augment Generation, para rodar localmente, utilizando:

    - Streamlit
    - LangChain
    - Hugging Face
    - Ollama com DeepSeek


Funcionalidade do Código:

    Interface Streamlit:
        Cria um título para a aplicação web: "📄 RAG System with DeepSeek R1 & Ollama".
        Fornece um widget para o usuário fazer o upload de um arquivo PDF.

    Carregamento do PDF:
        Quando um arquivo PDF é carregado, ele é salvo temporariamente como temp.pdf.
        Utiliza PDFPlumberLoader do Langchain para carregar o conteúdo do PDF em objetos Document.

    Chunking Semântico:
        Emprega SemanticChunker do Langchain Experimental, utilizando embeddings do Hugging Face
        (HuggingFaceEmbeddings), para dividir o texto do PDF em chunks semanticamente relevantes. A ideia por trás
        do chunking semântico é agrupar frases e parágrafos que compartilham um significado similar, o que pode
        melhorar a relevância do contexto recuperado.

    Criação do Vetor Store:
        Utiliza HuggingFaceEmbeddings para gerar embeddings para cada chunk de texto.
        Cria um índice FAISS (uma biblioteca para busca de similaridade eficiente) a partir dos embeddings e dos
        documentos (chunks). Este índice permitirá a busca rápida de chunks relevantes para uma dada pergunta.

    Configuração do Retriever:
        Cria um retriever (vector.as_retriever) a partir do índice FAISS.
        Configura a busca para ser por similaridade (search_type="similarity") e para retornar os 3 chunks mais
        relevantes (search_kwargs={"k": 3}).

    Configuração do LLM:
        Inicializa um modelo de linguagem grande (llm) utilizando Ollama, especificamente o modelo deepseek-r1:1.5b.
        Ollama permite executar modelos LLM localmente.

    Criação do Prompt:
        Define um template de prompt (PromptTemplate) que será usado para instruir o LLM a responder à pergunta do
        usuário com base no contexto recuperado. O prompt inclui as variáveis {context} (onde os chunks relevantes
        serão inseridos) e {question} (a pergunta do usuário).

    Criação das Chains:
        LLMChain: Cria uma chain que combina o LLM (Ollama) com o prompt.
        StuffDocumentsChain: Cria uma chain para combinar os múltiplos documentos (chunks) recuperados em uma única
        string de contexto que será inserida no prompt do LLM. A estratégia "stuff" simplesmente junta todos os
        documentos em um único contexto.
        RetrievalQA: Cria a chain principal de RAG, que orquestra a recuperação dos documentos relevantes
        (via retriever) e a geração da resposta pelo LLM (via combine_documents_chain).

    Interface de Pergunta e Resposta:
        Fornece um campo de texto (st.text_input) para o usuário inserir sua pergunta sobre o documento.
        Quando o usuário faz uma pergunta:
            A pergunta é passada para a chain qa.
            A resposta gerada pelo LLM é extraída do resultado (["result"]).
            A resposta é exibida na interface Streamlit.

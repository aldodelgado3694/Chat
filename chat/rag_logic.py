import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# --- Configuración (sin cambios) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_PATH = os.path.join(BASE_DIR, '..', 'documents')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, '..', 'vector_store')
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:8b" # o "llama3:latest"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Variables Globales (Mantenemos chain para caché de LLM y prompt) ---
# Quitamos vector_store y retriever de aquí, se crearán bajo demanda
chain = None
embedding_function = None # Guardamos la función de embedding para reutilizar

def get_embedding_function():
    """Inicializa y retorna la función de embeddings."""
    global embedding_function
    if embedding_function is None:
        logger.info(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME}")
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info("Modelo de embeddings cargado.")
    return embedding_function

# MODIFICADO: Esta función ahora SIEMPRE carga desde el disco
def load_vector_store_from_disk(embedding_function):
    """Carga la base de datos vectorial desde el disco."""
    if not os.path.exists(VECTOR_STORE_PATH):
        logger.warning(f"Directorio de Vector Store no encontrado en {VECTOR_STORE_PATH}. No se puede cargar.")
        return None # O crea un store vacío si prefieres: Chroma.from_documents([], embedding_function, persist_directory=VECTOR_STORE_PATH)

    logger.info(f"Cargando Vector Store desde: {VECTOR_STORE_PATH}")
    try:
        # Crea una NUEVA instancia directamente desde el directorio persistido
        vs = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embedding_function
        )
        logger.info("Vector Store cargado desde disco.")
        return vs
    except Exception as e:
        # Puede haber errores si el store está corrupto o vacío de forma inesperada
        logger.error(f"Error al cargar Vector Store desde disco: {e}", exc_info=True)
        return None


# MODIFICADO: Simplificado, solo añade documentos. La carga se hace en otro lado.
def add_document_to_vector_store(file_path):
    """Procesa un documento y lo añade al vector store existente en disco."""
    try:
        logger.info(f"Procesando y añadiendo documento: {file_path}")
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            logger.warning(f"Formato no soportado: {file_path}")
            return

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.warning(f"No se extrajeron chunks de: {file_path}")
            return

        # Obtener función de embedding
        embed_func = get_embedding_function()

        # Crear una instancia de Chroma para AÑADIR (necesita el directorio)
        # Esto implícitamente debería cargar el store existente y añadirle.
        # Chroma se encarga de la persistencia automática al usar persist_directory.
        vs_for_adding = Chroma(
             persist_directory=VECTOR_STORE_PATH,
             embedding_function=embed_func
        )
        vs_for_adding.add_documents(chunks)

        # --- NO SE NECESITA vs.persist() ---
        logger.info(f"Documento {file_path} procesado y añadido al vector store (auto-persistido).")

    except Exception as e:
        logger.error(f"Error procesando/añadiendo documento {file_path}: {e}", exc_info=True)


# MODIFICADO: Esta función ahora crea la cadena bajo demanda usando el store cargado del disco
def build_rag_chain():
    """Construye y retorna la cadena RAG completa cargando el VS desde disco."""
    try:
        embed_func = get_embedding_function()
        vector_store = load_vector_store_from_disk(embed_func)

        if vector_store is None:
            logger.error("No se pudo cargar el Vector Store, no se puede construir la cadena RAG.")
            return None

        retriever = vector_store.as_retriever(search_kwargs={'k': 5}) # Aumentamos k un poco

        logger.debug(f"Inicializando LLM desde Ollama: {OLLAMA_MODEL}") # Cambiado a debug
        llm = ChatOllama(model=OLLAMA_MODEL)

        template = """Eres un asistente de IA útil. Tu tarea es responder preguntas sobre documentos proporcionados.
        Usa SOLO la siguiente información de contexto para responder. NO uses conocimiento externo.
        Si la respuesta no está en el contexto, di explícitamente "La información no se encuentra en los documentos proporcionados".
        Si la respuesta no esta en el contecto, adicionalmente proporciona ideas o sugerencias sobre el contexto disponible.
        Sé conciso y responde directamente a la pregunta.

        Contexto:
        {context}

        Pregunta:
        {question}

        Respuesta útil:"""
        prompt = ChatPromptTemplate.from_template(template)

        # --- Logging de Contexto (Mantenido de la sugerencia anterior) ---
        def retrieve_and_log(query):
            logger.info(f"--- Recuperando contexto para query: '{query}' ---")
            try:
                retrieved_docs = retriever.invoke(query)
                logger.info(f"--- {len(retrieved_docs)} Documentos Recuperados ---")
                for i, doc in enumerate(retrieved_docs):
                    content_preview = doc.page_content[:200].strip().replace('\n', ' ') + "..." if len(doc.page_content) > 200 else doc.page_content.strip().replace('\n', ' ')
                    source = doc.metadata.get('source', 'N/A')
                    logger.info(f"Doc {i+1}: Origen='{os.path.basename(source)}', Preview='{content_preview}'")
                logger.info("-------------------------------------------")
                # Pasamos el contexto formateado a la siguiente etapa
                return "\n\n".join([doc.page_content for doc in retrieved_docs])
            except Exception as e_retrieve:
                 logger.error(f"Error durante la recuperación: {e_retrieve}", exc_info=True)
                 return "Error al recuperar contexto." # Evita que la cadena falle completamente

        # Construcción de la cadena RAG
        rag_chain = (
            # Usamos un diccionario para pasar la pregunta original Y el contexto recuperado
            {"context": RunnablePassthrough() | RunnableLambda(retrieve_and_log), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Cadena RAG construida bajo demanda.")
        return rag_chain

    except Exception as e:
        logger.error(f"Error construyendo la cadena RAG: {e}", exc_info=True)
        return None


# MODIFICADO: Ahora llama a build_rag_chain() en cada ejecución
def get_rag_response(query: str) -> str:
    """Obtiene una respuesta del LLM construyendo y usando la cadena RAG bajo demanda."""
    # Ya no dependemos de la variable global 'chain' para la ejecución
    rag_chain_instance = build_rag_chain()

    if rag_chain_instance is None:
         return "Error: No se pudo construir el sistema RAG. Revisa los logs del servidor."

    try:
        logger.info(f"Invocando cadena RAG para query: {query}")
        response = rag_chain_instance.invoke(query)
        logger.info(f"Respuesta generada por LLM: {response}")
        return response
    except Exception as e:
        logger.error(f"Error al invocar la cadena RAG: {e}", exc_info=True)
        return "Error: Ocurrió un problema al procesar tu pregunta."


# --- El monitor_docs ahora usa la nueva función ---
# (Asegúrate que monitor_docs.py importe y use add_document_to_vector_store)
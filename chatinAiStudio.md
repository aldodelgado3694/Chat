¡Excelente! Vamos a crear esa base funcional. Este código te proporcionará un proyecto Django con:

1.  Un **backend** que monitorea una carpeta (`documents/`), procesa nuevos archivos de texto, los indexa en una base de datos vectorial (ChromaDB) usando embeddings, y ofrece una API de chat RAG usando Langchain y Ollama.
2.  Un **frontend** Django simple con una interfaz de chat básica que interactúa con el backend.

**Requisitos Previos:**

1.  **Python Instalado:** Asegúrate de tener Python 3.8 o superior.
2.  **Ollama Instalado y Corriendo:**
    *   Descarga e instala Ollama desde [https://ollama.com/](https://ollama.com/)
    *   Descarga un modelo (Qwen o Llama 3 son buenas opciones). Abre tu terminal y ejecuta (por ejemplo):
        ```bash
        ollama pull qwen:latest
        # o
        ollama pull llama3:latest
        ```
    *   Asegúrate de que Ollama esté corriendo en segundo plano.

---

**Estructura del Proyecto:**

```
rag_project/
├── chat/                   # Nuestra aplicación Django
│   ├── migrations/
│   ├── static/chat/
│   │   └── chat.js         # Javascript para el chat
│   ├── templates/chat/
│   │   └── chat.html       # Plantilla HTML del chat
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py             # URLs específicas de la app chat
│   ├── views.py            # Vistas de Django (API y renderizado)
│   ├── rag_logic.py        # Lógica principal de RAG y procesamiento
│   └── management/
│       └── commands/
│           └── monitor_docs.py # Comando para monitorear documentos
├── documents/              # <-- COLOCA AQUÍ TUS ARCHIVOS .TXT
├── rag_project/            # Configuración del proyecto Django
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py         # Configuraciones del proyecto
│   ├── urls.py             # URLs principales del proyecto
│   └── wsgi.py
├── vector_store/           # Base de datos vectorial (se creará aquí)
├── manage.py               # Utilidad de Django
└── requirements.txt        # Dependencias de Python
└── .env                    # (Opcional) Para variables de entorno
└── .gitignore              # Para ignorar archivos/carpetas
```

---

**Paso 1: Crear el Proyecto y la App Django**

Abre tu terminal y ejecuta:

```bash
# Crea el directorio del proyecto y entra en él
mkdir rag_project
cd rag_project

# Crea un entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instala Django
pip install django

# Crea el proyecto Django
django-admin startproject rag_project .

# Crea la aplicación 'chat'
python manage.py startapp chat

# Crea los directorios necesarios
mkdir documents
mkdir vector_store
mkdir chat/static/chat
mkdir chat/templates/chat
mkdir chat/management/commands
touch chat/management/__init__.py
touch chat/management/commands/__init__.py
```

---

**Paso 2: Instalar Dependencias Adicionales**

Crea el archivo `requirements.txt` con el siguiente contenido:

```txt
django
langchain
langchain-community
langchain-chroma
langchain-text-splitters
ollama
sentence-transformers
watchdog
python-dotenv # Opcional pero buena práctica
# Si quieres procesar PDFs (requiere dependencias adicionales):
# pypdf
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

**Paso 3: Configurar Django (`rag_project/settings.py`)**

Edita `rag_project/settings.py`:

1.  Añade `'chat'` a la lista `INSTALLED_APPS`:

    ```python
    INSTALLED_APPS = [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'chat', # <-- Añade esto
    ]
    ```

2.  Configura la ruta de los archivos estáticos (al final del archivo):

    ```python
    STATIC_URL = 'static/'
    # Opcional: Si pones tus estáticos fuera de la app, descomenta y ajusta:
    # import os
    # STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
    ```

---

**Paso 4: Crear la Lógica RAG (`chat/rag_logic.py`)**

Crea el archivo `chat/rag_logic.py` con el siguiente contenido:

```python
import os
import logging
from langchain_community.document_loaders import TextLoader #, PyPDFLoader # Si usas PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Configuración ---
# Asegúrate que estas rutas sean correctas respecto a manage.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_PATH = os.path.join(BASE_DIR, '..', 'documents')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, '..', 'vector_store')
# Modelo de embedding (local, pequeño y efectivo)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Modelo LLM a usar desde Ollama (asegúrate que esté descargado)
OLLAMA_MODEL = "qwen:latest" # o "llama3:latest"

# Configura el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Variables Globales (Caché simple para no recargar en cada request) ---
vector_store = None
retriever = None
chain = None

def get_embedding_function():
    """Inicializa y retorna la función de embeddings."""
    logger.info(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME}")
    # Usamos HuggingFaceEmbeddings para modelos locales via sentence-transformers
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def load_or_create_vector_store(embedding_function):
    """Carga la base de datos vectorial existente o la crea si no existe."""
    global vector_store
    if vector_store is None:
        logger.info(f"Cargando/Creando base de datos vectorial en: {VECTOR_STORE_PATH}")
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embedding_function
        )
    return vector_store

def initialize_rag_chain():
    """Inicializa la cadena RAG completa."""
    global retriever, chain
    try:
        embedding_function = get_embedding_function()
        vs = load_or_create_vector_store(embedding_function)
        retriever = vs.as_retriever(search_kwargs={'k': 3}) # Obtener los 3 chunks más relevantes

        logger.info(f"Inicializando LLM desde Ollama: {OLLAMA_MODEL}")
        llm = ChatOllama(model=OLLAMA_MODEL)

        # Prompt Template
        template = """Eres un asistente de inteligencia artificial útil. Tu tarea es responder preguntas sobre documentos proporcionados.
        Utiliza la siguiente información de contexto para responder la pregunta del usuario.
        Si no sabes la respuesta basada en el contexto, di que no lo sabes, no intentes inventar una respuesta.
        Mantén la respuesta concisa y relevante.

        Contexto:
        {context}

        Pregunta:
        {question}

        Respuesta útil:"""
        prompt = ChatPromptTemplate.from_template(template)

        # Creación de la cadena RAG
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Cadena RAG inicializada correctamente.")
        return True

    except Exception as e:
        logger.error(f"Error inicializando la cadena RAG: {e}", exc_info=True)
        chain = None # Asegurar que no se use una cadena parcialmente inicializada
        return False

def process_document(file_path):
    """Procesa un único documento y lo añade al vector store."""
    global vector_store # Necesitamos modificar el store global si se crea/actualiza
    try:
        logger.info(f"Procesando documento: {file_path}")
        # Determinar el loader según la extensión (ampliable)
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        # elif file_path.endswith(".pdf"):
        #     loader = PyPDFLoader(file_path) # Necesita instalar pypdf
        else:
            logger.warning(f"Formato no soportado para el archivo: {file_path}")
            return

        documents = loader.load()

        # Dividir en chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.warning(f"No se extrajeron chunks del documento: {file_path}")
            return

        # Obtener la función de embeddings y el vector store
        embedding_function = get_embedding_function()
        vs = load_or_create_vector_store(embedding_function)

        # Añadir los nuevos chunks al vector store existente
        vs.add_documents(chunks)
        # Forzar la persistencia (Chroma debería hacerlo, pero por si acaso)
        vs.persist()
        logger.info(f"Documento {file_path} procesado y añadido al vector store.")

        # Re-inicializar la cadena si el vector store fue modificado
        # (Podría optimizarse, pero es seguro para el inicio)
        initialize_rag_chain()

    except Exception as e:
        logger.error(f"Error procesando el documento {file_path}: {e}", exc_info=True)


def get_rag_response(query: str) -> str:
    """Obtiene una respuesta del LLM usando la cadena RAG."""
    global chain
    if chain is None:
        logger.warning("Intentando obtener respuesta RAG pero la cadena no está inicializada. Intentando inicializar...")
        if not initialize_rag_chain():
             return "Error: No se pudo inicializar el sistema RAG. Revisa los logs."

    try:
        logger.info(f"Recibida query: {query}")
        response = chain.invoke(query)
        logger.info(f"Respuesta generada: {response}")
        return response
    except Exception as e:
        logger.error(f"Error al invocar la cadena RAG: {e}", exc_info=True)
        return "Error: Ocurrió un problema al procesar tu pregunta."

# --- Inicialización al cargar el módulo ---
# Intenta inicializar la cadena RAG cuando Django carga esta app por primera vez
# Esto puede ocurrir al iniciar el servidor o el monitor de documentos
if not initialize_rag_chain():
     logger.error("FALLO en la inicialización inicial de RAG al cargar rag_logic.py")

```

---

**Paso 5: Crear el Monitor de Documentos (`chat/management/commands/monitor_docs.py`)**

Crea el archivo `chat/management/commands/monitor_docs.py`:

```python
import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from django.core.management.base import BaseCommand
from chat.rag_logic import process_document, DOCUMENTS_PATH # Importa la lógica y la ruta

logger = logging.getLogger(__name__)

class DocumentEventHandler(FileSystemEventHandler):
    """Maneja eventos del sistema de archivos para nuevos documentos."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt')): # Añade más extensiones si es necesario , '.pdf'
            logger.info(f"Nuevo archivo detectado: {event.src_path}")
            try:
                # Espera un poco por si el archivo aún se está escribiendo
                time.sleep(1)
                process_document(event.src_path)
            except Exception as e:
                logger.error(f"Error al procesar el nuevo archivo {event.src_path}: {e}", exc_info=True)

class Command(BaseCommand):
    help = 'Monitorea el directorio de documentos en busca de nuevos archivos y los procesa.'

    def handle(self, *args, **options):
        if not os.path.exists(DOCUMENTS_PATH):
            os.makedirs(DOCUMENTS_PATH)
            self.stdout.write(self.style.SUCCESS(f'Directorio de documentos creado en: {DOCUMENTS_PATH}'))

        self.stdout.write(self.style.SUCCESS(f'Iniciando monitor de documentos en: {DOCUMENTS_PATH}'))
        event_handler = DocumentEventHandler()
        observer = Observer()
        observer.schedule(event_handler, DOCUMENTS_PATH, recursive=False) # No recursivo
        observer.start()
        self.stdout.write(self.style.SUCCESS('Monitor iniciado. Presiona Ctrl+C para detener.'))
        try:
            while True:
                time.sleep(5) # Revisa cada 5 segundos (watchdog es basado en eventos, esto es solo para mantener el proceso vivo)
        except KeyboardInterrupt:
            observer.stop()
            self.stdout.write(self.style.WARNING('Deteniendo el monitor...'))
        observer.join()
        self.stdout.write(self.style.SUCCESS('Monitor detenido.'))

```

---

**Paso 6: Crear las Vistas (`chat/views.py`)**

Edita `chat/views.py`:

```python
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # Para simplificar en el piloto
import json
from .rag_logic import get_rag_response, initialize_rag_chain

# Intenta inicializar RAG al iniciar el servidor web
# Esto es redundante si el monitor ya lo hizo, pero asegura que esté listo
if not initialize_rag_chain():
     print("ADVERTENCIA: Fallo al inicializar RAG desde views.py")

def chat_view(request):
    """Renderiza la página principal del chat."""
    return render(request, 'chat/chat.html')

# csrf_exempt es para pruebas rápidas. En producción, usa el manejo de CSRF de Django.
@csrf_exempt
def chat_api(request):
    """API para manejar los mensajes del chat."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message')

            if not user_message:
                return JsonResponse({'error': 'Mensaje vacío recibido'}, status=400)

            # Obtener respuesta del sistema RAG
            bot_response = get_rag_response(user_message)

            return JsonResponse({'response': bot_response})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido'}, status=400)
        except Exception as e:
            # Loggear el error real en el servidor
            print(f"Error en chat_api: {e}") # Reemplaza con logging real
            return JsonResponse({'error': 'Ocurrió un error interno'}, status=500)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)

```

---

**Paso 7: Crear las URLs (`chat/urls.py` y `rag_project/urls.py`)**

1.  Crea el archivo `chat/urls.py`:

    ```python
    from django.urls import path
    from . import views

    app_name = 'chat' # Namespace para las URLs de esta app

    urlpatterns = [
        path('', views.chat_view, name='chat_view'),        # Vista principal del chat
        path('api/chat/', views.chat_api, name='chat_api'), # Endpoint de la API del chat
    ]
    ```

2.  Edita `rag_project/urls.py` para incluir las URLs de la app `chat`:

    ```python
    from django.contrib import admin
    from django.urls import path, include # Asegúrate de importar include

    urlpatterns = [
        path('admin/', admin.site.urls),
        path('', include('chat.urls')), # Incluye las URLs de la app chat en la raíz
    ]
    ```

---

**Paso 8: Crear la Plantilla HTML (`chat/templates/chat/chat.html`)**

Crea el archivo `chat/templates/chat/chat.html`:

```html
{% load static %}
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat RAG Piloto</title>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; background-color: #f4f4f4; }
        #chat-container { width: 80%; max-width: 600px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); display: flex; flex-direction: column; height: 80vh; }
        #chat-box { flex-grow: 1; padding: 20px; overflow-y: auto; border-bottom: 1px solid #eee; }
        .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 18px; max-width: 80%; word-wrap: break-word; }
        .user-message { background-color: #dcf8c6; align-self: flex-end; margin-left: auto; }
        .bot-message { background-color: #f1f0f0; align-self: flex-start; margin-right: auto; }
        #input-area { display: flex; padding: 15px; border-top: 1px solid #eee; }
        #user-input { flex-grow: 1; padding: 10px; border: 1px solid #ccc; border-radius: 20px; margin-right: 10px; }
        #send-button { padding: 10px 20px; background-color: #5cb85c; color: white; border: none; border-radius: 20px; cursor: pointer; }
        #send-button:hover { background-color: #4cae4c; }
        .loading { text-align: center; color: #888; font-style: italic;}
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="chat-box">
            <div class="bot-message message">Hola 👋 Soy tu asistente RAG. Pregúntame algo sobre los documentos cargados.</div>
            <!-- Los mensajes se añadirán aquí -->
        </div>
        <div id="input-area">
            <input type="text" id="user-input" placeholder="Escribe tu mensaje...">
            <button id="send-button">Enviar</button>
        </div>
    </div>

    <script src="{% static 'chat/chat.js' %}"></script>
</body>
</html>
```

---

**Paso 9: Crear el JavaScript del Chat (`chat/static/chat/chat.js`)**

Crea el archivo `chat/static/chat/chat.js`:

```javascript
document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    // Necesitamos obtener el token CSRF para las peticiones POST en Django
    // Una forma es buscarlo en las cookies si está configurado, o pasarlo desde el template.
    // Para este ejemplo simple con @csrf_exempt en la vista API, no es estrictamente necesario,
    // pero es buena práctica incluirlo para el futuro.
    // const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || ''; // Necesitarías añadir {% csrf_token %} en el form del HTML

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        messageDiv.textContent = text;
        chatBox.appendChild(messageDiv);
        // Scroll automático al último mensaje
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function showLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'bot-message', 'loading');
        loadingDiv.textContent = 'Pensando...';
        loadingDiv.id = 'loading-indicator';
        chatBox.appendChild(loadingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function hideLoadingIndicator() {
        const loadingDiv = document.getElementById('loading-indicator');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }


    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (!messageText) return; // No enviar mensajes vacíos

        addMessage(messageText, 'user');
        userInput.value = ''; // Limpiar input
        showLoadingIndicator(); // Mostrar indicador de carga

        try {
            const response = await fetch('/api/chat/', { // URL de nuestra API
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // Descomenta y ajusta si usas CSRF token
                    // 'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({ message: messageText })
            });

            hideLoadingIndicator(); // Ocultar indicador

            if (!response.ok) {
                // Intentar leer el error del JSON si es posible
                let errorMsg = `Error ${response.status}: ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* No hacer nada si no hay JSON */ }
                addMessage(`Error al obtener respuesta: ${errorMsg}`, 'bot');
                return;
            }

            const data = await response.json();
            addMessage(data.response, 'bot');

        } catch (error) {
            hideLoadingIndicator(); // Ocultar indicador en caso de error de red
            console.error("Error en fetch:", error);
            addMessage(`Error de conexión: ${error.message}`, 'bot');
        }
    }

    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});

```

---

**Paso 10: Crear `.gitignore` (Opcional pero Recomendado)**

Crea un archivo `.gitignore` en la raíz del proyecto:

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# Usually these files are written by a python script from a template
# before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal
media/

# Vector Store (si no quieres versionarlo)
vector_store/

# Mac files
.DS_Store

# IDE files
.idea/
.vscode/
```

---

**Paso 11: Ejecutar el Proyecto**

1.  **Añade Documentos:** Coloca algunos archivos `.txt` con información relevante dentro de la carpeta `documents/`.
2.  **Ejecuta las Migraciones de Django:** (Aunque no creamos modelos propios, Django necesita su estructura inicial)
    ```bash
    python manage.py migrate
    ```
3.  **Inicia el Monitor de Documentos (en una terminal separada):**
    ```bash
    python manage.py monitor_docs
    ```
    Deberías ver mensajes indicando que está monitoreando y procesando los archivos que añadiste. Espera a que termine de procesar.
4.  **Inicia el Servidor de Desarrollo Django (en otra terminal):**
    ```bash
    python manage.py runserver
    ```
5.  **Abre tu Navegador:** Ve a `http://127.0.0.1:8000/`. Deberías ver la interfaz de chat.
6.  **Prueba:** Escribe preguntas relacionadas con el contenido de tus documentos `.txt`.

---

¡Listo! Ahora tienes una base funcional para tu chatbot RAG. Puedes empezar a experimentar añadiendo más documentos, probando diferentes modelos en Ollama, mejorando el prompt, y expandiendo la funcionalidad.
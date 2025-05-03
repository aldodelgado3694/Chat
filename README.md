# Chat IA en local

En este proyecto de VibeCoding, ejecuto codigo generado por Gemini 2.5 Pro Preview 03-25, el codigo fue funcional desde el incicio. Se corrigen errores menores.

## Funcionalidad basica

- Carga documentos ".txt".
- Pregunta al chat sobre los documentos cargados.
- No es posible cargar documentos mientras el servidor est funcionando.
- Ejecuta desde otro dispositivo en la red local

## Requisitos Previos

Python Instalado: Asegúrate de tener Python 3.8 o superior.

Ollama Instalado y Corriendo:
Descarga e instala Ollama desde [ollama.com](https://ollama.com/)
Descarga un modelo (Qwen o Llama 3 son buenas opciones). Abre tu terminal y ejecuta (por ejemplo):

```sh
ollama pull qwen3:8b
# o
ollama pull <OTROLLM>
```

Asegúrate de que Ollama esté corriendo en segundo plano.

### Crear y activa ambiente virtual

```sh
python -m venv venv # Crear ambiente
source venv/bin/activate  #Activa ambiente
# En Windows: venv\Scripts\activate

```

### Instala Django

```sh
pip install django
```

### Instala las dependencias

```sh
pip install -r requirements.txt
```

### Modifica "Chat\rag_project\settings.py"

```Python
ALLOWED_HOSTS = ['<IP_CLIENTE>', '0.0.0.0', 'localhost', '127.0.0.1', "<IP_HOST>"] 
```

## Iniciando proyecto

### Inicia el Monitor de Documentos (en una terminal separada)

- Ejecuta en terminal

```sh
python manage.py monitor_docs
```

- Carga documentos a la carpeta doccuments fuera de la carpeta del proyecto.

### Inicia el Servidor de Desarrollo Django (en otra terminal)

```sh
python manage.py runserver <IP_HOST>:8000
```

### Abre el puerto 8000

Si se deseas acceder desde otro dispositivo en la red local, no olvidez abrir el puerto TCP 8000 en en firewall.

## Disfruta

Abre en el navegador [http://<IP_HOST>:8000/](http://<IP_HOST>:8000/)

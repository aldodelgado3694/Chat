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
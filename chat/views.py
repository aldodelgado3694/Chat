from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # Para simplificar en el piloto
import json
# MODIFICADO: Solo importamos get_rag_response
from .rag_logic import get_rag_response
import logging # Añadir para loguear errores

logger = logging.getLogger(__name__) # Configurar logger para esta vista

# YA NO es necesaria la inicialización global aquí

def chat_view(request):
    """Renderiza la página principal del chat."""
    return render(request, 'chat/chat.html')

@csrf_exempt
def chat_api(request):
    """API para manejar los mensajes del chat."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message')

            if not user_message:
                return JsonResponse({'error': 'Mensaje vacío recibido'}, status=400)

            # Obtener respuesta del sistema RAG (que ahora construye la cadena internamente)
            bot_response = get_rag_response(user_message)

            return JsonResponse({'response': bot_response})

        except json.JSONDecodeError:
            logger.warning("Error de decodificación JSON en chat_api", exc_info=True)
            return JsonResponse({'error': 'JSON inválido recibido'}, status=400)
        except Exception as e:
            # Loguear el error real en el servidor para poder depurar
            logger.error(f"Error inesperado en chat_api: {e}", exc_info=True)
            return JsonResponse({'error': 'Ocurrió un error interno en el servidor'}, status=500)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)
import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from django.core.management.base import BaseCommand
from chat.rag_logic import add_document_to_vector_store, DOCUMENTS_PATH # Importa la lógica y la ruta

logger = logging.getLogger(__name__)

class DocumentEventHandler(FileSystemEventHandler):
    """Maneja eventos del sistema de archivos para nuevos documentos."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt')): # Añade más extensiones si es necesario , '.pdf'
            logger.info(f"Nuevo archivo detectado: {event.src_path}")
            try:
                time.sleep(1)
                add_document_to_vector_store(event.src_path) # <--- Cambiado
            except Exception as e:
                logger.error(f"Error al procesar/añadir el nuevo archivo {event.src_path}: {e}", exc_info=True)

class Command(BaseCommand):
    help = 'Monitorea el directorio de documentos en busca de nuevos archivos y los procesa.'

    def handle(self, *args, **options):
        if not os.path.exists(DOCUMENTS_PATH):
            os.makedirs(DOCUMENTS_PATH)
            self.stdout.write(self.style.SUCCESS(f'Directorio de documentos creado en: {DOCUMENTS_PATH}'))

        self.stdout.write(self.style.SUCCESS(f'Iniciando monitor de documentos en: {DOCUMENTS_PATH}'))
        event_handler = DocumentEventHandler()
        observer = Observer()
        observer.schedule(event_handler, DOCUMENTS_PATH, recursive=True) # No recursivo
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
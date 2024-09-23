# TeenTalk
#Proyecto de IA de salud mental 
# Instalar las bibliotecas necesarias
!pip install --upgrade openai gradio PyPDF2 numpy textblob nltk faiss-cpu tiktoken

# Importar las bibliotecas
import openai
import numpy as np
import sqlite3
import uuid
from datetime import datetime
from textblob import TextBlob
import nltk
import faiss
import os
import pickle
import tiktoken
import gradio as gr
import PyPDF2  # Importar PyPDF2

# Descargar datos de NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Montar Google Drive para acceder a los PDFs
from google.colab import drive
drive.mount('/content/drive')

# Solicitar la clave de API de OpenAI de manera segura
import getpass
openai.api_key = getpass.getpass('Introduce tu clave de API de OpenAI: ')

# Función para extraer texto de un PDF
def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Función para dividir el texto en fragmentos
def split_text(text, max_tokens=500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    chunk = ''
    tokens = 0
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if tokens + sentence_tokens <= max_tokens:
            chunk += ' ' + sentence
            tokens += sentence_tokens
        else:
            if chunk:
                chunks.append(chunk.strip())
            chunk = sentence
            tokens = sentence_tokens
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Función para crear embeddings
def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = openai.Embedding.create(
            input=chunk,
            model='text-embedding-ada-002'
        )['data'][0]['embedding']
        embeddings.append(embedding)
    return embeddings

# Cargar y procesar los PDFs
def load_and_process_pdfs(pdf_paths):
    cache_file = 'embeddings_cache.pkl'
    if os.path.exists(cache_file):
        # Cargar embeddings y chunks desde el archivo cache
        with open(cache_file, 'rb') as f:
            all_chunks, all_embeddings = pickle.load(f)
    else:
        all_chunks = []
        all_embeddings = []
        for pdf_path in pdf_paths:
            print(f"Procesando: {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text(text)
            embeddings = create_embeddings(chunks)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
        # Guardar embeddings y chunks en el archivo cache
        with open(cache_file, 'wb') as f:
            pickle.dump((all_chunks, all_embeddings), f)
    return all_chunks, np.array(all_embeddings, dtype='float32')

# Especifica las rutas a tus archivos PDF en Google Drive
pdf_paths = [
    '/content/drive/MyDrive/TeenTalk/429452144-Resumen-DSM-5.pdf',
    '/content/drive/MyDrive/TeenTalk/CBT_adolescents_individual_participante_esp.pdf',
    '/content/drive/MyDrive/TeenTalk/estados.pdf',
    '/content/drive/MyDrive/TeenTalk/GUIA-PREVENCION-SUICIDIO-EN-ESTABLECIMIENTOS-EDUCACIONALES-web.pdf'
]

# Asegúrate de reemplazar 'TU_CARPETA' y los nombres de los archivos PDF con los correctos.

# Cargar y procesar los PDFs
chunks, embeddings = load_and_process_pdfs(pdf_paths)

# Construir el índice FAISS
if embeddings is not None and len(embeddings) > 0:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
else:
    index = None

# Función para detectar la intención (intent)
def detect_intent(message):
    intents = {
        'consulta_diagnostica': ['diagnóstico', 'síntomas', 'padezco', 'tengo', 'siento', 'me siento'],
        'busqueda_ayuda': ['ayuda', 'necesito', 'apoyo', 'orientación', 'consejo', 'aconseja'],
        'informacion_general': ['qué es', 'información', 'explica', 'dime'],
        # Agrega más intenciones y palabras clave si es necesario
    }
    for intent, keywords in intents.items():
        for keyword in keywords:
            if keyword in message.lower():
                return intent
    return 'desconocido'

# Función para analizar el sentimiento del mensaje
def analyze_sentiment(message):
    blob = TextBlob(message)
    return blob.sentiment.polarity  # Devuelve un valor entre -1 (negativo) y 1 (positivo)

# Función para verificar si el usuario desea conectarse con un psicólogo
def check_for_connection_request(message):
    keywords = ['sí', 'si', 'quiero', 'me gustaría', 'deseo', 'por favor', 'conectar', 'psicólogo', 'psicologa', 'ayuda profesional', 'terapia']
    for keyword in keywords:
        if keyword in message.lower():
            return True
    return False

# Función para inicializar la base de datos
def init_db():
    conn = sqlite3.connect('chatbot_conversations.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT,
            user_id TEXT,
            user_message TEXT,
            assistant_response TEXT,
            intent TEXT,
            sentiment REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Función para guardar la conversación en la base de datos
def save_conversation(conversation_id, user_id, user_message, assistant_response, intent, sentiment):
    conn = sqlite3.connect('chatbot_conversations.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute('''
        INSERT INTO conversations (conversation_id, user_id, user_message, assistant_response, intent, sentiment, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (conversation_id, user_id, user_message, assistant_response, intent, sentiment, timestamp))
    conn.commit()
    conn.close()

# Inicializar la base de datos
init_db()

# Generar IDs únicos para la conversación y el usuario
conversation_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())  # En un caso real, esto vendría de un sistema de autenticación

# Historial de la conversación
historial_conversacion = []

# Variable para rastrear el estado de la conversación
estado_conversacion = {
    'indagacion_inicial': False,
    'ofrecio_consejo': False,
    'indagacion_adicional': False,
    'pregunto_utilidad': False,
    'derivacion_ofrecida': False
}

# Función para contar tokens
def contar_tokens(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Cada mensaje tiene tokens adicionales
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # Tokens adicionales de finalización
    return num_tokens

# Función para recuperar documentos relevantes usando FAISS
def retrieve_documents(query, chunks, index, embeddings, top_k=3):
    if index is None:
        return []
    # Crear embedding de la consulta
    query_embedding = openai.Embedding.create(
        input=query,
        model='text-embedding-ada-002'
    )['data'][0]['embedding']
    query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
    # Buscar en el índice
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Función principal para obtener la respuesta
def obtener_respuesta(mensaje, historial, estado):
    # Agregar el mensaje del usuario al historial
    historial.append({"role": "user", "content": mensaje})

    # Detectar intención y analizar sentimiento
    intent = detect_intent(mensaje)
    sentiment = analyze_sentiment(mensaje)

    # Recuperar documentos relevantes usando RAG
    documents = retrieve_documents(mensaje, chunks, index, embeddings)
    documents_prompt = "\n\n".join(documents)

    # Construir el prompt para el modelo
    system_prompt = (
        "Eres un asistente virtual de salud mental que proporciona consejos útiles y empáticos "
        "basados en información confiable. Muestras empatía y comprensión en tus respuestas. "
        "Utiliza la siguiente información de apoyo para ayudar al usuario:\n\n"
        f"{documents_prompt}\n\n"
        "Si la información no es suficiente, ofrece apoyo general pero no inventes información."
        "Eres experto detectando problemas psicologicos e indagas al menos 5 veces antes de ofrecer derivación"
        "Si el usuario describe situaciones que en tu juicio requieran una atención mas profunda, derivar"
        "La derivación que ofreces es a un psicologo en linea dentro del mismo chat"
    )

    prompt = [
        {"role": "system", "content": system_prompt},
    ]

    # Añadir los últimos mensajes del historial para mantener el contexto
    mensajes_recientes = historial[-6:]  # Limita a los últimos 6 mensajes (3 interacciones)
    prompt.extend(mensajes_recientes)

    # Verificar el número de tokens para no exceder el límite
    num_tokens = contar_tokens(prompt)
    max_tokens = 4096 - num_tokens - 500  # Dejar espacio para la respuesta
    if max_tokens <= 0:
        # Si excede el límite, reducir el historial
        mensajes_recientes = historial[-4:]
        prompt = [{"role": "system", "content": system_prompt}] + mensajes_recientes
        max_tokens = 4096 - contar_tokens(prompt) - 500

    # Realizar la solicitud a la API de OpenAI
    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-4o",  # Cambia a "gpt-4" si tienes acceso
            messages=prompt,
            max_tokens=500,
            temperature=0.7,
        )
        # Obtener la respuesta del asistente
        respuesta_asistente = respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al generar la respuesta: {e}")
        respuesta_asistente = "Lo siento, estoy teniendo dificultades para procesar tu solicitud en este momento."

    # Agregar la respuesta al historial
    historial.append({"role": "assistant", "content": respuesta_asistente})

    # Guardar la conversación en la base de datos
    save_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        user_message=mensaje,
        assistant_response=respuesta_asistente,
        intent=intent,
        sentiment=sentiment
    )

    # Preparar las respuestas para el chatbot de Gradio
    conversaciones = []
    for i in range(0, len(historial), 2):
        user_msg = historial[i]['content']
        assistant_msg = ''
        if i+1 < len(historial):
            assistant_msg = historial[i+1]['content']
        conversaciones.append((user_msg, assistant_msg))
    return conversaciones, historial, estado

# Crear la interfaz con Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Asistente Virtual de Salud Mental")
    chatbot = gr.Chatbot()
    with gr.Row():
        mensaje = gr.Textbox(
            show_label=False,
            placeholder='Escribe tu mensaje aquí...',
        )
    estado_historial = gr.State([])
    estado_conversacion = gr.State({
        'indagacion_inicial': False,
        'ofrecio_consejo': False,
        'indagacion_adicional': False,
        'pregunto_utilidad': False,
        'derivacion_ofrecida': False
    })

    def respuesta_usuario(mensaje, historial, estado):
        conversaciones, historial_actualizado, estado_actualizado = obtener_respuesta(mensaje, historial, estado)
        return conversaciones, historial_actualizado, estado_actualizado

    mensaje.submit(respuesta_usuario, [mensaje, estado_historial, estado_conversacion], [chatbot, estado_historial, estado_conversacion])

# Lanzar la aplicación
demo.launch(share=True)

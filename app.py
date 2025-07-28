import streamlit as st
import os
from dotenv import load_dotenv

# Carga las variables de entorno desde .env (para tu API Key)
load_dotenv()

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Insight Navigator AI",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Insight Navigator AI")
st.markdown("### Tu Analista de Datos No Estructurados 24/7")

st.write(
    "Sube documentos (PDFs, TXT) o pega texto para que la IA los analice y obtén insights valiosos al instante."
)

# --- Contenedor para Carga de Documentos ---
st.header("1. Carga o Pega tus Datos")

uploaded_files = st.file_uploader(
    "Sube tus documentos (PDF, TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

text_input = st.text_area(
    "O pega el texto directamente aquí:",
    height=200,
    placeholder="Pega tu texto aquí (ej. un artículo, feedback de clientes, etc.)"
)

# Aquí es donde procesaremos los archivos/texto en la siguiente etapa
if uploaded_files or text_input:
    st.success("Archivos o texto recibidos. ¡Listo para analizar!")
    # Aquí irá la lógica para procesar estos datos en la base de conocimiento.

st.markdown("---")

# --- Contenedor del Chatbot ---
st.header("2. Pregunta a tu AI Navigator")

# Inicializa el historial del chat si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra los mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario para el chatbot
if prompt := st.chat_input("Haz una pregunta sobre tus datos o pide un análisis..."):
    # Añade el mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Aquí es donde la IA generará la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Esta parte se reemplazará con la lógica real de la IA
            response = "Todavía estoy aprendiendo sobre tus datos. ¡Pronto podré responder a esto!"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.info("Desarrollado con ❤️ y IA. Potenciado por modelos de lenguaje avanzados.")
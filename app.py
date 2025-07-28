import streamlit as st
import os
# No necesitamos dotenv para st.secrets en despliegues en la nube,
# pero lo mantenemos para pruebas locales con .env si es necesario, aunque st.secrets es preferible.
# from dotenv import load_dotenv # Comentado si usas solo st.secrets para producción

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # NUEVAS IMPORTACIONES
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai # Necesario para configurar la clave directamente

# --- 0. Configuración de la API Key (usando st.secrets) ---

# Para el despliegue en Streamlit Cloud, st.secrets es la forma segura.
# Para pruebas locales, puedes crear un archivo .streamlit/secrets.toml
# o usar un .env como fallback (aunque st.secrets es preferible).
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Error: La clave de API de Google Gemini no se encuentra en st.secrets."
             "Por favor, configúrala en tu archivo .streamlit/secrets.toml (local) o en la configuración de secretos de Streamlit Cloud.")
    st.stop()

# Configurar la API Key para google.generativeai (para embeddings directamente)
genai.configure(api_key=GOOGLE_API_KEY)


# --- 1. Funciones de Procesamiento de Documentos ---

def get_document_text(docs):
    """Extrae texto de una lista de objetos de archivo subidos (PDF, TXT)."""
    text = ""
    for doc in docs:
        file_extension = os.path.splitext(doc.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file_extension == ".txt":
            text += doc.read().decode("utf-8")
    return text

def get_text_chunks(text):
    """Divide el texto largo en fragmentos más pequeños y manejables."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Crea un vectorstore (base de datos vectorial) a partir de los fragmentos de texto."""
    st.info("Creando base de conocimiento. Esto puede tardar unos segundos...")
    try:
        # Usamos GoogleGenerativeAIEmbeddings ahora
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.success("Base de conocimiento creada exitosamente.")
        return vectorstore
    except Exception as e:
        st.error(f"Error al crear la base de conocimiento: {e}")
        st.warning("Asegúrate de que tu clave de Google Gemini sea válida y que tengas acceso al modelo 'embedding-001'.")
        st.stop()

# --- 2. Función para la Cadena Conversacional (Chatbot) ---

def get_conversation_chain(vectorstore):
    """Configura la cadena de conversación para el chatbot."""
    # Usamos ChatGoogleGenerativeAI ahora
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- 3. Función Principal para el Manejo del Chat ---

def handle_user_input(user_question):
    """Maneja la pregunta del usuario y genera la respuesta de la IA."""
    if st.session_state.conversation:
        with st.spinner("Generando respuesta..."):
            try:
                response = st.session_state.conversation({'question': user_question})
                # El formato de respuesta puede variar ligeramente entre modelos/cadenas
                # Asegúrate de acceder al contenido final de la respuesta de la IA
                ai_response_content = response['chat_history'][-1].content
                st.session_state.messages.append({"role": "assistant", "content": ai_response_content})
            except Exception as e:
                st.error(f"Error al generar la respuesta de la IA: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Lo siento, hubo un error al procesar tu solicitud. Por favor, intenta de nuevo."})
    else:
        st.warning("Por favor, sube y procesa documentos primero para que la IA tenga información.")

# --- 4. Interfaz de Usuario de Streamlit (Función main) ---

def main():
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

    # Inicializa el estado de la sesión si no existe
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_text_exists" not in st.session_state:
        st.session_state.processed_text_exists = False

    # --- Contenedor Lateral (Sidebar) para Carga y Procesamiento ---
    with st.sidebar:
        st.header("1. Carga o Pega tus Datos")

        uploaded_files = st.file_uploader(
            "Sube tus documentos (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        text_input = st.text_area(
            "O pega el texto directamente aquí:",
            height=200,
            placeholder="Pega tu texto aquí (ej. un artículo, feedback de clientes, etc.)",
            key="text_area"
        )

        # Botón para procesar los datos
        if st.button("Procesar Datos", key="process_button"):
            raw_text = ""
            if uploaded_files:
                raw_text += get_document_text(uploaded_files)
            if text_input:
                raw_text += text_input

            if raw_text:
                st.info("Procesando tus datos. Esto puede tomar un momento...")
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.processed_text_exists = True
                st.success("¡Datos procesados! Ahora puedes empezar a preguntar.")
                # Limpiar la entrada de texto después de procesar para evitar doble procesamiento al recargar
                # st.session_state.text_area = "" # Esto no siempre funciona bien con st.text_area
            else:
                st.warning("Por favor, sube al menos un archivo o pega texto antes de procesar.")

        st.markdown("---")
        st.info("Para reiniciar el chat y cargar nuevos documentos, simplemente actualiza la página.")

    # --- Contenedor Principal del Chatbot ---
    st.header("2. Pregunta a tu AI Navigator")

    # Muestra los mensajes anteriores en el chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de usuario para el chatbot
    if user_question := st.chat_input("Haz una pregunta sobre tus datos o pide un análisis..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        handle_user_input(user_question)

    st.markdown("---")
    st.info("Desarrollado con ❤️ y IA. Potenciado por modelos de lenguaje avanzados.")

if __name__ == "__main__":
    main()

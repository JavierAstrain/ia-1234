import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai

# --- 0. Configuración de la API Key (usando st.secrets) ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Error: La clave de API de Google Gemini no se encuentra en st.secrets.")
    st.error("Por favor, configura GOOGLE_API_KEY en tu archivo .streamlit/secrets.toml (para pruebas locales) o en la configuración de secretos de Streamlit Cloud.")
    st.stop() # Detiene la ejecución si la clave no se encuentra

# Configurar la API Key para el SDK de Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)


# --- 1. Funciones de Procesamiento de Documentos ---

def get_document_text(docs):
    """Extrae texto de una lista de objetos de archivo subidos (PDF, TXT)."""
    text = ""
    for doc in docs:
        file_extension = os.path.splitext(doc.name)[1].lower()
        if file_extension == ".pdf":
            try:
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                st.warning(f"No se pudo leer el archivo PDF '{doc.name}': {e}. Asegúrate de que no esté corrupto o cifrado. Ignorando este archivo.")
        elif file_extension == ".txt":
            try:
                text += doc.read().decode("utf-8")
            except Exception as e:
                st.warning(f"No se pudo leer el archivo TXT '{doc.name}': {e}. Asegúrate de que la codificación sea UTF-8. Ignorando este archivo.")
        else:
            st.warning(f"Tipo de archivo no soportado: '{doc.name}'. Solo se admiten .pdf y .txt por ahora.")
    return text

def get_text_chunks(text):
    """Divide el texto largo en fragmentos más pequeños y manejables."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Tamaño de cada fragmento
        chunk_overlap=200,        # Superposición entre fragmentos para mantener contexto
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource # Almacena en caché la base vectorial para no recrearla en cada rerun de la app
def get_vectorstore(text_chunks, api_key):
    """Crea un vectorstore (base de datos vectorial) a partir de los fragmentos de texto."""
    st.info("Creando base de conocimiento. Esto puede tardar unos segundos...")
    try:
        # Usamos GoogleGenerativeAIEmbeddings con el nombre de modelo correcto
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.success("Base de conocimiento creada exitosamente.")
        return vectorstore
    except Exception as e:
        st.error(f"Error al crear la base de conocimiento: {e}")
        st.warning("Asegúrate de que tu clave de Google Gemini sea válida y que tengas acceso al modelo 'models/embedding-001'.")
        st.stop()

# --- 2. Función para la Cadena Conversacional (Chatbot) ---

# Se eliminó @st.cache_resource aquí porque el objeto vectorstore no es hashable fácilmente.
# La creación de la cadena y la memoria es rápida y no justifica el caché aquí.
def get_conversation_chain(vectorstore, api_key):
    """Configura la cadena de conversación para el chatbot."""
    # Usamos ChatGoogleGenerativeAI para el LLM principal
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.7, convert_system_message_to_human=True)
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
                ai_response_content = response['chat_history'][-1].content
                st.session_state.messages.append({"role": "assistant", "content": ai_response_content})
            except Exception as e:
                st.error(f"Error al generar la respuesta de la IA: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Lo siento, hubo un error al procesar tu solicitud. Por favor, revisa los logs o la consola para más detalles, y asegúrate de que tu clave de API sea válida y el modelo esté disponible."})
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
    # Usamos una clave para el uploader para que se resetee si cambiamos de archivos
    if "last_processed_data_hash" not in st.session_state: # Para rastrear si los datos de entrada han cambiado
        st.session_state.last_processed_data_hash = None


    # --- Contenedor Lateral (Sidebar) para Carga y Procesamiento ---
    with st.sidebar:
        st.header("1. Carga o Pega tus Datos")

        uploaded_files = st.file_uploader(
            "Sube tus documentos (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader_key"
        )

        text_input = st.text_area(
            "O pega el texto directamente aquí:",
            height=200,
            placeholder="Pega tu texto aquí (ej. un artículo, feedback de clientes, etc.)",
            key="text_area_key"
        )

        # Crear un hash de los contenidos para detectar cambios en los datos de entrada
        # Necesitamos leer los archivos aquí para el hash, y luego re-apuntar para la función
        raw_uploaded_contents = []
        if uploaded_files:
            for f in uploaded_files:
                raw_uploaded_contents.append(f.read())
                f.seek(0) # Resetear el puntero del archivo para que pueda ser leído de nuevo en get_document_text

        current_data_hash = hash((
            tuple(raw_uploaded_contents),
            text_input
        ))
        
        # Botón para procesar los datos
        # El procesamiento se activa si se pulsa el botón O si los datos han cambiado y no se han procesado aún
        if st.button("Procesar Datos", key="process_button"): # Siempre procesar si se pulsa el botón
            should_process = True
        elif (uploaded_files or text_input) and current_data_hash != st.session_state.last_processed_data_hash:
            # Procesar automáticamente si hay datos y han cambiado
            should_process = True
        else:
            should_process = False

        if should_process:
            raw_text = ""
            if uploaded_files:
                raw_text += get_document_text(uploaded_files)
            if text_input:
                raw_text += text_input

            if raw_text:
                st.session_state.messages = [] # Limpia el chat al procesar nuevos datos
                st.session_state.conversation = None # Resetea la cadena de conversación (necesario si cambia la base de conocimiento)
                
                text_chunks = get_text_chunks(raw_text)
                if text_chunks: # Asegurarse de que haya chunks antes de crear el vectorstore
                    vectorstore = get_vectorstore(text_chunks, GOOGLE_API_KEY)
                    st.session_state.conversation = get_conversation_chain(vectorstore, GOOGLE_API_KEY)
                    st.session_state.last_processed_data_hash = current_data_hash # Actualiza el hash de lo último procesado
                    st.success("¡Datos procesados! Ahora puedes empezar a preguntar.")
                else:
                    st.warning("No se pudo extraer texto procesable de los documentos/texto proporcionados. Asegúrate de que los archivos contengan texto o que el formato sea correcto.")
            else:
                st.warning("Por favor, sube al menos un archivo o pega texto antes de procesar.")
        elif st.session_state.last_processed_data_hash is not None and current_data_hash == st.session_state.last_processed_data_hash:
             st.info("Los datos actuales ya han sido procesados. Modifica los datos de entrada o haz clic en 'Procesar Datos' para forzar el reprocesamiento.")
        
        st.markdown("---")
        st.info("Para procesar nuevos documentos/texto, simplemente actualiza los datos de entrada (sube nuevos archivos o modifica el texto) y haz clic en 'Procesar Datos'. El chat se reseteará.")
        st.info("Para reiniciar completamente la aplicación (y el chat/datos), puedes actualizar la página web.")


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

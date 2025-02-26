import os
import json
import streamlit as st
import chromadb
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# 🔑 Configuration de l'API Groq
API_KEY = "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"
if not API_KEY:
    raise ValueError("La clé API GROQ_API_KEY n'est pas définie.")

# 🟢 Initialisation de ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="seatech_chunks")

# 📂 Chargement des données JSON
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json("donnees_seatech.json")
chunks = [entry['markdown'] for entry in data]  # Assurez-vous que la clé 'markdown' existe

# 🔄 Ajout des documents à ChromaDB
collection.upsert(
    documents=chunks,
    ids=[str(i) for i in range(len(chunks))]
)

# 📜 Création du modèle de réponse avec historique
template = """
Tu es un assistant intelligent et utile. Réponds aux questions de l'utilisateur en français de manière claire et concise.

**Historique de la conversation :**  
{history}

**Nouvelle question :** {question}

📌 **Documents pertinents :**  
{documents}

💡 **Réponse :**
"""

prompt = PromptTemplate(template=template, input_variables=["history", "question", "documents"])
llm = ChatGroq(api_key=API_KEY, temperature=0, model="llama3-70b-8192")
chain = prompt | llm

# 🎨 **Interface Streamlit**
st.set_page_config(page_title="Chatbot Seatech", page_icon="💬", layout="wide")

# 📌 **Ajout du logo et du titre**
st.sidebar.image("seatech.png", width=250)
st.markdown("<h1 style='text-align: center; color: #0066cc;'>🔹 Chatbot Seatech 🔹</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Posez votre question sur l'école et obtenez une réponse instantanée !</p>", unsafe_allow_html=True)

# 🔄 **Initialisation de l'historique de la conversation**
if "history" not in st.session_state:
    st.session_state.history = []

# 📝 **Champ de saisie (centré)**
st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
question = st.text_input("💬 Posez votre question ici :", placeholder="Exemple : Admission sur dossier", key="user_question")
st.markdown("</div>", unsafe_allow_html=True)

if st.button("🔍 Rechercher", use_container_width=True):
    if question:
        if "seatech" not in question.lower():
            st.warning("⚠️ Ce chatbot est conçu uniquement pour répondre aux questions sur Seatech.")
        else:
            with st.spinner("🔄 Recherche en cours..."):
                try:
                    # 🔎 Recherche des documents pertinents
                    results = collection.query(query_texts=[question], n_results=3)
                    
                    if not results['documents']:
                        st.error("Aucun document pertinent trouvé. Veuillez reformuler votre question.")
                    else:
                        retrieved_docs = "\n".join(results['documents'][0])

                        # 🔥 Génération de la réponse avec historique
                        history_text = "\n".join(st.session_state.history[-5:])  # Garde les 5 derniers échanges
                        response = chain.invoke({"history": history_text, "question": question, "documents": retrieved_docs})

                        # 📝 **Mise à jour de l'historique**
                        st.session_state.history.append(f"👤 **Vous :** {question}")
                        st.session_state.history.append(f"<span style='color: green;'>🤖 **Chatbot :** {response.content}</span>")

                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de la recherche : {e}")

# 📌 **Affichage de l'historique de la conversation**
st.markdown("### 💬 Historique de la conversation")
for message in st.session_state.history:
    st.markdown(message, unsafe_allow_html=True)

# 📌 **Pied de page**
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2025 Seatech AI | Développé en Python</p>", unsafe_allow_html=True)


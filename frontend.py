import streamlit as st
import requests

# ✅ FastAPI Backend URL
API_URL = "https://assignment-499w.onrender.com//chat/"

st.title("Titanic AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display chat history
for message in st.session_state.messages:
    role = "🤖" if message["role"] == "assistant" else "👤"
    st.markdown(f"**{role}:** {message['content']}")

# ✅ User Input
user_input = st.text_input("You:")

if st.button("Send") and user_input:
    response = requests.post(API_URL, json={"message": user_input}).json()
    
    bot_reply = response.get("response", "")

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    st.markdown(f"**🤖 AI:** {bot_reply}")

    st.rerun()

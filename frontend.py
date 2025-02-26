import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

# FastAPI Backend URL
API_URL = "https://assignment-499w.onrender.com//chat/"

st.title("Titanic AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = "🤖" if message["role"] == "assistant" else "👤"
    st.markdown(f"**{role}:** {message['content']}")
    
    # Display histogram if it exists
    if "histogram" in message:
        try:
            # Decode the base64 image
            image_data = base64.b64decode(message["histogram"])
            image = Image.open(BytesIO(image_data))
            st.image(image, caption=f"Histogram for {message['column_name']}")
        except Exception as e:
            st.error(f"Error displaying histogram: {e}")

# User Input
user_input = st.text_input("You:")

if st.button("Send") and user_input:
    response = requests.post(API_URL, json={"message": user_input}).json()
    
    bot_reply = response.get("response", "")
    histogram_image = response.get("histogram", None)

    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Append bot response to chat history
    if histogram_image:
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_reply,
            "histogram": histogram_image,
            "column_name": user_input.split()[-1]  # Extract column name from user input
        })
    else:
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Display bot response
    st.markdown(f"**🤖 AI:** {bot_reply}")
    
    # Rerun to update the chat history
    st.rerun()






import streamlit as st
from pathlib import Path

from framework.framework import AIFramework


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Framework",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# Google Authentication
# ============================================================

if not st.user.is_logged_in:

    st.title("🤖 AI Framework")

    st.write(
        "Please sign in with your Google account to continue."
    )

    if st.button("Sign in with Google"):

        st.login()

    st.stop()


# ============================================================
# Authenticated Google User
# ============================================================

authenticated_email = st.user.email.strip().lower()


# ============================================================
# User Documents Directory
# ============================================================

# Project root
#     └── rag
#         └── docs
#             └── <user Gmail>
#
# Example:
#     rag/docs/aashutoshsr27@gmail.com/

user_docs_dir = (
    Path("rag")
    / "docs"
    / authenticated_email
)

# Create the directory if it doesn't exist
user_docs_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Header
# ============================================================

st.title("🤖 AI Framework")

st.caption(
    f"Signed in as: {authenticated_email}"
)


# ============================================================
# Add Documents
# ============================================================

st.subheader("📄 Add Documents")

uploaded_file = st.file_uploader(
    "Upload a text document",
    type=["txt"],
    help="Only .txt files are supported."
)


if uploaded_file is not None:

    file_path = user_docs_dir / uploaded_file.name

    # ----------------------------------------
    # Save uploaded file
    # ----------------------------------------

    if not file_path.exists():

        with open(file_path, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )

        st.success(
            f"Uploaded '{uploaded_file.name}' successfully."
        )

    else:

        st.info(
            f"'{uploaded_file.name}' is already in your documents."
        )


# ============================================================
# Initialize AI Framework
# ============================================================

if "framework" not in st.session_state:

    st.session_state.framework = AIFramework()


framework = st.session_state.framework


# ============================================================
# Chat History
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# Display Previous Messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# Chat Input
# ============================================================

question = st.chat_input(
    "Ask me anything..."
)


# ============================================================
# Process User Question
# ============================================================

if question:

    # ----------------------------------------
    # Display user message
    # ----------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ----------------------------------------
    # Process request
    # ----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = framework.process_request(
                    question,
                    google_email=authenticated_email
                )

                st.markdown(response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

            except Exception as e:

                st.exception(e)


# ============================================================
# Logout
# ============================================================

st.divider()

if st.button("Log out"):

    st.logout()
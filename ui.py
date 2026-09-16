import streamlit as st
from pathlib import Path

from framework.framework import AIFramework


st.set_page_config(
    page_title="AI Framework",
    page_icon="🤖",
    layout="centered",
)


if not st.user.is_logged_in:
    st.title("🤖 AI Framework")
    st.write("Please sign in with your Google account to continue.")

    if st.button("Sign in with Google"):
        st.login()

    st.stop()


authenticated_email = st.user.email.strip().lower()


# User-specific RAG documents directory
user_docs_dir = (
    Path("rag")
    / "docs"
    / authenticated_email
)

user_docs_dir.mkdir(
    parents=True,
    exist_ok=True
)


# User-specific CSV directory
user_csv_dir = (
    Path("rag")
    / "csv"
    / authenticated_email
)

user_csv_dir.mkdir(
    parents=True,
    exist_ok=True
)


st.title("🤖 AI Framework")
st.caption(f"Signed in as: {authenticated_email}")


# File upload
st.subheader("📄 Add Documents / CSV Files")

uploaded_file = st.file_uploader(
    "Upload a document or CSV file",
    type=["txt", "csv"],
    help=(
        "Upload .txt files for RAG or .csv files "
        "for data analysis."
    )
)


if uploaded_file is not None:

    file_extension = Path(
        uploaded_file.name
    ).suffix.lower()


    # TXT → RAG documents
    if file_extension == ".txt":

        file_path = (
            user_docs_dir
            / uploaded_file.name
        )

        if not file_path.exists():

            with open(
                file_path,
                "wb"
            ) as file:
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


    # CSV → Data analysis
    elif file_extension == ".csv":

        file_path = (
            user_csv_dir
            / uploaded_file.name
        )

        if not file_path.exists():

            with open(
                file_path,
                "wb"
            ) as file:
                file.write(
                    uploaded_file.getbuffer()
                )

            st.success(
                f"Uploaded '{uploaded_file.name}' successfully "
                "for data analysis."
            )

        else:
            st.info(
                f"'{uploaded_file.name}' is already available "
                "for data analysis."
            )


# Initialize framework
if "framework" not in st.session_state:
    st.session_state.framework = AIFramework()


framework = st.session_state.framework


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input("Ask me anything...")


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


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
                        "content": response
                    }
                )

            except Exception as e:
                st.exception(e)


st.divider()


if st.button("Log out"):
    st.logout()
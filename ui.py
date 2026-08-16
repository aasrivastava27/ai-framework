import streamlit as st

from framework.framework import AIFramework

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Framework",
    page_icon="🤖",
    layout="centered",
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 AI Framework")

st.caption(
    "LLM • RAG • CrewAI • Gmail • Google Calendar"
)

st.divider()

# --------------------------------------------------
# Google account
# --------------------------------------------------

google_email = st.text_input(
    "Google Email",
    placeholder="Enter your Google email",
)

# --------------------------------------------------
# Initialize framework
# --------------------------------------------------

if google_email:

    if (
        "framework" not in st.session_state
        or st.session_state.get("google_email") != google_email
    ):

        st.session_state.google_email = google_email
        st.session_state.framework = AIFramework()

# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# User input
# --------------------------------------------------

question = st.chat_input(
    "What would you like me to do?"
)

# --------------------------------------------------
# Process request
# --------------------------------------------------

if question:

    if not google_email:

        st.warning(
            "Please enter your Google email first."
        )

        st.stop()

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Execute your existing AI framework

    with st.chat_message("assistant"):

        with st.spinner(
            "AI Framework is working..."
        ):

            try:

                response = (
                    st.session_state.framework
                    .process_request(question, google_email)
                )

                response = str(response)

                st.markdown(response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

            except Exception as e:

                error_message = (
                    f"Something went wrong:\n\n"
                    f"`{str(e)}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
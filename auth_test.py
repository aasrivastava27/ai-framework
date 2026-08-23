import streamlit as st

st.title("Google Authentication Test")


if not st.user.is_logged_in:

    st.write("You are not logged in.")

    if st.button("Sign in with Google"):
        st.login()

else:

    st.success("You are logged in!")

    st.write("Google email:")
    st.write(st.user.email)

    if st.button("Log out"):
        st.logout()
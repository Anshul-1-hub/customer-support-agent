from graph.agent import app
import streamlit as st

st.title("ResolveAI")
ticket = st.text_area("Enter your ticket:", placeholder="Write your ticket here...")
submitted = st.button("Submit")               # submitted becomes true when the button is clicked

if ticket and submitted:
    state={
        "ticket": ticket,
    }
    
    result = app.invoke(state)
    st.write(result["response"])
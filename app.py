import streamlit as st

from knowledge_loader import (
    pdf_agent,
)


# Function to ask a question to the agent and get an answer; wrapper that hides the Agno API details from app.py.
def ask(agent: pdf_agent, question: str) -> str:
     # run() performs the full RAG loop: retrieve top-k from Chroma -> call LLM -> response object
    return agent.run(question, stream=False).content 


# Streamlit UI
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚") # this i don't see in the screenshot
st.title("📚 PDF Knowledge Assistant")


# Keeps the agent in session so there is no need to rebuild the knowledge base on every rerun
if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):   # show loading while building the agent
        st.session_state["agent"] = pdf_agent() # PDFs -> KB -> agent (RAG)


# Input area where the user types the question
question = st.text_area(
    "Ask a question (any language):",
    value="What is supervised learning?", #example, the user can change it
    help="The agent will retrieve from your PDF knowledge base and cite sources with page numbers."
)

# Button to trigger the query instead of running automatically
ask_btn = st.button("Ask", type="primary", use_container_width=True)

# If the button is pressed and the text is not empty: show spinner, call ask(), and display the answer
if ask_btn and question.strip():
    with st.spinner("Thinking..."):
        answer = ask(st.session_state["agent"], question.strip())
    st.markdown("---")
    st.subheader("Answer")
    st.markdown(answer if answer else "_(No output)_")

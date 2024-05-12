import embeddings
import chroma_database
from query_data import QueryDataBase
import streamlit as st
import numpy as np
import pprint


def query_rag(question):
    retriever = QueryDataBase()

    response = retriever.query_rag(question)

    return response



def main():
    response = None
    st.title("Sperimenting RAG...")

    with st.container():
        input = st.chat_input("Say something")

        if input != None:
            st.write(input)
            print(input)
            with st.spinner(text="In progress..."):
                response, sources = query_rag(input)
            st.success("Done")

    # with st.form('my_form'):
    #     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    #     submitted = st.form_submit_button('Submit')

    #     if submitted:
    #         print(text)
    if response != None:
        with st.chat_message("AI"):
            st.write(response.content)





if __name__ == "__main__":
    main()
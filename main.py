import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio



def create_streamlit_app(llm,portfolio):
    st.title("Cold Email Generator")
    url_input = st.text_input("Enter the URL of the webpage to extract content from:")
    submit_button = st.button("Generate Cold Email")
    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            docs=loader.load().pop().page_content
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(docs)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email,language='markdown')
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio)

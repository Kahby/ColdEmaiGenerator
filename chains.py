import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import re

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="qwen/qwen3-32b")

    def clean_text(text):
        # 1. Remove HTML tags (if using a raw loader)
        text = re.sub(r'<[^>]*?>', '', text)
        
        # 2. Remove URLS, special characters, and excessive whitespace
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces/newlines with single space
        
        # 3. TRUNCATE THE TEXT (The most important fix)
        # Most job descriptions don't need more than 3000-5000 characters to understand.
        # If the text is huge, we chop off the rest to save the context window.
        max_length = 6000 
        if len(text) > max_length:
            text = text[:max_length]
            
        return text

    def extract_jobs(self, clean_text(cleaned_text)):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `companyname`,`role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            In EMAIL STARTING instead of writing "Dear [Client's Name],", Write Dear/Hi Hiring Manager of the company with company name mentioned in the job description.
            You are Amritanshu, a Student at Vit Bhopal University. 
            Specializing in Web Development:HTML/CSS/JAVASCRIPT/React/TypeScript and Data Science,ML and DL.
            You are writing a cold email to the client regarding the job mentioned above describing your capability
            Also add the most relevant ones from the following links to showcase Amritanshu's portfolio: {link_list}
            Remember Amritanshu, a Student at Vit Bhopal University. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        cleaned_content = res.content.replace(r'<think>', '').replace(r'</think>', '')
        return cleaned_content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))

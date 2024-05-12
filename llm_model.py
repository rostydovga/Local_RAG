from langchain_groq import ChatGroq
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


class LLM_Model():
    def __init__(self) -> None:
        self.model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.environ.get('GROQ_API_KEY'))

    def get_model(self):
        return self.model
    
    def get_chain_extraction_info(self):
        template_extract_user_info = """
        You are an information extractor tasked with identifying and extracting patient identification details from a set of documents. 

        Given the following documents, your goal is to extract only the patient identification information. 
        {{docs}}

        The output must be only one JSON object and no further information, the JSON object must have the following keys: 'ID', 'Age', 'Gender', 'Profession'.

        """

        prompt_template = PromptTemplate(template=template_extract_user_info, input_variables=['docs'], template_format='jinja2')

        model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.environ.get('GROQ_API_KEY'))

        chain = prompt_template | model | JsonOutputParser()

        return chain
                

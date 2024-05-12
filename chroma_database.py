from embeddings import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from llm_model import LLM_Model
import json

DATA_PATH = 'data/'
CHROMA_PATH = "chroma/"

class ChromaDataBase():

    def __init__(self) -> None:
        self.model_class = LLM_Model()

    def load_documents(self):
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()

    def extract_patient_info():
        pass

    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 80,
            length_function=len,
            is_separator_regex=False
        )

        return text_splitter.split_documents(documents)


    # Function needed for the loading of new elements to the existing db
    def calculate_chunk_ids(self, chunks):

        # This will create IDs like "<directory>/<file_name>.pdf:<page_number>:<chunk_id>"
        # data/Med_record_patient_2:1:2

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks



    def add_to_chroma(self, chunks: list[Document]):
        # Load the existing database.
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )

        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("No new documents to add")


    def add_infos_to_chunks(self, chunks):
        
        def get_sources(list_chunks):
            return set([c.metadata['source'] for c in list_chunks])


        for s in get_sources(chunks):
            list_chunks_s = [c for c in chunks if c.metadata['source'] == s]
            list_docs = [c for c in chunks if c.metadata['source'] == s and c.metadata['page'] == 0]
            # use those docs to extract the identification infos of the patient
            patient_id = self.model_class.get_chain_extraction_info().invoke({'docs':list_docs})
            
            # add information to chunks
            for c_s in list_chunks_s:
                c_s.page_content += f"\nPatient Info: {json.dumps(patient_id)}"

        return chunks

        



    def upload_docs(self):     
        # Create (or update) the data store.
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        # add infos to chunks
        chunks = self.add_infos_to_chunks(chunks)
        self.add_to_chroma(chunks)


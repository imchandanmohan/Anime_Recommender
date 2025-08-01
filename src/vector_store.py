from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import logging
from utils.custom_exception import CustomException

from dotenv import load_dotenv
load_dotenv()


class VectorStoreBuilder:
    """
    Builds and manages a Chroma vector store from a CSV file using HuggingFace embeddings.

    Attributes:
        csv_path (str): Path to the input CSV file.
        persist_dir (str): Directory where Chroma DB will be saved.
        embedding (HuggingFaceEmbeddings): Embedding model used for vectorization.
    """
    
    csv_path: str
    persist_dir: str
    embedding: HuggingFaceEmbeddings

    def __init__(self, csv_path: str, persist_dir: str = "chroma_db") -> None:
        """
        Initializes the vector store builder.

        Args:
            csv_path (str): Path to the CSV file containing documents.
            persist_dir (str, optional): Directory to persist Chroma DB. Defaults to "chroma_db".
        """
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.info(f"VectorStoreBuilder initialized with CSV: {csv_path} and persist_dir: {persist_dir}")

    def build_and_save_vectorstore(self) -> None:
        """
        Loads documents from CSV, splits them into chunks, builds a vector store, and persists it.
        
        Raises:
            CustomException: If any step fails.
        """
        try:
            logging.info("Loading documents from CSV...")
            loader = CSVLoader(
                file_path=self.csv_path,
                encoding='utf-8',
                metadata_columns=[]
            )
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents from CSV.")

            logging.info("Splitting documents into chunks...")
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
            texts = splitter.split_documents(documents)
            logging.info(f"Split into {len(texts)} chunks.")

            logging.info("Creating and saving Chroma vector store...")
            db = Chroma.from_documents(texts, self.embedding, persist_directory=self.persist_dir)
            db.persist()
            logging.info(f"Vector store saved to directory: {self.persist_dir}")

        except Exception as e:
            logging.exception("Failed to build and save vector store.")
            raise CustomException("Vector store creation failed", e)

    def load_vector_store(self) -> Chroma:
        """
        Loads the persisted Chroma vector store.

        Returns:
            Chroma: Loaded vector store.

        Raises:
            CustomException: If loading fails.
        """
        try:
            logging.info(f"Attempting to load vector store from: {self.persist_dir}")
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding
            )
        except Exception as e:
            logging.exception("Failed to load vector store.")
            raise CustomException("Failed to load vector store", e)

    def documents(self) -> list:
        """
        Loads raw documents from the CSV file.

        Returns:
            list: List of documents.

        Raises:
            CustomException: If document loading fails.
        """
        try:
            logging.info(f"Loading documents directly from: {self.csv_path}")
            loader = CSVLoader(file_path=self.csv_path, encoding='utf-8', metadata_columns=[])
            return loader.load()
        except Exception as e:
            logging.exception("Failed to load documents from CSV.")
            raise CustomException("Document loading failed", e)



if __name__ == "__main__":
    try:
        builder = VectorStoreBuilder(csv_path="data/anime_with_synopsis_processed.csv")
        builder.build_and_save_vectorstore()

        store = builder.load_vector_store()
        logging.info("Vector store loaded successfully.")

        docs = builder.documents()
        logging.info(f"Loaded {len(docs)} raw documents from CSV.")

    except CustomException as ce:
        logging.error(f"Custom exception occurred: {ce}")
    except Exception as e:
        logging.exception("Unexpected error in vector store build process.")
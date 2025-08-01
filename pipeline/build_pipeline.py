from src.data_loader import AnimeDataloader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()

logger = get_logger()

def main():

    try:
        logger.info("Starting to build pipeline...")

        loader = AnimeDataloader("data/anime_with_synopsis.csv","data/anime_with_synopsis_processed.csv")
        processed_csv = loader.load_and_process()

        logger.info("Data loded and processed...")

        vector_builder = VectorStoreBuilder(processed_csv)
        vector_builder.build_and_save_vectorstore()

        logger.info("Vector store built successfully...")

        logger.info("Pipeline built successfully")

    except Exception as e:
            logger.error(f" Failed to excecute pipeline {str(e)}")
            raise CustomException("Error during pipeline excecute", e)



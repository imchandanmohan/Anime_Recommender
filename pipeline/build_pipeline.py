from dotenv import load_dotenv

from src.data_loader import AnimeDataloader
from src.vector_store import VectorStoreBuilder
from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()
logger = get_logger(__name__)


def main() -> None:
    """
    Entry point for building the anime recommendation pipeline.
    
    This script loads and processes raw anime data,
    builds the vector store, and persists it for retrieval.
    
    Raises:
        CustomException: If any step in the pipeline fails.
    """
    try:
        logger.info("Starting anime recommendation pipeline build...")

        # Step 1: Load and process the data
        loader = AnimeDataloader(
            original_csv="data/anime_with_synopsis.csv",
            processed_csv="data/anime_with_synopsis_processed.csv"
        )
        processed_csv_path = loader.load_and_process()
        logger.info("Data loaded and processed successfully: %s", processed_csv_path)

        # Step 2: Build and save vector store
        vector_builder = VectorStoreBuilder(csv_path=processed_csv_path)
        vector_builder.build_and_save_vectorstore()
        logger.info("Vector store built and saved successfully.")

        logger.info("Anime recommendation pipeline built successfully.")

    except Exception as e:
        logger.exception("Pipeline execution failed.")
        raise CustomException("Error occurred during pipeline execution.", e)


if __name__ == "__main__":
    main()

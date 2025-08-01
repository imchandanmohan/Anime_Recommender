from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class AnimeRecommendationPipeline:
    """
    A pipeline class to generate anime recommendations.

    This class initializes the vector store, loads the retriever,
    and delegates recommendation generation to AnimeRecommender.
    """

    def __init__(self, persist_dir: str = "chroma_db") -> None:
        """
        Initializes the AnimeRecommendationPipeline by setting up the vector store
        and the anime recommendation engine.

        Args:
            persist_dir (str): Directory where the vector store is persisted.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            logger.info("Initializing AnimeRecommendationPipeline with persist_dir='%s'", persist_dir)

            vector_builder = VectorStoreBuilder(
                csv_path="",  # Assuming CSV path empty means loading from persist_dir
                persist_dir=persist_dir
            )
            logger.info("VectorStoreBuilder initialized with persist_dir='%s'.", persist_dir)

            retriever = vector_builder.load_vector_store().as_retriever()
            logger.info("Vector store loaded and retriever created.")

            self.recommender = AnimeRecommender(
                retriever=retriever,
                api_key=str(GROQ_API_KEY),
                model_name=MODEL_NAME
            )
            logger.info("AnimeRecommender initialized successfully. Pipeline is ready.")

        except Exception as e:
            error_msg = (
                f"Pipeline initialization failed for persist_dir='{persist_dir}'. "
                f"Make sure the vector store exists and is not corrupted. "
                f"Original error: {str(e)}"
            )
            logger.exception(error_msg)
            raise CustomException(error_msg, e)

    def recommend(self, query: str) -> str:
        """
        Generates an anime recommendation for the given query.

        Args:
            query (str): A user-provided query, e.g., "I liked Attack on Titan".

        Returns:
            str: A recommendation string from the model.

        Raises:
            CustomException: If recommendation generation fails.
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty or whitespace.")

            logger.info("Received query for recommendation: '%s'", query)

            recommendation = self.recommender.get_recommendation(query=query)

            logger.info("Anime recommendation generated successfully.")
            return recommendation

        except ValueError as ve:
            # For validation errors, give clear feedback
            error_msg = f"Invalid query provided: '{query}'. Query must be a non-empty string."
            logger.error(error_msg)
            raise CustomException(error_msg, ve)

        except Exception as e:
            error_msg = (
                f"Failed to generate recommendation for query: '{query}'. "
                f"Check if the retriever and LLM are properly initialized. "
                f"Original error: {str(e)}"
            )
            logger.exception(error_msg)
            raise CustomException(error_msg, e)

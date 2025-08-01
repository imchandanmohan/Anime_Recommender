from langchain.prompts import PromptTemplate
from utils.logger import logging
from utils.custom_exception import CustomException

def get_anime_prompt() -> PromptTemplate:
    """
    Creates a prompt template for the anime recommender system.

    Returns:
        PromptTemplate: The prompt template configured for anime recommendations.

    Raises:
        CustomException: If the prompt template fails to initialize.
    """
    try:
        logging.info("Initializing anime recommendation prompt template...")

        template = """
            You are an expert anime recommender. Your job is to help users find the perfect anime based on their preferences.

            Using the following context, provide a detailed and engaging response to the user's question.

            For each question, suggest exactly three anime titles. For each recommendation, include:
            1. The anime title.
            2. A concise plot summary (2-3 sentences).
            3. A clear explanation of why this anime matches the user's preferences.

            Present your recommendations in a numbered list format for easy reading.

            If you don't know the answer, respond honestly by saying you don't know — do not fabricate any information.

            Context:
            {context}

            User's question:
            {question}

            Your well-structured response:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        logging.info("Prompt template successfully initialized.")
        return prompt

    except Exception as e:
        logging.exception("Failed to create the anime prompt template.")
        raise CustomException("Error initializing anime prompt template", e)
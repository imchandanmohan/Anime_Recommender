import pytest
from unittest.mock import MagicMock
from pipeline.pipeline import AnimeRecommendationPipeline
from utils.custom_exception import CustomException

def test_recommend_success(mocker):
    # Mock VectorStoreBuilder.load_vector_store().as_retriever() chain
    mock_retriever = MagicMock(name="retriever")

    mock_vector_builder = mocker.patch("pipeline.pipeline.VectorStoreBuilder")
    mock_vector_builder.return_value.load_vector_store.return_value.as_retriever.return_value = mock_retriever

    # Mock AnimeRecommender to avoid real LLM calls
    mock_recommender_instance = MagicMock()
    mock_recommender_instance.get_recommendation.return_value = "Naruto is recommended"

    mocker.patch(
        "src.pipeline.AnimeRecommender",
        return_value=mock_recommender_instance
    )

    # Initialize pipeline (will use mocks)
    pipeline = AnimeRecommendationPipeline(persist_dir="test_dir")

    # Call recommend and verify it returns mocked response
    result = pipeline.recommend("I like action anime")

    assert result == "Naruto is recommended"
    mock_recommender_instance.get_recommendation.assert_called_once_with(query="I like action anime")

def test_recommend_empty_query():
    pipeline = AnimeRecommendationPipeline.__new__(AnimeRecommendationPipeline)  # bypass __init__
    pipeline.recommender = MagicMock()

    with pytest.raises(CustomException) as excinfo:
        pipeline.recommend("  ")  # empty string with whitespace

    assert "Invalid query provided" in str(excinfo.value)

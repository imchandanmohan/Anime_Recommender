import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv

# Set up Streamlit page configuration
st.set_page_config(page_title="Anime Recommender", layout="wide")

# Load environment variables
load_dotenv()

@st.cache_resource
def init_pipeline():
    """
    Initializes and caches the AnimeRecommendationPipeline instance
    to avoid reloading the model on every interaction.
    """
    return AnimeRecommendationPipeline()

# Initialize pipeline once per session
pipeline = init_pipeline()

# App title
st.title("Anime Recommender System")

# Input field for user preferences
query = st.text_input("Enter your anime preferences (e.g., light-hearted anime with school settings)")

# Generate recommendation if query is submitted
if query:
    with st.spinner("Fetching recommendations for you..."):
        response = pipeline.recommend(query)
        st.markdown("### Recommendations")
        st.write(response)

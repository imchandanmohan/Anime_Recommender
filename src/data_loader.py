import pandas as pd
from utils.logger import logging
from utils.custom_exception import CustomException

class AnimeDataloader:

    """
    Loads, validates, and processes anime data from a CSV file.

    This class reads an input CSV file containing anime metadata,
    checks for required columns, combines selected fields into a 
    new 'combined_info' column, and saves the result to a new CSV.

    Attributes:
        original_csv (str): Path to the input CSV file.
        processed_csv (str): Path where the processed CSV will be saved.
    """

    original_csv: str
    processed_csv: str

    def __init__(self, original_csv: str, processed_csv: str) -> None:

        """
        Initializes the AnimeDataloader with input and output CSV paths.

        Args:
            original_csv (str): Path to the raw anime dataset CSV file.
            processed_csv (str): Path where the cleaned and processed CSV will be saved.
        """

        self.original_csv = original_csv
        self.processed_csv = processed_csv

    def load_and_process(self) -> str:
        
        """
        Loads a CSV file, cleans missing data, and creates a 'combined_info' column.

        Returns:
            str: The path to the saved processed CSV file.

        Raises:
            CustomException: If loading, processing, or saving fails.
        """


        logging.info(f"Starting to load data from: {self.original_csv}")

        try:
            df: pd.DataFrame = pd.read_csv(
                self.original_csv,
                encoding='utf-8',
                on_bad_lines='skip'
            ).dropna()

            logging.info("Successfully loaded and cleaned CSV data.")

            # Check required columns
            required_cols = {'Name', 'Genres', 'sypnopsis'}
            missing = required_cols - set(df.columns)

            if missing:
                logging.error(f"Missing columns in CSV: {missing}")
                raise ValueError(f"Missing required columns: {missing}")

            logging.info("All required columns are present.")

            # Combine info
            try:
                df['combined_info'] = (
                    "Title: " + df['Name'].astype(str) +
                    " .. Overview: " + df['sypnopsis'].astype(str) +
                    " Genres: " + df['Genres'].astype(str)
                )
            except KeyError as key_err:
                logging.exception("Key error while creating 'combined_info' column.")
                raise CustomException("Error during combined_info generation", key_err)

            # Save processed data
            df[['combined_info']].to_csv(self.processed_csv, index=False, encoding='utf-8')
            logging.info(f"Processed data saved to: {self.processed_csv}")

            return self.processed_csv
        
        except FileNotFoundError as fnf_err:
            logging.error(f"File not found: {fnf_err}")
            raise CustomException("Input CSV file not found", fnf_err)

        except Exception as e:
            logging.exception("An error occurred during data loading or processing.")
            raise CustomException("Failed to load and process anime data", e)
        

if __name__ == "__main__":
    from utils.logger import get_logger
    from utils.custom_exception import CustomException

    logger = get_logger(__name__)

    try:
        original_csv_path = "data/anime_with_synopsis.csv"       # Replace with your actual input CSV path
        processed_csv_path = "data/anime_with_synopsis_processed.csv"  # Replace with desired output path

        dataloader = AnimeDataloader(original_csv=original_csv_path, processed_csv=processed_csv_path)
        output_path = dataloader.load_and_process()

        logger.info(f"Data successfully processed and saved to: {output_path}")

    except CustomException as ce:
        logger.error(f"CustomException caught: {ce}")
    except Exception as e:
        logger.exception("An unexpected error occurred during test run.")
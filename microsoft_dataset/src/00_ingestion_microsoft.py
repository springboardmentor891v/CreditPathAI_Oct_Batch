import os
import pandas as pd

class DataIngestion:
    """
    Universal ingestion engine used for ANY dataset.
    Reads CSV, removes duplicates, prints summary, saves interim cleaned file.
    """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self):
        """Loads CSV using best-guess encoding."""
        try:
            df = pd.read_csv(self.input_path)
        except UnicodeDecodeError:
            df = pd.read_csv(self.input_path, encoding='latin1')
        except Exception as e:
            raise RuntimeError(f"Error reading CSV: {e}")

        print(f"\n📌 Loaded data: {self.input_path}")
        print("Shape:", df.shape)
        return df

    def summarize(self, df):
        """Prints useful dataset summary."""
        print("\n🔍 COLUMN LIST:")
        print(list(df.columns))

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        print("\n❗ Missing values:")
        print(missing if not missing.empty else "No missing values")

    def remove_duplicates(self, df):
        """Removes duplicate rows."""
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        print(f"\n🧹 Removed {before - after} duplicate rows.")
        return df

    def save_data(self, df):
        """Saves cleaned data to interim folder."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"\n💾 Cleaned data saved to: {self.output_path}")

    def run(self):
        """Runs entire ingestion pipeline."""
        df = self.load_data()
        self.summarize(df)
        df = self.remove_duplicates(df)
        self.save_data(df)
        return df


# ---------------------------------
# EXAMPLE USAGE
# ---------------------------------
if __name__ == "__main__":
    ingestion = DataIngestion(
        input_path=r"data/raw/Kaggle_loan_default.csv",
        output_path="data/interim/loan_data_clean.csv"
    )
    df_cleaned = ingestion.run()

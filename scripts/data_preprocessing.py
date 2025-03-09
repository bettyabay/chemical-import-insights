import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    """
    DataPreprocessor class for cleaning and analyzing Ethiopian import data from an Excel file.
    """

    def __init__(self, data_path="../data/raw/Import Data Mining.xlsx", logger=None):
        """
        Initializes the DataPreprocessor instance.

        Parameters:
        - data_path (str): Path to the Excel file containing the import data. Defaults to "../data/import_data.xlsx".
        - logger (logging.Logger): Optional logger for logging information and errors.
        """
        self.data_path = data_path
        self.logger = logger

    def load_data(self):
        """
        Loads data from an Excel file.

        Returns:
        - pd.DataFrame: DataFrame with loaded data.
        """
        try:
            data = pd.read_excel(self.data_path, engine='openpyxl')
            return data
        except Exception as e:
            self._log_error(f"Failed to load data: {e}")
            raise



    def inspect_data(self, data):
        """
        Inspects the data and provides a concise, tabular summary with full value counts.
        """
        try:
            inspection_results = []
            total_rows = len(data)

            for col in data.columns:
                unique_counts = len(data[col].unique())
                missing_count = data[col].isnull().sum()
                missing_percentage = (missing_count / total_rows) * 100
                value_counts = data[col].value_counts().to_dict() # get all value counts as a dictionary.

                data_type = str(data[col].dtype)
                data_category = "Numeric" if pd.api.types.is_numeric_dtype(data[col]) else "Categorical/Text"

                inspection_results.append({
                    "Column": col,
                    "Data Category": data_category,
                    "Data Type": data_type,
                    "Missing Values": missing_count,
                    "Missing (%)": round(missing_percentage, 2),
                    "Unique Values Count": unique_counts,
                    "Value Counts": value_counts, # Include full value counts.
                })

            summary_df = pd.DataFrame(inspection_results)
            print("Data Inspection Summary:")
            display(summary_df)
            self._log_info(f"Data inspection results:\n{summary_df.to_string()}")
            return summary_df

        except Exception as e:
            print(f"Error during data inspection: {e}")
            self._log_info(f"Error during data inspection: {e}")
            return None
        
        
    def plot_missing_values(self, data):
            """
            Visualizes the missing values in the dataset.
            """
            plt.figure(figsize=(12, 6))
            missing_values = data.isnull().sum()
            missing_values[missing_values > 0].plot(kind='bar', color='orange')
            plt.title('Missing Values per Column')
            plt.xlabel('Columns')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    

 
    def handle_missing_values(self, data, logger=None):
        """
        Handles missing values in chemical import data based on specified strategies,
        imputing numerical values with 0 and categorical values with 'Unknown'.
        """ 

        # Rename 'Reg. Date (Day/Mon/Year)' to 'reg date'
        if 'Reg. Date (Day/Mon/Year)' in data.columns:
            data.rename(columns={'Reg. Date (Day/Mon/Year)': 'Reg. date'}, inplace=True)
            if logger:
                logger.info("Renamed 'Reg. Date (Day/Mon/Year)' to 'Reg. date'.")

        # a. Unique Identifiers and Key Columns
        if 'BANK PERMIT NUMBER' in data.columns:
            data['BANK PERMIT NUMBER'].fillna("Unknown", inplace=True)
            if logger:
                logger.info("Imputed 'BANK PERMIT NUMBER' with 'Unknown'.")
        else:
            if logger:
                logger.warning("Column 'BANK PERMIT NUMBER' not found.")

        if 'TIN' in data.columns:
            data['TIN'].fillna("Unknown", inplace=True)
            if logger:
                logger.info("Imputed 'TIN' with 'Unknown'.")
        else:
            if logger:
                logger.warning("Column 'TIN' not found.")

        # b. Date Columns
        if 'Reg. date' in data.columns:
            data['Reg. date'].fillna("Unknown", inplace=True)
            if logger:
                logger.info("Imputed 'reg date' with 'Unknown'.")

        # c. Categorical Columns
        for col in ['Trader', 'Trader Address']:
            if col in data.columns:
                data[col].fillna("Unknown", inplace=True)
                if logger:
                    logger.info(f"Imputed '{col}' with 'Unknown'.")

        # d. Numerical Columns
        for col in ['CIF/FOB Value (ETB)', 'Total Tax (ETB)', 'Gross Wt./Net Wt. (Kg)', 'Duty/Excise/VAT']:
            if col in data.columns:
                data[col].fillna(0, inplace=True)
                if logger:
                    logger.info(f"Imputed '{col}' with 0.")

        # e. HS Code and HS Description
        if 'HS Code' in data.columns and 'HS Description' in data.columns:
            data = data.dropna(subset=['HS Code', 'HS Description'])
            if logger:
                logger.info("Dropped rows with missing 'HS Code' or 'HS Description'.")
        else:
            if logger:
                if 'HS Code' not in data.columns:
                    logger.warning("Column 'HS Code' not found.")
                if 'HS Description' not in data.columns:
                    logger.warning("Column 'HS Description' not found.")

        # Remove missing flag columns
        flag_columns = [col for col in data.columns if col.endswith('_missing_flag')]
        data = data.drop(columns=flag_columns, errors='ignore')

        if self.logger is None:
            print("Missing values handled as follows:")
            print(f"  Imputed 'BANK PERMIT NUMBER' and 'TIN' with 'Unknown'.")
            print(f"  Imputed 'reg date', 'Trader', and 'Trader Address' with 'Unknown'.")
            print(f"  Imputed numerical columns with 0.")
            print(f"  Dropped rows with missing 'HS Code' or 'HS Description'.")
            print(f"  Removed missing flag columns.")

        return data

       
    def convert_data_types(self, data, logger=None):
        """
        Converts data types of columns in a DataFrame, automatically detecting
        numerical and categorical columns.
        """

        if logger:
            logger.info("Starting automatic data type conversion.")

        # Convert 'reg date' to datetime (if applicable)
        if 'Reg. date' in data.columns:
            data['Reg. date'] = pd.to_datetime(data['Reg. date'], format='%d/%m/%Y', errors='coerce')
            if logger:
                logger.info("Converted 'reg date' to datetime.")

        # Automatically identify numerical columns and convert
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if logger:
                    logger.info(f"Converted '{col}' to numeric.")

        # Convert specified columns to integer
        int_cols = ['Item', 'HS Code', '# of packages', 'MoT Code']
        for col in int_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                if logger:
                    logger.info(f"Converted '{col}' to integer.")

        return data

   
    def clean_text_data(self, data, text_columns):
        """
        Cleans text data by normalizing text.

        Parameters:
        - data (pd.DataFrame): DataFrame containing data.
        - text_columns (list): List of text columns to clean.

        Returns:
        - pd.DataFrame: DataFrame with cleaned text data.
        """
        for col in text_columns:
            data[col] = data[col].str.upper().str.strip().replace(r'\s+', ' ', regex=True)
        self._log_info(f"Text data cleaned for columns: {text_columns}")
        return data

    def remove_duplicates_by_bank_permit(self, logger=None):
        """
        Removes duplicate rows based on the 'BANK PERMIT NUMBER' column.

        Args:
            logger (logging.Logger, optional): Logger for logging messages.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        if 'BANK PERMIT NUMBER' not in self.dataframe.columns:
            if logger:
                logger.warning("Column 'BANK PERMIT NUMBER' not found. Cannot remove duplicates.")
            print("Column 'BANK PERMIT NUMBER' not found. Cannot remove duplicates.")
            return self.dataframe

        before_count = len(self.dataframe)
        self.dataframe.drop_duplicates(subset='BANK PERMIT NUMBER', keep='first', inplace=True)
        after_count = len(self.dataframe)

        if logger:
            logger.info(f"Removed {before_count - after_count} duplicate rows based on 'BANK PERMIT NUMBER'.")

        print(f"Removed {before_count - after_count} duplicate rows based on 'BANK PERMIT NUMBER'.")
        return self.dataframe

    def detect_outliers(self, data, method="iqr", z_threshold=3):
        """
        Detects outliers in the data using either the IQR or Z-score method.

        Parameters:
        - data (pd.DataFrame): DataFrame containing data.
        - method (str): Outlier detection method ('iqr' or 'z_score'). Default is 'iqr'.
        - z_threshold (int): Z-score threshold for outliers. Default is 3.

        Returns:
        - pd.DataFrame: Boolean DataFrame indicating outliers.
        """
        outliers = pd.DataFrame(index=data.index)

        for col in data.select_dtypes(include=np.number).columns:
            if method == "z_score":
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers[col] = z_scores > z_threshold
            elif method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))

        self._log_info("Outliers detected using {} method.".format(method.capitalize()))
        return outliers

    def handle_outliers(self, data, outliers):
        """
        Handles detected outliers by replacing them with NaN for later filling.

        Parameters:
        - data (pd.DataFrame): DataFrame containing data.
        - outliers (pd.DataFrame): Boolean DataFrame indicating positions of outliers.

        Returns:
        - pd.DataFrame: DataFrame with outliers handled.
        """
        cleaned_data = data.copy()
        cleaned_data[outliers] = np.nan
        cleaned_data.interpolate(method="linear", inplace=True)
        cleaned_data.bfill(inplace=True)
        cleaned_data.ffill(inplace=True)

        self._log_info("Outliers handled by setting to NaN and filling with interpolation.")
        return cleaned_data

    def clean_data(self, data, critical_columns, text_columns):
        """
        Cleans the loaded data by detecting and handling missing values and outliers.

        Parameters:
        - data (pd.DataFrame): DataFrame containing data to be cleaned.
        - critical_columns (list): List of critical columns to check for missing values.
        - text_columns (list): List of text columns to clean.

        Returns:
        - pd.DataFrame: Cleaned DataFrame.
        """
        data = self.handle_missing_values(data, critical_columns)
        data = self.clean_text_data(data, text_columns)
        

        data_cleaned = self.handle_outliers(data, outliers)
        return data_cleaned

    def normalize_data(self, data):
        """
        Normalizes numeric columns using standard scaling.

        Parameters:
        - data (pd.DataFrame): DataFrame containing data to be normalized.

        Returns:
        - pd.DataFrame: DataFrame with normalized columns.
        """
        scaler = StandardScaler()
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        self._log_info("Data normalized using standard scaling.")
        return data

    def analyze_data(self, data):
        """
        Analyzes data by calculating basic statistics and checking for anomalies.

        Parameters:
        - data (pd.DataFrame): DataFrame containing data for analysis.

        Returns:
        - dict: Summary statistics including mean, median, standard deviation, and count of missing values.
        """
        analysis_results = {
            "mean": data.mean(),
            "median": data.median(),
            "std_dev": data.std(),
            "missing_values": data.isnull().sum()
        }
        self._log_info(f"Basic statistics calculated for data:\n{analysis_results}")
        return analysis_results

    def plot_outliers(self, data, outliers):
        """
        Plots box plots to visualize outliers in the data.

        Parameters:
        - data (pd.DataFrame): DataFrame containing data.
        - outliers (pd.DataFrame): Boolean DataFrame indicating outliers.
        """
        columns_with_outliers = [col for col in data.columns if col in outliers.columns and outliers[col].any()]

        if not columns_with_outliers:
            self._log_info("No outliers detected in any columns.")
            return

        num_plots = len(columns_with_outliers)
        grid_size = math.ceil(math.sqrt(num_plots))  # Calculate grid dimensions

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12 * grid_size, 4 * grid_size))
        
        # Flatten axes to make sure we can iterate over it regardless of grid size
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for i, col in enumerate(columns_with_outliers):
            ax = axes[i]
            ax.plot(data.index, data[col], label=col, color="skyblue")  # Time series line
            ax.scatter(data.index[outliers[col]], data[col][outliers[col]], 
                       color='red', s=20, label="Outliers")  # Outliers as red dots

            ax.set_title(f"{col} - Time Series with Outliers")
            ax.set_xlabel("Index")
            ax.set_ylabel(col)
            ax.legend()

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, data):
        """
        Plots a heatmap for the correlation matrix of numeric columns.
        """
        plt.figure(figsize=(12, 10))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()


    def _log_info(self, message):
        """Logs informational messages."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _log_error(self, message):
        """Logs error messages."""
        if self.logger:
            self.logger.error(message)
        else:
            print(message)
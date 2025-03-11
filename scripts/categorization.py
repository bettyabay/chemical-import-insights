import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

class HSCodeAnalyzer:
    def __init__(self, data):
        """
        Initialize the HSCodeAnalyzer with the dataset.
        
        :param data: DataFrame containing import data
        """
        self.data = data

    def filter_hs_code(self, data, hs_code):
        """
        Filter the DataFrame for a specific HS Code.
        
        :param data: DataFrame to filter
        :param hs_code: The HS Code to filter by (e.g., '3905')
        :return: Filtered DataFrame
        """
        filtered_data = data[data['HS Code'].astype(str).str.startswith(hs_code)]
        return filtered_data

    def analyze_sub_classifications(self, filtered_data):
        """
        Analyze sub-classifications within the filtered HS Code data.
        
        :param filtered_data: DataFrame filtered by HS Code
        :return: Series with counts of sub-classifications
        """
        sub_classification_counts = filtered_data['HS Code'].value_counts()
        return sub_classification_counts

    def visualize_hs_code_analysis(self, filtered_data):
        """
        Visualize trends in the HS Code imports.
        
        :param filtered_data: DataFrame containing filtered HS Code data
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=filtered_data, x='HS Code', order=filtered_data['HS Code'].value_counts().index)
        plt.title('Distribution of HS Code Imports (3905)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def perform_analysis(self, data, hs_code):
        """
        Perform the complete HS Code analysis workflow.
        
        :param hs_code: The HS Code to analyze (e.g., '3905')
        :return: Tuple of the filtered DataFrame and sub-classification analysis results
        """
        # Step 1: Filter data for the specified HS Code
        filtered_data = self.filter_hs_code(self.data, hs_code)

        # Step 2: Analyze sub-classifications
        sub_classification_analysis = self.analyze_sub_classifications(filtered_data)

        # Step 3: Visualize the analysis results
        self.visualize_hs_code_analysis(filtered_data)

        return filtered_data, sub_classification_analysis

class Categorizer:

    def __init__(self, data):
        """
        Initialize the Categorizer with HS Code and Description data.

        :param hs_data: DataFrame containing HS Codes and Descriptions
        """
        self.data = data

    def categorize_item(self, description):
        """
        Simply return the HS Description of the item.

        :param description: The HS Description of the item
        :return: The HS Description of the item
        """
        return description  # Return the description as is

    def categorize_items(self):
        """
        Return a new DataFrame containing HS Codes and Descriptions.

        :return: DataFrame with HS Codes and Descriptions
        """
        categorized_data = pd.DataFrame({
            'HS Code': self.data['HS Code'],
            'HS Description': self.data['HS Description'],
        })
        return categorized_data
    
class NLP:
    
    def categorize_brand(description):
        for category, keywords in category_mapping.items():
            if any(keyword in description.upper() for keyword in keywords):
                return category
        return 'Other'
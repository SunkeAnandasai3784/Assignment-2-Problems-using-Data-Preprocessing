import pandas as pd

# Load the dataset from the specified file path
df = pd.read_csv('E:/obesity/obesity_data.csv')

# Display the first five rows of the dataset
print(df.head())
print(df.columns)
# Calculate the percentage of obese population by age, gender, and income
age_gender_income = df.groupby(['Age(years)', 'Gender', 'Income']).mean()[['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit ']]

# Print the resulting DataFrame
print(age_gender_income)
# Remove the extra space in the column label
df = df.rename(columns={'High_Confidence_Limit ': 'High_Confidence_Limit'})

# Calculate the percentage of obese population by age, gender, and income
age_gender_income = df.groupby(['Age(years)', 'Gender', 'Income']).mean()[['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit']]

# Print the resulting DataFrame
print(age_gender_income)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df.drop('Data_Value', axis=1)
y = df['Data_Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

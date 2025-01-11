# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('Churn_Modelling.csv')

# Print shape, data types and number of missing values to log
print(data.shape)
print(data.info())
print(data.isnull().sum())

# Save description of data to a csv file
data.describe().to_csv('description.csv')

# Analysis of Exited variable
Exited_counts = data['Exited'].value_counts()
print(Exited_counts)
Exited_counts.plot(kind='bar', title='Rozkład zmiennej Exited')
plt.xticks(rotation=0)
plt.xlabel('')
plt.show()

# Remove columns responsible for record identification
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis='columns', inplace=True)

# Show histograms for numerical variables
numerical_variables = ['CreditScore', 'Age', 'Tenure','Balance', 'NumOfProducts','EstimatedSalary']
data[numerical_variables].hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Show histograms for binary variables
binary_variables = ['HasCrCard', 'IsActiveMember']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
data[binary_variables].hist(bins=3, ax=axes)
plt.setp(axes, xticks=[0,1])
plt.show()

# Boxplots for numerical variables
for column in numerical_variables:
    sns.boxplot(x='Exited', y=column, data=data)
    plt.title(f"{column} vs Exited")
    plt.show()

# Correlation for numerical variables
corr_matrix = data[numerical_variables].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.xticks(rotation=45, ha='right', fontsize=6)
plt.yticks(rotation=45, ha='right', fontsize=6)
plt.title('Macierz korelacji')
plt.show()

# Analysis of categorical variables
categorical_variables = ['Geography', 'Gender']
for column in categorical_variables:
    print(data[column].value_counts())
    sns.countplot(x=column, hue='Exited', data=data)
    plt.title(f"{column} vs Exited")
    plt.show()
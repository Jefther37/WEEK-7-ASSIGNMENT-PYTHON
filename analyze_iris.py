
# analyze_iris.py

# -------------------------------------------
# Step 1: Import Libraries
# -------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -------------------------------------------
# Step 2: Generate iris.csv File
# -------------------------------------------
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df.to_csv('iris.csv', index=False)
    print("iris.csv file created successfully!\n")
except Exception as e:
    print(f"Error creating iris.csv file: {e}")

# -------------------------------------------
# Step 3: Load Dataset
# -------------------------------------------
try:
    df = pd.read_csv('iris.csv')
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print("Error: iris.csv file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# -------------------------------------------
# Step 4: Explore Dataset
# -------------------------------------------
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Clean dataset if missing values are found
if df.isnull().sum().any():
    df = df.dropna()
    print("\nMissing values detected and dropped.")
else:
    print("\nNo missing values found.")

# -------------------------------------------
# Step 5: Basic Data Analysis
# -------------------------------------------
print("\nBasic Statistical Description:")
print(df.describe())

grouped_means = df.groupby('species').mean()
print("\nMean values for each species:")
print(grouped_means)

# -------------------------------------------
# Step 6: Data Visualization
# -------------------------------------------

# Set style
sns.set(style="whitegrid")

# 1. Line Chart: Average petal length per species
plt.figure(figsize=(8,6))
grouped_means['petal_length'].plot(marker='o')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.show()

# 2. Bar Chart: Average Sepal Width per Species
plt.figure(figsize=(8,6))
grouped_means['sepal_width'].plot(kind='bar', color='skyblue')
plt.title('Average Sepal Width per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.xticks(rotation=45)
plt.show()

# 3. Histogram: Distribution of Petal Length
plt.figure(figsize=(8,6))
plt.hist(df['petal_length'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# -------------------------------------------
# Step 7: Findings and Observations
# -------------------------------------------
print("""
Findings and Observations:

- Setosa species have much smaller petal lengths compared to Versicolor and Virginica.
- There is a positive relationship between Sepal Length and Petal Length across all species.
- Petal Length shows a bimodal distribution.
- Setosa has the highest average Sepal Width among the species.
""")

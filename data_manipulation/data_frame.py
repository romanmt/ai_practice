import pandas as pd

# Creating a DataFrame from a dictionary
#data = {
# 'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
#    'Age': [25, 30, 35, 40, 45],
#    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
#}

df = pd.read_csv("people.csv")

print(df.head())
print(df.describe())
print(df.info())


names = df['Name']
print(names)

subset = df[['Name', 'City']]
print(subset)

# Filter rows where age is greater than 30
filtered_df = df[df['Age'] > 30]
print(filtered_df)

df.fillna('Unknown', inplace=True)

df['Age'] = df['Age'].astype(int)

print(df.head)

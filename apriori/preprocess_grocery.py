import pandas as pd

# Load the dataset
groceries_df = pd.read_csv("Groceries_dataset.csv")

groceries_df['Date'] = pd.to_datetime(groceries_df['Date'], format='%d-%m-%Y')

# Group by customer and date to form transactions
transactions_df = groceries_df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()

# Remove duplicate items within each transaction
transactions_df['itemDescription'] = transactions_df['itemDescription'].apply(lambda items: list(set(items)))

transactions_df.columns = ['CustomerID', 'Date', 'Items']

# Export the processed transactions to a CSV file
transactions_df.to_csv("processed_grocery_transactions.csv", index=False)

print("Preprocessing complete. Processed transactions saved to 'processed_grocery_transactions.csv'.")

"""
Script to split the cleaned dataset into training and testing sets.
Split ratio: 80% train, 20% test
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/cleaned_dataset.csv')

print(f"Total number of rows in dataset: {len(df)}")
print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Split the dataset into features (X) and target (y)
X = df['text']
y = df['label']

# Split into training and testing sets (80% train, 20% test)
# shuffle=True ensures data is shuffled before splitting
# random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
    stratify=y  # Ensures proportional distribution of labels in both sets
)

# Print the results
print("\n" + "="*60)
print("DATASET SPLIT RESULTS")
print("="*60)
print(f"\nTraining set size: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Testing set size: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

print(f"\nTraining set label distribution:")
print(y_train.value_counts())
print(f"\nTesting set label distribution:")
print(y_test.value_counts())

# Save the split datasets for future use
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv('data/train_dataset.csv', index=False)
test_df.to_csv('data/test_dataset.csv', index=False)

print("\n" + "="*60)
print("Split datasets saved:")
print("  - data/train_dataset.csv")
print("  - data/test_dataset.csv")
print("="*60)

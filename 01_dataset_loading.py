import pandas as pd

train_df = pd.read_csv("data/sent_train.csv")
valid_df = pd.read_csv("data/sent_valid.csv")
excel_df = pd.read_excel("data/sent_valid.xlsx")

print("Training dataset shape:", train_df.shape)
print("Validation CSV shape:", valid_df.shape)
print("Validation Excel shape:", excel_df.shape)

print("\nTraining dataset preview:")
print(train_df.head())

print("\nValidation dataset preview:")
print(valid_df.head())

print("\nExcel dataset preview:")
print(excel_df.head())

# Compare validation CSV and Excel datasets

print("\nAre both validation datasets identical?")
print(valid_df.equals(excel_df))

print("\nValidation CSV columns:")
print(valid_df.columns.tolist())

print("\nValidation Excel columns:")
print(excel_df.columns.tolist())

# Check data types
print("\nCSV data types:")
print(valid_df.dtypes)

print("\nExcel data types:")
print(excel_df.dtypes)

# Compare rows one by one
text_diff = (
    valid_df["text"].fillna("").astype(str)
    != excel_df["text"].fillna("").astype(str)
)

label_diff = (
    valid_df["label"].fillna(-1).astype(str)
    != excel_df["label"].fillna(-1).astype(str)
)

diff_mask = text_diff | label_diff

print("\nNumber of different rows:", diff_mask.sum())

# Display first 5 differences
print("\nDifferences in CSV:")
print(valid_df[diff_mask].head())

print("\nDifferences in Excel:")
print(excel_df[diff_mask].head())

# Count text differences
text_diff = (
    valid_df["text"].fillna("").astype(str)
    != excel_df["text"].fillna("").astype(str)
)

# Count label differences
label_diff = (
    valid_df["label"] != excel_df["label"]
)

print("\nText differences:", text_diff.sum())
print("Label differences:", label_diff.sum())

# Display all differing rows
for i in valid_df.index[text_diff | label_diff]:
    print(f"\nRow index: {i}")
    print("CSV text:", repr(valid_df.loc[i, "text"]))
    print("Excel text:", repr(excel_df.loc[i, "text"]))
    print("CSV label:", valid_df.loc[i, "label"])
    print("Excel label:", excel_df.loc[i, "label"])

    # Dataset quality checks

print("\n--- TRAINING DATA QUALITY ---")

print("Missing values:")
print(train_df.isnull().sum())

print("\nDuplicate rows:")
print(train_df.duplicated().sum())


print("\n--- VALIDATION DATA QUALITY ---")

print("Missing values:")
print(valid_df.isnull().sum())

print("\nDuplicate rows:")
print(valid_df.duplicated().sum())


# Check unique sentiment labels

print("\nUnique labels in training data:")
print(train_df["label"].unique())

print("\nUnique labels in validation data:")
print(valid_df["label"].unique())
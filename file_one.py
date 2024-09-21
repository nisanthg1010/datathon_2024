import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error

# Load the train and test datasets
train_df = pd.read_csv('/content/Login train.csv')
test_df = pd.read_csv('/content/LoginTest.csv')  # Adjust this path based on your test dataset

# Select features and target for training
X = train_df[['CATEGORY_ID', 'ENTITY_DESCRIPTION']]
y = train_df['ENTITY_LENGTH']

# Split train dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the transformers: TF-IDF for text and StandardScaler for numeric features
text_transformer = TfidfVectorizer(stop_words='english', max_features=100)
numeric_transformer = StandardScaler()

# Column transformer to apply TF-IDF to ENTITY_DESCRIPTION and scaling to CATEGORY_ID
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'ENTITY_DESCRIPTION'),
        ('num', numeric_transformer, ['CATEGORY_ID'])
    ])

# Create a pipeline that first preprocesses the data and then applies Gradient Boosting
model = make_pipeline(preprocessor, GradientBoostingRegressor(random_state=42))

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = model.predict(X_val)

# Calculate the Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_val, y_val_pred)

# Final score based on MAPE as given by the challenge
final_score = max(0, 100 * (1 - mape))
print(f"MAPE: {mape}")
print(f"Final Score: {final_score}")

# Make predictions on the test dataset (for submission)
test_predictions = model.predict(test_df[['CATEGORY_ID', 'ENTITY_DESCRIPTION']])

# Prepare the submission dataframe with Entity_ID and predicted Entity_Length
submission_df = pd.DataFrame({
    'Entity_ID': test_df['ENTITY_ID'],
    'Entity_Length': test_predictions
})

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Graphical representation of Entity_ID vs. predicted Entity_Length
plt.figure(figsize=(10, 6))
sns.lineplot(x=submission_df['Entity_ID'], y=submission_df['Entity_Length'], marker='o', color='b')
plt.title('Predicted Entity Lengths vs Entity IDs')
plt.xlabel('Entity_ID')
plt.ylabel('Predicted Entity_Length')
plt.grid(True)
plt.show()

print("Submission file created successfully and graph displayed.")

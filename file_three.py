import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error,
                             confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

# Load the train and test datasets
train_df = pd.read_csv('/content/Login train1.csv')  # Adjust the path as per your file
test_df = pd.read_csv('LoginTest.csv')  # Adjust the path as per your file

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

# Calculate Regression Metrics
mae = mean_absolute_error(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)
rmse = mse ** 0.5
r2 = r2_score(y_val, y_val_pred)
mape = mean_absolute_percentage_error(y_val, y_val_pred)
final_score = max(0, 100 * (1 - mape))

# Print Regression Metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Final Score: {final_score}")

# Binarize the predictions for classification metrics
threshold = 100  # Define a threshold for ENTITY_LENGTH
y_val_binary = (y_val > threshold).astype(int)  # Actual binary labels
y_val_pred_binary = (y_val_pred > threshold).astype(int)  # Predicted binary labels

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_val_binary, y_val_pred_binary)
class_report = classification_report(y_val_binary, y_val_pred_binary)
roc_auc = roc_auc_score(y_val_binary, y_val_pred_binary)

# Print Classification Metrics
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Visualizations

# Metrics Bar Plot
metrics_df = pd.DataFrame({
    'Metrics': ['MAE', 'MSE', 'RMSE', 'MAPE', 'R²'],
    'Values': [mae, mse, rmse, mape, r2]
})
plt.figure(figsize=(8, 5))
sns.barplot(x='Metrics', y='Values', data=metrics_df)
plt.title('Model Performance Metrics')
plt.show()

# Histogram of Entity Lengths
plt.figure(figsize=(8, 5))
sns.histplot(train_df['ENTITY_LENGTH'], bins=30, kde=True)
plt.title('Distribution of Actual Entity Lengths')
plt.xlabel('Entity Length')
plt.ylabel('Frequency')
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix Heatmap')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Scatter Plot for Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_val_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Entity Lengths')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_val_binary, y_val_pred_binary)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Predictions on Test Dataset
test_predictions = model.predict(test_df[['CATEGORY_ID', 'ENTITY_DESCRIPTION']])

# Prepare the submission dataframe with Entity_ID and predicted Entity_Length
submission_df = pd.DataFrame({
    'Entity_ID': test_df['ENTITY_ID'],
    'Entity_Length': test_predictions
})

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Line Plot for Predicted Entity Lengths vs Entity IDs
plt.figure(figsize=(10, 6))
sns.lineplot(x=submission_df['Entity_ID'], y=submission_df['Entity_Length'], marker='o', color='b')
plt.title('Predicted Entity Lengths vs Entity IDs')
plt.xlabel('Entity_ID')
plt.ylabel('Predicted Entity_Length')
plt.grid(True)
plt.show()

print("Submission file created successfully and graph displayed.")

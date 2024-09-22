# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error

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

# Define Decision Tree Regressor model with hyperparameter tuning
decision_tree_model = DecisionTreeRegressor(random_state=42)

# Create a pipeline that first preprocesses the data and then applies Decision Tree
model = make_pipeline(preprocessor, decision_tree_model)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'decisiontreeregressor__max_depth': [None, 5, 10, 15],
    'decisiontreeregressor__min_samples_split': [2, 5, 10],
    'decisiontreeregressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predict on the validation set
y_val_pred = best_model.predict(X_val)

# Calculate the Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_val, y_val_pred)

# Final score based on MAPE as given by the challenge
final_score = max(0, 100 * (1 - mape))
print(f"MAPE: {mape}")
print(f"Final Score: {final_score}")

# Make predictions on the test dataset (for submission)
test_predictions = best_model.predict(test_df[['CATEGORY_ID', 'ENTITY_DESCRIPTION']])

# Prepare the submission dataframe with Entity_ID and predicted Entity_Length
submission_df = pd.DataFrame({
    'Entity_ID': test_df['ENTITY_ID'],
    'Entity_Length': test_predictions
})

# Save the predictions to a CSV file
submission_df.to_csv('submission_DT_optimized.csv', index=False)

# Graphical representation of Entity_ID vs. predicted Entity_Length
plt.figure(figsize=(10, 6))
sns.lineplot(x=submission_df['Entity_ID'], y=submission_df['Entity_Length'], marker='o', color='b')
plt.title('Predicted Entity Lengths vs Entity IDs')
plt.xlabel('Entity_ID')
plt.ylabel('Predicted Entity_Length')
plt.grid(True)
plt.show()

print("Submission file created successfully and graph displayed.")

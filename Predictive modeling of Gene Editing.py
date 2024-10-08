#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reimport pandas since it was not defined in this new session
import pandas as pd

# Load the new file uploaded by the user
file_path_new = '/Users/rajeevkumar/Documents/Documents - Rajeev’s MacBook Pro/Bioinformatics/Predictive modeling/gRNA_data.csv'

# Load the CSV to check its structure
new_data = pd.read_csv(file_path_new)

# Display the first few rows of the new dataset
new_data.head()


# In[2]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset (adjust the path if necessary)
file_path_new = '/Users/rajeevkumar/Documents/Documents - Rajeev’s MacBook Pro/Bioinformatics/Predictive modeling/gRNA_data.csv'
new_data = pd.read_csv(file_path_new)

# 1. Feature: Nucleotide Frequencies
def nucleotide_frequencies(gRNA_sequence):
    A_count = gRNA_sequence.count('A') / len(gRNA_sequence)
    T_count = gRNA_sequence.count('T') / len(gRNA_sequence)
    G_count = gRNA_sequence.count('G') / len(gRNA_sequence)
    C_count = gRNA_sequence.count('C') / len(gRNA_sequence)
    return [A_count, T_count, G_count, C_count]

# Apply the nucleotide frequency function to the dataset
new_data[['A_freq', 'T_freq', 'G_freq', 'C_freq']] = new_data['gRNA_sequence'].apply(lambda x: pd.Series(nucleotide_frequencies(x)))

# 2. Feature: PAM Sequence Detection (e.g., NGG for common CRISPR systems)
def has_pam_sequence(gRNA_sequence, pam="GG"):
    return int(gRNA_sequence[-len(pam):] == pam)

# Apply the PAM sequence check to the dataset
new_data['has_PAM'] = new_data['gRNA_sequence'].apply(lambda x: has_pam_sequence(x))

# 3. Optional Feature: Mismatch Count
# Ensure 'target_sequence' column exists and is of the same length as 'gRNA_sequence'
# You can comment this out if 'target_sequence' is not available
if 'target_sequence' in new_data.columns and len(new_data['target_sequence']) == len(new_data['gRNA_sequence']):
    def count_mismatches(gRNA_sequence, target_sequence):
        mismatches = sum(1 for g, t in zip(gRNA_sequence, target_sequence) if g != t)
        return mismatches
    
    new_data['mismatch_count'] = new_data.apply(lambda row: count_mismatches(row['gRNA_sequence'], row['target_sequence']), axis=1)
else:
    # Fallback if no target sequence is available
    new_data['mismatch_count'] = 0  # Placeholder or can be removed

# 4. Feature: Convert gRNA sequences to numerical format (for machine learning)
new_data['gRNA_numeric'] = new_data['gRNA_sequence'].apply(lambda x: [ord(c) for c in x])

# Ensure all sequences have consistent lengths
max_len = max(new_data['gRNA_numeric'].apply(len))
new_data['gRNA_numeric'] = new_data['gRNA_numeric'].apply(lambda x: x + [0] * (max_len - len(x)))

# Prepare the feature matrix (X) and target (y)
# Extract all features including numeric gRNA features
gRNA_numeric_df = pd.DataFrame(new_data['gRNA_numeric'].to_list())  # Convert list to DataFrame

# Join numeric gRNA features with engineered features
X = pd.concat([new_data[['A_freq', 'T_freq', 'G_freq', 'C_freq', 'has_PAM', 'mismatch_count']], gRNA_numeric_df], axis=1).values
y = new_data[['efficiency_score', 'off_site_efficiency']].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a multi-output Random Forest model
multi_target_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

# Train the model
multi_target_model.fit(X_train, y_train)

# Make predictions
y_pred = multi_target_model.predict(X_test)

# Evaluate the model's performance
mse_on_target = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_off_target = mean_squared_error(y_test[:, 1], y_pred[:, 1])

print(f"On-target efficiency MSE: {mse_on_target}")
print(f"Off-target efficiency MSE: {mse_off_target}")

# Save the trained model for future use
model_filename = 'gRNA_efficiency_model_with_features.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(multi_target_model, model_file)

print(f"Model saved as {model_filename}")


# In[3]:


import matplotlib.pyplot as plt

# Plot for on-target efficiency: Actual vs Predicted
plt.figure(figsize=(10, 5))

# On-Target Efficiency
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
plt.title('On-Target Efficiency: Actual vs Predicted')
plt.xlabel('Actual On-Target Efficiency')
plt.ylabel('Predicted On-Target Efficiency')
plt.plot([min(y_test[:, 0]), max(y_test[:, 0])], 
         [min(y_test[:, 0]), max(y_test[:, 0])], 
         color='red', linestyle='--')  # Diagonal line for perfect prediction

# Off-Target Efficiency
plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
plt.title('Off-Target Efficiency: Actual vs Predicted')
plt.xlabel('Actual Off-Target Efficiency')
plt.ylabel('Predicted Off-Target Efficiency')
plt.plot([min(y_test[:, 1]), max(y_test[:, 1])], 
         [min(y_test[:, 1]), max(y_test[:, 1])], 
         color='red', linestyle='--')  # Diagonal line for perfect prediction

plt.tight_layout()
plt.show()


# In[4]:


import numpy as np

# Function to convert gRNA sequence to numeric format (padded to the same length as during training)
def gRNA_to_numeric(gRNA_sequence, max_len=20):
    numeric_rep = [ord(c) for c in gRNA_sequence] + [0] * (max_len - len(gRNA_sequence))  # Padding to max_len
    return numeric_rep

# Function to calculate nucleotide frequencies (as done during training)
def nucleotide_frequencies(gRNA_sequence):
    A_count = gRNA_sequence.count('A') / len(gRNA_sequence)
    T_count = gRNA_sequence.count('T') / len(gRNA_sequence)
    G_count = gRNA_sequence.count('G') / len(gRNA_sequence)
    C_count = gRNA_sequence.count('C') / len(gRNA_sequence)
    return [A_count, T_count, G_count, C_count]

# Function to detect if PAM sequence (e.g., "GG") is present
def has_pam_sequence(gRNA_sequence, pam="GG"):
    return int(gRNA_sequence[-len(pam):] == pam)

# Example of mismatch count (if applicable, otherwise can be omitted)
# Adjust this function if you have the target sequence available
def count_mismatches(gRNA_sequence, target_sequence):
    mismatches = sum(1 for g, t in zip(gRNA_sequence, target_sequence) if g != t)
    return mismatches

# Sample input gRNA sequence
gRNA_sequence = "GATGTCCACTATGACAATTG"  # Replace with your gRNA sequence

# Assuming you have 6 additional features from feature engineering (adjust according to your feature engineering steps)
additional_features = nucleotide_frequencies(gRNA_sequence) + [has_pam_sequence(gRNA_sequence)] + [0]  # For mismatch count or other features

# Convert the gRNA sequence into numeric format and combine it with the additional features
gRNA_numeric = gRNA_to_numeric(gRNA_sequence)  # This should be padded to match the length used in training
input_features = additional_features + gRNA_numeric  # Combine additional features with gRNA numeric encoding

# Ensure the input has the correct number of features (26 in this case)
if len(input_features) != 26:
    raise ValueError(f"Expected 26 features, but got {len(input_features)}")

# Convert to the correct format for model prediction
input_array = np.array([input_features])

# Predict using the trained model
predicted_efficiencies = multi_target_model.predict(input_array)

# Print the prediction results
predicted_on_target_efficiency = predicted_efficiencies[0][0]
predicted_off_target_efficiency = predicted_efficiencies[0][1]

print(f"Predicted On-Target Efficiency: {predicted_on_target_efficiency}")
print(f"Predicted Off-Target Efficiency: {predicted_off_target_efficiency}")



# In[5]:


get_ipython().system('pip install flask sqlalchemy sqlite3')


# In[6]:


import sqlite3
print(sqlite3.version)



# In[7]:


get_ipython().system('pip install flask sqlalchemy')



# In[8]:


!pip install flask-sqlalchemy


# In[10]:


import flask_sqlalchemy
print(flask_sqlalchemy.__version__)


# In[12]:


from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Configure the SQLAlchemy part of the app
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gRNA_predictions.db'  # For SQLite, or use MySQL/PostgreSQL URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Model for storing gRNA sequences and their predictions in the database
class GRNAPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gRNA_sequence = db.Column(db.String(255), nullable=False)
    predicted_on_target_efficiency = db.Column(db.Float, nullable=False)
    predicted_off_target_efficiency = db.Column(db.Float, nullable=False)

# Create the database (use the application context to avoid the error)
with app.app_context():
    if not os.path.exists('gRNA_predictions.db'):
        db.create_all()

# Load the trained model
with open('gRNA_efficiency_model_with_features.pkl', 'rb') as model_file:
    multi_target_model = pickle.load(model_file)

# Helper function to convert gRNA sequence to numerical format
def gRNA_to_numeric(gRNA_sequence):
    max_len = 20  # Adjust based on your gRNA sequence length
    numeric_rep = [ord(c) for c in gRNA_sequence] + [0] * (max_len - len(gRNA_sequence))  # Padding to max_len
    return numeric_rep

# API endpoint to predict efficiencies
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    gRNA_sequence = data['gRNA_sequence']
    
    # Convert gRNA sequence into numeric format
    gRNA_numeric = np.array([gRNA_to_numeric(gRNA_sequence)])
    
    # Make predictions using the trained model
    predicted_efficiencies = multi_target_model.predict(gRNA_numeric)
    
    predicted_on_target_efficiency = predicted_efficiencies[0][0]
    predicted_off_target_efficiency = predicted_efficiencies[0][1]
    
    # Save the prediction to the database
    new_prediction = GRNAPrediction(
        gRNA_sequence=gRNA_sequence,
        predicted_on_target_efficiency=predicted_on_target_efficiency,
        predicted_off_target_efficiency=predicted_off_target_efficiency
    )
    db.session.add(new_prediction)
    db.session.commit()

    return jsonify({
        'gRNA_sequence': gRNA_sequence,
        'predicted_on_target_efficiency': predicted_on_target_efficiency,
        'predicted_off_target_efficiency': predicted_off_target_efficiency
    })

# API endpoint to retrieve all predictions
@app.route('/predictions', methods=['GET'])
def get_predictions():
    predictions = GRNAPrediction.query.all()
    output = []
    
    for pred in predictions:
        output.append({
            'gRNA_sequence': pred.gRNA_sequence,
            'predicted_on_target_efficiency': pred.predicted_on_target_efficiency,
            'predicted_off_target_efficiency': pred.predicted_off_target_efficiency
        })
    
    return jsonify({'predictions': output})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)



# In[ ]:
import requests
response = requests.post('http://127.0.0.1:5000/predict', json={'gRNA_sequence': 'GATGTCCACTATGACAATTG'})
print(response.json())




# In[ ]:
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Extract JSON data from the POST request
    gRNA_sequence = data.get('gRNA_sequence')
    
    # Perform the prediction logic here (this is just an example)
    predicted_on_target_efficiency = 95.5  # Replace with your actual prediction logic
    predicted_off_target_efficiency = 80.3  # Replace with your actual prediction logic

    # Return a JSON response using jsonify
    return jsonify({
        'gRNA_sequence': gRNA_sequence,
        'predicted_on_target_efficiency': predicted_on_target_efficiency,
        'predicted_off_target_efficiency': predicted_off_target_efficiency
    })

if __name__ == '__main__':
    app.run(debug=True)





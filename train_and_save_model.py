import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Disable parallel processing to avoid issues
os.environ['JOBLIB_NUM_CPU'] = '1'

# Load the dataset
df = pd.read_csv('spotify_iitd1.csv')

# Use a smaller subset for faster training
df = df.sample(n=2000, random_state=42)

# Drop columns that are not useful for prediction
drop_cols = ['track_id', 'track_name', 'album_name']
df = df.drop(columns=drop_cols)

# Handle missing values
df = df.dropna()

# Encode 'explicit' as integer
df['explicit'] = df['explicit'].astype(int)

# Create and fit encoders for categorical columns
le_artists = LabelEncoder()
df['artists'] = le_artists.fit_transform(df['artists'])

le_genre = LabelEncoder()
df['track_genre'] = le_genre.fit_transform(df['track_genre'])

# Encode all object (string) columns
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Feature/Target Split
y = df['popularity']
X = df.drop(columns=['popularity'])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit preprocessing
preprocessing = StandardScaler()
preprocessing.fit(X_train)  # Fit the scaler with training data

# Create and fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Create directory if it doesn't exist
os.makedirs('Streamlit_objects', exist_ok=True)

# Save fitted preprocessing, model, and encoders
joblib.dump(preprocessing, 'Streamlit_objects/preprocessing.pkl')
joblib.dump(model, 'Streamlit_objects/random_forest_regressor.pkl')
joblib.dump(le_artists, 'Streamlit_objects/artists_encoder.pkl')
joblib.dump(le_genre, 'Streamlit_objects/genre_encoder.pkl')

print("Model and preprocessing pipeline saved successfully!")
print("Files created:")
print("- Streamlit_objects/preprocessing.pkl")
print("- Streamlit_objects/random_forest_regressor.pkl")
print("- Streamlit_objects/artists_encoder.pkl")
print("- Streamlit_objects/genre_encoder.pkl") 
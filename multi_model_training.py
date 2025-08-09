import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Training Multiple Regression Models for Spotify Popularity Prediction")
    print("Including Neural Network (MLP Regressor)")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv('spotify_iitd1.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Use subset for faster training
    df = df.sample(n=5000, random_state=42)
    print(f"Using subset: {df.shape}")
    
    # Preprocessing
    drop_cols = ['track_id', 'track_name', 'album_name']
    df = df.drop(columns=drop_cols)
    df = df.dropna()
    df['explicit'] = df['explicit'].astype(int)
    
    # Encode categorical variables
    le_artists = LabelEncoder()
    df['artists'] = le_artists.fit_transform(df['artists'])
    
    le_genre = LabelEncoder()
    df['track_genre'] = le_genre.fit_transform(df['track_genre'])
    
    # Handle any remaining object columns
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Split features and target
    y = df['popularity']
    X = df.drop(columns=['popularity'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print()
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42, max_iter=2000),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers with decreasing neurons
            activation='relu',                  # ReLU activation for non-linearity
            solver='adam',                     # Adam optimizer for better convergence
            alpha=0.001,                       # L2 regularization to prevent overfitting
            batch_size='auto',                 # Automatic batch size selection
            learning_rate='adaptive',          # Adaptive learning rate
            learning_rate_init=0.001,          # Initial learning rate
            max_iter=500,                      # Maximum iterations
            early_stopping=True,               # Stop early if no improvement
            validation_fraction=0.1,           # 10% for validation during training
            n_iter_no_change=20,              # Patience for early stopping
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = -float('inf')
    best_name = ""
    
    print("Training and Evaluating Models:")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use scaled data for linear models and neural networks, original for tree-based
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Neural Network']:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # Train model
        model.fit(X_tr, y_train)
        
        # Predictions
        y_pred = model.predict(X_te)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        # Track best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
        
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE:  {mae:.3f}")
        print(f"  R2:   {r2:.3f}")
        print(f"  CV R2: {cv_mean:.3f} (+/- {cv_std:.3f})")
        print()
    
    # Print summary
    print("=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R2':<8} {'CV R2':<10}")
    print("-" * 70)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        print(f"{name:<20} {result['rmse']:<8.3f} {result['mae']:<8.3f} {result['r2']:<8.3f} {result['cv_mean']:<10.3f}")
    
    print("=" * 70)
    print(f"BEST MODEL: {best_name} (R2 = {best_score:.3f})")
    print("=" * 70)
    
    # Save models and preprocessing
    os.makedirs('Streamlit_objects', exist_ok=True)
    
    # Save preprocessing and encoders
    joblib.dump(scaler, 'Streamlit_objects/preprocessing.pkl')
    joblib.dump(le_artists, 'Streamlit_objects/artists_encoder.pkl')
    joblib.dump(le_genre, 'Streamlit_objects/genre_encoder.pkl')
    
    # Save best model
    joblib.dump(best_model, 'Streamlit_objects/best_model.pkl')
    
    # Save all results
    joblib.dump(results, 'Streamlit_objects/all_models_results.pkl')
    
    print("Models and preprocessing saved to Streamlit_objects/")
    print("Files created:")
    print("- preprocessing.pkl")
    print("- artists_encoder.pkl") 
    print("- genre_encoder.pkl")
    print("- best_model.pkl")
    print("- all_models_results.pkl")

if __name__ == "__main__":
    main()

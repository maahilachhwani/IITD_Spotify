import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the dataset
DATA_PATH = 'spotify_iitd1.csv'
df = pd.read_csv(DATA_PATH)
df.head()
df.shape
df.info()
df.describe().transpose()
# df.hist(bins = 20, color = 'orange', figsize = (20, 14))
df.isnull().sum() #df.isnull().sum() in pandas is used to count the number of missing (null or NaN) values in each column of a DataFrame

# Plot distributions of numeric features
df.hist(bins=20, figsize=(20, 14), color='skyblue')
plt.suptitle('Feature Distributions', fontsize=20)
plt.show()


# artists, album_name and track_name needs to be looked at
# 2. Data Preprocessing
# Drop columns that are not useful for prediction
# (track_id is just an identifier, track_name and album_name may be too granular)
drop_cols = ['track_id', 'track_name', 'album_name']
df = df.drop(columns=drop_cols)



# Handle missing values (simple strategy: drop rows with missing values)
df = df.dropna()

# Encode 'explicit' as integer (True/False to 1/0)
df['explicit'] = df['explicit'].astype(int)

# Encode categorical columns: 'artists', 'track_genre'
le_artists = LabelEncoder()
df['artists'] = le_artists.fit_transform(df['artists'])
le_genre = LabelEncoder()
df['track_genre'] = le_genre.fit_transform(df['track_genre'])

# Encode all object (string) columns
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Popularity by genre (top 10 genres)
top_genres = df['track_genre'].value_counts().index[:10]
df_top_genres = df[df['track_genre'].isin(top_genres)]
plt.figure(figsize=(12, 6))
sns.boxplot(x='track_genre', y='popularity', data=df_top_genres)
plt.title('Popularity by Top 10 Genres')
plt.xlabel('Genre (encoded)')
plt.ylabel('Popularity')
plt.show()

# 3. Feature/Target Split
y = df['popularity']
X = df.drop(columns=['popularity'])

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Random Forest Regression Results:')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R^2: {r2:.2f}')

# 7. Feature Importance (optional)
importances = model.feature_importances_
feature_names = X.columns
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print('\nFeature Importances:')
for name, imp in feat_imp:
    print(f'{name}: {imp:.3f}')
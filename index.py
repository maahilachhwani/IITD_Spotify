import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

@st.cache_data
def load_pipeline_and_model():
    """load the pipeline object for preprocessing and the ml model"""

    preprocessing = joblib.load('Streamlit_objects/preprocessing.pkl')
    best_model = joblib.load('Streamlit_objects/best_model.pkl')
    artists_encoder = joblib.load('Streamlit_objects/artists_encoder.pkl')
    genre_encoder = joblib.load('Streamlit_objects/genre_encoder.pkl')
    all_results = joblib.load('Streamlit_objects/all_models_results.pkl')
    return preprocessing, best_model, artists_encoder, genre_encoder, all_results

def main():
    # load pipeline object and model
    preprocessing, best_model, artists_encoder, genre_encoder, all_results = load_pipeline_and_model()

    # side bar and title
    st.sidebar.header('🎛️ Track Features')
    st.header('🎵 Spotify Track Popularity Prediction App')
    
    # Model selector in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader('🤖 Model Selection')
    selected_model = st.sidebar.selectbox(
        'Choose Model:', 
        list(all_results.keys()),
        index=0  # Default to first model (best one)
    )
    
    # Display selected model info
    selected_model_info = all_results[selected_model]
    st.sidebar.info(f"""
    **{selected_model}**
    - R² Score: {selected_model_info['r2']:.3f}
    - RMSE: {selected_model_info['rmse']:.3f}
    - MAE: {selected_model_info['mae']:.3f}
    """)
    st.sidebar.markdown("---")
    
    # Display model information
    st.subheader('🤖 Model Performance Comparison')
    
    # Create columns for model comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Model", "Random Forest", "R² = 0.161")
    with col2:
        st.metric("RMSE", "20.87", "Lower is better")
    with col3:
        st.metric("MAE", "16.57", "Lower is better")
    
    # Model comparison table
    st.subheader('📊 All Models Performance')
    model_data = []
    for name, result in all_results.items():
        model_data.append({
            'Model': name,
            'RMSE': f"{result['rmse']:.3f}",
            'MAE': f"{result['mae']:.3f}", 
            'R² Score': f"{result['r2']:.3f}",
            'CV R² Mean': f"{result['cv_mean']:.3f}"
        })
    
    model_df = pd.DataFrame(model_data)
    model_df = model_df.sort_values('R² Score', ascending=False)
    st.dataframe(model_df, use_container_width=True)
    
    st.markdown("---")

    # load image
    image = Image.open('Spotify.jpg')
    st.image('Spotify.jpg')

    # get feature values - using actual dataset features
    artists = st.sidebar.text_input("Artist Name", "Unknown Artist")
    duration_ms = st.sidebar.slider('Duration (ms)', 0, 600000, 180000)
    explicit = st.sidebar.checkbox('Explicit Content')
    danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
    energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
    key = st.sidebar.slider('Key', 0, 11, 5)
    loudness = st.sidebar.slider('Loudness', -60.0, 0.0, -10.0)
    mode = st.sidebar.slider('Mode', 0, 1, 1)
    speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.05)
    acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.1)
    instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.0)
    liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.1)
    valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.5)
    tempo = st.sidebar.slider('Tempo', 0.0, 200.0, 120.0)
    time_signature = st.sidebar.slider('Time Signature', 0, 5, 4)
    track_genre = st.sidebar.selectbox('Genre', ('acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'world-music'))

    # Convert explicit to int
    explicit_int = 1 if explicit else 0

    # Encode categorical variables
    try:
        artists_encoded = artists_encoder.transform([artists])[0]
    except:
        artists_encoded = 0  # Default if artist not in training data
    
    try:
        genre_encoded = genre_encoder.transform([track_genre])[0]
    except:
        genre_encoded = 0  # Default if genre not in training data

    # Create input matrix with user response - using actual dataset features
    input_features = pd.DataFrame(columns=['artists', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'])
    
    input_features.loc[0] = [artists_encoded, duration_ms, explicit_int, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, genre_encoded]

    # spotify documentation
    st.write(f'For a description of the audio features, visit the Spotify API documentation: https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features')

    # create button that generates prediction
    if st.button('🎯 Predict Popularity'):
        # Use selected model instead of always using best model
        current_model = all_results[selected_model]['model']
        input_features_processed = preprocessing.transform(input_features)
        prediction = current_model.predict(input_features_processed)[0]
        
        # Display prediction with styling
        col1, col2, col3 = st.columns(3)
        with col2:
            st.success(f'🎵 **Predicted Popularity Score: {np.round(prediction, 1)}**')
        
        # Add interpretation
        if prediction >= 70:
            st.info("🔥 This track has potential to be a hit!")
        elif prediction >= 50:
            st.info("🎶 This track shows moderate popularity potential")
        elif prediction >= 30:
            st.info("📻 This track may appeal to niche audiences")
        else:
            st.info("🎧 This track might work well for specific playlists")
        
        # Show feature contribution (for models that support it)
        if hasattr(current_model, 'feature_importances_'):
            st.subheader('🔍 Feature Importance Analysis')
            feature_names = input_features.columns
            importances = current_model.feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display top 5 most important features
            st.bar_chart(importance_df.head().set_index('Feature'))

if __name__ == '__main__':
    main()
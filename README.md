Build a model to predict the success of a song on spotify. Th column of interest is popularity in the dataset.
Please refer Spoitify_iitd.csv for the given dataset to analyze.
Columns Description:
1. track_id: The unique Spotify id for each track.
2. artists: Names of the artists who prfrmed the track, separated by ';'.
3. album_name: The name of the album in which the track appears.
4. track_name: The title of the track.
5. popularity: A value between 0 and 100. indicating the tracks popularity based on recent plays.
6. duration_ms : The length of the track in milliseconds.
7. explicit: Boolean idicating whether the track contains explicit content.
8. danceability: Describes how suitable a track is for dancing: (0.0 = least danceable, 1.0 = most danceable).
9. energy : Represents the intensity and activity of a track(0.0 = low energy, 1.0 = high energy).
10. key: the usical key of the track mapped using standard pitch class notation.
11. loudness: overall loudness of the track in decibels(dB).
12. mode: indicates the modality(major or minor) of the track.
13. speechiness: detects the presence of spoken words in the track.
14. acousticness:confidence measure of whether the track is acousti (0.0 = not acoustic, 1.0 = highly acoustic).
15. instrumentalness: predicts whether a track contains vocals (0.0 = contains vocals, 1.0 = instrumental).
16. liveness: detects the presence of an audience in the recording(0.0 = studio recording, 1.0 = live performance).
17. valence: measures the musical positiveness conveyed by atrack (0.0 = negative, 1.0 = positive).
18. tempo : estimested tempo of the track in beats per minute. (BPM)
19. time_signature: estimated time signature of the track (3 to 7).
20. track_genre : the genre of the track.

Current state - EDA (Work in progress)
References: 

https://www.kaggle.com/code/adisongoh/predicting-the-success-of-spotify-songs
https://github.com/cristobalvch/Spotify-Machine-Learning/blob/master/playlist.ipynb
https://maxgmarin.github.io/AC209a_FinalProject_EEM/
https://www.kaggle.com/datasets/lehaknarnauli/spotify-datasets
https://www.kaggle.com/code/adisongoh/predicting-the-success-of-spotify-songs
https://medium.com/@nthnil/hit-or-miss-predicting-the-success-of-songs-on-spotify-ccc736b80e90
https://www.irjet.net/archives/V8/i10/IRJET-V8I1046.pdf
https://github.com/Thomas-George-T/Prediciting-Hits-on-Spotify/blob/main/Readme.md
https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeRegressor.html
https://dorazaria.github.io/machinelearning/spotify-popularity-prediction/

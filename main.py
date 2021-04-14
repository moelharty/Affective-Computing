import pandas as pd
import matplotlib.pyplot as plt

df_features = pd.read_csv('dataset/phase_2_TRAIN_ab71fa4d86c94323_05dcbf4_MLPC2021_features_generic.csv')

df_raw = pd.read_csv('dataset/phase_2_TRAIN_8d8d624e3190984c_05dcbf4_MLPC2021_raw_annotations_generic.csv')

df_weka = df_features[
    ["id", "essentia_dissonance_mean", "essentia_dissonance_stdev", "essentia_dynamic_complexity", "essentia_loudness",
     "essentia_onset_rate", "essentia_pitch_salience_mean", "essentia_pitch_salience_stdev",
     "essentia_spectral_centroid_mean", "essentia_spectral_centroid_stdev", "essentia_spectral_complexity_mean",
     "essentia_spectral_complexity_stdev", "essentia_spectral_rolloff_mean", "essentia_spectral_rolloff_stdev",
     "essentia_strong_peak_mean", "essentia_strong_peak_stdev", "librosa_bpm", "librosa_chroma_mean_0",
     "librosa_chroma_var_0", "librosa_chroma_pct_10_0", "librosa_chroma_pct_50_0", "librosa_chroma_pct_90_0",
     "librosa_chroma_mean_1", "librosa_chroma_var_1", "librosa_chroma_pct_10_1", "librosa_chroma_pct_50_1",
     "librosa_chroma_pct_90_1", "librosa_chroma_mean_2", "librosa_chroma_var_2", "librosa_chroma_pct_10_2",
     "librosa_chroma_pct_50_2", "librosa_chroma_pct_90_2", "librosa_chroma_mean_3", "librosa_chroma_var_3",
     "librosa_chroma_pct_10_3", "librosa_chroma_pct_50_3", "librosa_chroma_pct_90_3", "librosa_chroma_mean_4",
     "librosa_chroma_var_4", "librosa_chroma_pct_10_4", "librosa_chroma_pct_50_4", "librosa_chroma_pct_90_4",
     "librosa_chroma_mean_5", "librosa_chroma_var_5", "librosa_chroma_pct_10_5", "librosa_chroma_pct_50_5",
     "librosa_chroma_pct_90_5", "librosa_chroma_mean_6", "librosa_chroma_var_6", "librosa_chroma_pct_10_6",
     "librosa_chroma_pct_50_6", "librosa_chroma_pct_90_6", "librosa_chroma_mean_7", "librosa_chroma_var_7",
     "librosa_chroma_pct_10_7", "librosa_chroma_pct_50_7", "librosa_chroma_pct_90_7", "librosa_chroma_mean_8",
     "librosa_chroma_var_8", "librosa_chroma_pct_10_8", "librosa_chroma_pct_50_8", "librosa_chroma_pct_90_8",
     "librosa_chroma_mean_9", "librosa_chroma_var_9", "librosa_chroma_pct_10_9", "librosa_chroma_pct_50_9",
     "librosa_chroma_pct_90_9", "librosa_chroma_mean_10", "librosa_chroma_var_10", "librosa_chroma_pct_10",
     "librosa_chroma_pct_50_10", "librosa_chroma_pct_90_10", "librosa_chroma_mean_11", "librosa_chroma_var_11",
     "librosa_chroma_pct_10_11", "librosa_chroma_pct_50_11", "librosa_chroma_pct_90_11",
     "librosa_spectral_bandwidth_mean", "librosa_spectral_bandwidth_stdev", "librosa_spectral_flatness_mean",
     "librosa_spectral_flatness_stdev", "midlevel_features_melody", "midlevel_features_articulation",
     "midlevel_features_rhythm_complexity", "midlevel_features_rhythm_stability", "midlevel_features_dissonance",
     "midlevel_features_tonal_stability", "midlevel_features_minorness", "score_mode", "score_key_strength"
     ]].copy()

df_weka['pianist'] = df_weka.apply(lambda row: row.id[0:2], axis=1)
df_weka['piece'] = df_weka.apply(lambda row: int(row.id[3:5]), axis=1)
df_weka['counter'] = df_weka.apply(lambda row: int(row.id[6:9]), axis=1)

df_new = pd.merge(df_weka, df_raw, how='left', on=['pianist', 'piece'])

df_new.loc[((df_new['arousal'] >= 50) & (df_new['valence'] >= 0)), 'class'] = 'Happy'
df_new.loc[((df_new['arousal'] < 50) & (df_new['valence'] >= 0)), 'class'] = 'Relaxed'
df_new.loc[((df_new['arousal'] < 50) & (df_new['valence'] < 0)), 'class'] = 'Sad'
df_new.loc[((df_new['arousal'] >= 50) & (df_new['valence'] < 0)), 'class'] = 'Angry'

#df_new.drop('arousal', axis='columns', inplace=True)
#df_new.drop('valence', axis='columns', inplace=True)

pianists = ['GG', 'FG', 'AH', 'SR', 'AS', 'RT']

for i in range(1, 2, 1):
    df_piece = df_new['piece']==i
    df_filtered_piece = df_new[df_piece]
    for j in pianists:
        df_artist = df_filtered_piece['pianist']==j
        df_filtered_artist = df_filtered_piece[df_artist]
        df_filtered_artist.boxplot(column=['arousal', 'valence'])
        #print(df_filtered_artist['piece'])
        print(df_filtered_artist['arousal'].min())
        print(df_filtered_artist['arousal'].max())
        print(df_filtered_artist['valence'].min())
        print(df_filtered_artist['valence'].max())
        plt.show()
        #print(df_filtered['piece'])


# you can run this code
# but the generated csv file is too big for git so you have to delete the content of the weka.csv first to commit your changes
#df_new.to_csv('weka_output/weka.csv', index=False, header=True, line_terminator='\n', sep=",")

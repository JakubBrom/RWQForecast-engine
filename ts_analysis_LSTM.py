import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed


# Předpokládejme, že máte funkci na načítání snímků z disku nebo jiného zdroje
def load_images(date):
    # Implementujte funkci na načtení obrázků pro daný datum
    pass


# Načtení koregresorů
df = pd.read_csv('dataset.csv', parse_dates=['Date'], index_col='Date')

# Normalizace dat (kromě obrázků)
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df.iloc[:, 1:])


# Příprava sekvencí
def create_sequences_with_images(df, time_steps=1):
    sequences = []
    targets = []
    for i in range(len(df) - time_steps):
        seq_images = []
        for j in range(i, i + time_steps):
            seq_images.append(load_images(df.index[j]))
        seq_images = np.array(seq_images)

        seq_koregresory = df.iloc[i:(i + time_steps), 1:].values
        target = df.iloc[i + time_steps, 0]  # MP is the target variable

        sequences.append((seq_images, seq_koregresory))
        targets.append(target)
    return sequences, targets


time_steps = 10
sequences, targets = create_sequences_with_images(df, time_steps)
#
#
#
# # Rozdělení na trénovací a testovací sady
# train_size = int(len(sequences) * 0.8)
# train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
# train_targets, test_targets = targets[:train_size], targets[train_size:]
#
# input_images = Input(shape=(time_steps, image_height, image_width, image_channels))
# input_koregresory = Input(shape=(time_steps, num_koregresory))
#
# # CNN část
# cnn = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(input_images)
# cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
# cnn = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(cnn)
# cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
# cnn = TimeDistributed(Flatten())(cnn)
#
# # Spojení CNN výstupů a koregresorů
# merged = concatenate([cnn, input_koregresory])
#
# # LSTM část
# lstm_out = LSTM(50)(merged)
#
# # Výstupní vrstva
# output = Dense(1)(lstm_out)
#
# # Sestavení modelu
# model = Model(inputs=[input_images, input_koregresory], outputs=output)
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # Zobrazení architektury modelu
# model.summary()
#
# # Trening modelu
# # Příprava trénovacích a testovacích dat
# X_train_images = np.array([seq[0] for seq in train_sequences])
# X_train_koregresory = np.array([seq[1] for seq in train_sequences])
# X_test_images = np.array([seq[0] for seq in test_sequences])
# X_test_koregresory = np.array([seq[1] for seq in test_sequences])
# y_train = np.array(train_targets)
# y_test = np.array(test_targets)
#
# # Trénování modelu
# model.fit([X_train_images, X_train_koregresory], y_train, epochs=50, batch_size=32, validation_split=0.2)
#
# # Predikce na testovacích datech
# y_pred = model.predict([X_test_images, X_test_koregresory])
#
# # Inverzní transformace předpovědí na původní měřítko
# y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), df.shape[1] - 1)))))[:, 0]
# y_pred_rescaled = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), df.shape[1] - 1)))))[:, 0]
#
# # Vyhodnocení výkonu modelu
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
#
# mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
# print(f'Mean Squared Error: {mse}')
#
# # Graf predikcí a skutečných hodnot
# plt.figure(figsize=(14, 5))
# plt.plot(df.index[train_size + time_steps:], y_test_rescaled, label='Skutečné hodnoty')
# plt.plot(df.index[train_size + time_steps:], y_pred_rescaled, label='Predikované hodnoty')
# plt.legend()
# plt.show()



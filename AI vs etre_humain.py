import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Charger les données depuis le fichier CSV
df = pd.read_csv('/home/noureddine/train_drcat_04.csv')
# Sélectionner les colonnes pour les entrées (textes) et les sorties (labels)
textes = df.iloc[:, 1].values
labels = df.iloc[:, 2].values
print(textes[44205])
print(labels[44205])
############Diviser le data set en train set et test set
textes_train, textes_test, labels_train, labels_test = train_test_split(textes, labels, test_size=0.2, random_state=42)

'''Le Tokenizer dans Keras est utilisé pour transformer des textes en séquences d'entiers,
Il est utilisé pour le traitement du langage '''
tokenizer = Tokenizer(num_words=10000)#Il divise le texte en mots individuels.
tokenizer.fit_on_texts(textes_train)#il donne un indice à chaque mots
sequences_train = tokenizer.texts_to_sequences(textes_train)#Convertit les textes d'entraînement en séquences d'entiers en remplaçant chaque mot par son index correspondant dans le vocabulaire construit par le Tokenizer
sequences_test = tokenizer.texts_to_sequences(textes_test)#Convertit les textes de test en séquences d'entiers en utilisant le même vocabulaire construit par le Tokenizer à partir des textes d'entraînement.
word_index = tokenizer.word_index#Récupère le dictionnaire contenant la correspondance entre les mots et leurs index dans le vocabulaire.
'''Calcule la longueur maximale parmi toutes les séquences d'entraînement,
Cette longueur maximale sera utilisée pour le rembourrage (padding) des séquences afin qu'elles aient toutes la même longueur lors de l'entraînement du modèle'''
max_len = max([len(seq) for seq in sequences_train])



# Rembourrage des séquences pour qu'elles aient toutes la même longueur c'est à dire ajouter des zéros ou tronquer les séquences pour qu'elles aient toutes la même longueur
sequences_train = pad_sequences(sequences_train, maxlen=max_len)
sequences_test = pad_sequences(sequences_test, maxlen=max_len)
print(sequences_train[2])

# Créer le réseau de neurones pour de classifier les textes
model = Sequential()

model.add(Embedding(10000, 16, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(sequences_train, labels_train, epochs=10, batch_size=32, validation_split=0.2)

# Évaluer le modèle sur l'ensemble de test
loss, accuracy = model.evaluate(sequences_test, labels_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(sequences_test)

#test sur le dernier texte du fichier csv
# Séparer le dernier texte du fichier
dernier_texte = textes[44205]  # Dernier texte de la première colonne

# Tokenization et séquençage du dernier texte
sequence_dernier_texte = tokenizer.texts_to_sequences([dernier_texte])
sequence_dernier_texte = pad_sequences(sequence_dernier_texte, maxlen=max_len)

# Prédire si le dernier texte est généré par l'IA
prediction_dernier = model.predict(sequence_dernier_texte)[0][0]
print("Dernier texte du fichier :")
print(dernier_texte)
# Afficher le résultat de la prédiction
if prediction_dernier >= 0.5:
    print("Le dernier texte est généré par l'IA.")
else:
    print("Le dernier texte n'est pas généré par l'IA.")

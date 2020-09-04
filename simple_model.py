import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import pandas as pd
from model import Model

LEARNING_RATE = 0.01
EPOCHS = 200
BATCH_SIZE = 5

ERROR_THRESHOLD = 0.6

# stems word
def stem(word):
    return LancasterStemmer().stem(word)

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(sentence, words):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stem(word.lower()) for word in sentence_words]
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for (i, w) in enumerate(words):
            if (w == s):
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)

class SimpleModel(Model):
    # processes word returning documents, classes and words
    def process(self, intents, verbose=True):
        words = []
        classes = []
        documents = []
        ignore_words = ['?']
        # loop through each sentence in our intents patterns
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                w = nltk.word_tokenize(pattern)
                # add to our words list
                words.extend(w)
                # add to documents in our corpus
                documents.append((w, intent['tag']))
                # add to our classes list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        # stem and lower each word and remove duplicates
        words = [stem(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))

        # remove duplicates
        classes = sorted(list(set(classes)))

        if (verbose):
            print(len(documents), "documents")
            print(len(classes), "classes", classes)
            print(len(words), "unique stemmed words", words)

        return documents, classes, words

    # transforms documents, classes and words into training data train_x, train_y
    def transform(self, documents, classes, words):
        # create our training data
        training = []
        # create an empty array for our output
        output_empty = [0] * len(classes)

        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in words:
                if (w in pattern_words):
                    bag.append(1)
                else:
                    bag.append(0)

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training, dtype=object)

        # create train and test lists
        train_x = list(training[:,0])
        train_y = list(training[:,1])

        return train_x, train_y

    # generates a tensorflow keras model dependant on the training data
    def generate_model(self, train_x, train_y):
        # 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        # equal to number of intents to predict output intent with softmax

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

        # Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = tf.keras.optimizers.SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # Fit the model
        model.fit(np.array(train_x), np.array(train_y), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, use_multiprocessing=True)

        return model

    # classifies a sentence into one of the given classes
    def classify(self, sentence, classes, model, *args, verbose=True):
        # process softmax output
        input_data = pd.DataFrame([bag_of_words(sentence, args[0])], dtype=float, index=['input'])
        results = model.predict([input_data])[0]

        if (verbose):
            d = list(zip(classes, results))
            d.sort(key=lambda x: x[1], reverse=True)
            print(d)

        results = [[i , r] for i , r in enumerate(results) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        if (len(results) == 0):
            return ""
        return classes[results[0][0]]
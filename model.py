class Model:
    def __init__(self):
        pass

    def process(self, intents, verbose=True):
        """
        Processes intents returning documents, classes and words
        """

    def transform(self, documents, classes, words):
        """
        Transforms documents, classes and words into training data train_x, train_y
        """

    def generate_model(self, train_x, train_y):
        """
        Generates a tensorflow keras model dependant on the training data
        """

    def classify(self, sentence, classes, model, *args, verbose=True):
        """
        Classifies a sentence into one of the given classes
        """
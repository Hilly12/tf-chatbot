import json
import random
from simple_model import SimpleModel as Model

with open('tf_chatbot/intents.json') as json_data:
    intents = json.load(json_data)

model = Model()

documents, classes, words = model.process(intents, verbose=True)
train_x, train_y = model.transform(documents, classes, words)

classifier = model.generate_model(train_x, train_y)

print("Hello :-)")
# The following loop will execute each time the user enters input
while True:
    try:
        text = input("> ")
        tag = model.classify(text, classes, classifier, words)
        response = "I did not understand"
        for intent in intents['intents']:
            if (intent['tag'] == tag):
                response = random.choice(intent['responses'])
        print(response)
    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break
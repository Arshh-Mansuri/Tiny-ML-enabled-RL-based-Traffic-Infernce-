import numpy as np
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from collections import deque
import random
from keras.optimizers import Adam
import h5py


class Learner:
    def __init__(self, state_space_size, action_space_size, exploration, tflite_model_path):
        self.state_size = state_space_size
        self.action_size = action_space_size
        self.learning_rate = 0.001
        self.firstHidden = 604
        self.secondHidden = 1166
        #self.regressor = self._build_model()
        self.exploration = exploration
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.memory = deque(maxlen=2000)
        self.batch_size = 200
        self.gamma = 0.95
        self.load_tflite_model('/home/arsh/BTEP/DITLACS_Git/CodeBase/7_TF_Lite_Inference/keras_model.tflite')

    '''def _build_model(self):
        regressor = Sequential()
        regressor.add(Dense(units=self.firstHidden, input_dim=self.state_size, activation='relu'))
        regressor.add(Dense(units=self.secondHidden, activation='relu'))
        regressor.add(Dense(units=self.action_size, activation='linear'))
        regressor.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return regressor'''

    def act(self, state):
        input_details = self.tflite_interpreter.get_input_details()
        output_details = self.tflite_interpreter.get_output_details()

        # Prepare input for TFLite model
        input_state = [state[0].astype(np.float32), state[1].astype(np.float32), state[2].astype(np.float32)]
        self.tflite_interpreter.set_tensor(input_details[0]['index'], input_state)

        # Perform inference
        self.tflite_interpreter.invoke()

        # Get the output
        output_action = self.tflite_interpreter.get_tensor(output_details[0]['index'])

        # Choose the action
        action = np.argmax(output_action)

        return action

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        minibatch = random.sample(list(self.memory), self.batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * np.max(self.regressor.predict(next_state)[0])
            target_f = self.regressor.predict(state)
            target_f[0][action] = target
            self.regressor.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration > self.min_exploration:
            self.exploration *= self.exploration_decay

    def load(self, name):
        self.regressor.load_weights(name)

    def save(self, name):
        self.regressor.save_weights(name)

    def save_keras_model(self, model_name):
        # Save Keras model architecture and weights
        model_json = self.regressor.to_json()
        with open(f"{model_name}.json", "w") as json_file:
            json_file.write(model_json)
        self.regressor.save_weights(f"{model_name}.h5")

    def convert_to_tflite(self, model_name):
        # Load the saved Keras model
        with open(f"{model_name}.json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(f"{model_name}.h5")

        # Convert the Keras model to TensorFlow Lite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        # Save the TensorFlow Lite model
        with open(f"{model_name}.tflite", "wb") as tflite_file:
            tflite_file.write(tflite_model)

    def load_tflite_model(self, tflite_model_path):
        self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.tflite_interpreter.allocate_tensors()


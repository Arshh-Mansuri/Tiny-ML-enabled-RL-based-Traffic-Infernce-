import numpy as np
import keras
from keras.models import Sequential,model_from_json
from keras.layers import Dense
from collections import deque
import random
from keras.optimizers import Adam
import h5py
import tensorflow as tf
import tensorflow_model_optimization as tfmot

class Learner:
    def __init__(self, state_space_size, action_space_size, exploration):
        self.state_size = state_space_size
        self.action_size = action_space_size
        self.learning_rate = 0.001
        self.firstHidden = 604
        self.secondHidden = 1166
        self.regressor = self._build_model()
        self.exploration = exploration
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.memory = deque(maxlen=2000)
        self.batch_size = 200
        self.gamma = 0.95

    def _build_model(self):
        # Define your model
        model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.state_size,)),
                tf.keras.layers.Dense(self.firstHidden, activation='relu'),
                tf.keras.layers.Dense(self.secondHidden, activation='relu'),
                tf.keras.layers.Dense(self.action_size, activation='linear')
                ])

        #  Compile the model
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
        # Apply pruning to the dense layers
        pruned_dense1 = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(self.firstHidden, activation='relu'))
        pruned_dense2 = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(self.secondHidden, activation='relu'))
        pruned_output_layer = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(self.action_size, activation='linear'))


        # Define quantization annotation for the dense layers
        # Annotate each dense layer with quantization annotation
        annotated_dense1 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(self.firstHidden, activation='relu'))
        annotated_dense2 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(self.secondHidden, activation='relu'))
        annotated_output_layer = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(self.action_size, activation='linear'))


        # Apply quantization to make the model quantization-aware
        # Create the model with annotated layers
        quant_aware_model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=(self.state_size,)),
                            annotated_dense1,
                            annotated_dense2,
                            annotated_output_layer
                            ])
        return quant_aware_model
        
        
        
        

    def act(self, state):
        if np.random.rand() <= self.exploration:
            action = np.random.choice(range(self.action_size))
        else:
            action = np.argmax(self.regressor.predict(state), axis=1)[0]
        return action

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        minibatch = random.sample(list(self.memory), self.batch_size)
        for state, action, reward, next_state in minibatch:
            # print "Reward: {}".format(type(reward))
            target = reward + self.gamma*np.max(self.regressor.predict(next_state)[0])
            target_f = self.regressor.predict(state)
            # print target_f
            # print target
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
       try:
          # Load the saved Keras model weights
          self.regressor.load_weights(f"{model_name}.h5")

          # Convert the Keras model to TensorFlow Lite with quantization
          converter = tf.lite.TFLiteConverter.from_keras_model(self.regressor)
          converter.optimizations = [tf.lite.Optimize.DEFAULT]
          tflite_model = converter.convert()

          # Save the TensorFlow Lite model
          with open(f"{model_name}.tflite", "wb") as tflite_file:
               tflite_file.write(tflite_model)

       except FileNotFoundError:
              print(f"Error: Model weights file {model_name}.h5 not found.")
       except Exception as e:
              print(f"Error: {e}")




        
   

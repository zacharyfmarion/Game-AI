import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model, clone_model

from gameai.core import Network


class BasicConvNet(Network):
    '''
    Basic convolutional network

     Attributes:
        input_shape (tuple): Shape of the input
        policy_size (int): Shape of the policy output
        num_layers (int): Number of convolutional layers in the network
        model (tf.keras.Model): Keras model

    Examples:
        >>> BasicConvNet(input_shape=(3,3,1), policy_size=(9,), num_layers=1)
    '''

    def __init__(self, weights=None, input_shape=None, policy_size=None, num_layers=3):
        self.input_shape = input_shape
        self.policy_size = policy_size
        self.num_layers = num_layers

        input_layer = Input(shape=input_shape)
        layer = input_layer
        for _ in range(num_layers):
            layer = Conv2D(10, (1, 1), padding='same',
                           activation='relu')(layer)
        flatten = Flatten()(layer)
        dense1 = Dense(10)(flatten)
        dense2 = Dense(10)(dense1)
        policy = Dense(policy_size, activation='softmax')(dense2)
        reward = Dense(1, activation="tanh")(dense2)

        self.model = Model(inputs=input_layer, outputs=[policy, reward])
        self.model.compile(optimizer='rmsprop',
                           loss=['categorical_crossentropy',
                                 'mean_squared_error'],
                           metrics=['accuracy'])
        if weights:
            self.model.set_weights(weights)

    def train(self, examples, **kwargs):
        '''
        Train the network

        Args:
            examples (list): list of the format [state, policy, reward]
        '''

        num_epochs = kwargs.get('num_epochs', 50)
        batch_size = kwargs.get('batch_size', 32)

        input_shape = list(self.input_shape)
        input_shape.insert(0, -1)

        x_states = np.array([state for [state, _, _] in examples]).reshape(
            tuple(input_shape))
        y_policies = np.array([policy for [_, policy, _] in examples])
        y_rewards = np.array([reward for [_, _, reward] in examples])
        self.model.fit(x_states, [y_policies, y_rewards],
                       epochs=num_epochs, batch_size=batch_size, verbose=1)

    def predict(self, inputs, **kwargs):
        input_shape = list(self.input_shape)
        input_shape.insert(0, -1)
        reshaped_inputs = np.array(inputs).reshape(tuple(input_shape))
        return tuple(self.model.predict(reshaped_inputs))

    def predict_single(self, x, **kwargs):
        policies, rewards = self.predict(
            np.expand_dims(np.array(x), 1), **kwargs)
        return (policies[0], rewards[0][0])

    def weights(self):
        return self.model.get_weights()

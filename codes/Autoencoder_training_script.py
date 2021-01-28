import pickle
import gym
import cv2
import numpy as np
from keras import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt

def preprocess(observation):
    grayscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(grayscale, (53, 70))
    cropped = resize[8:65, 3:50]  # [from top : to bottom, from left : to right]
    _, thresh1 = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY)
    normalized = thresh1/255
    return normalized

env = gym.make('SpaceInvaders-v0')
observation = env.reset()
done = False

trainingdata = np.empty([43200, 57, 47])
env.action_space.seed(0)
for i in range(len(trainingdata)):
    observation, _, done, _ = env.step(env.action_space.sample())
    trainingdata[i] = preprocess(observation)
    if done is True:
        env.reset()
env.close()

testdata = trainingdata[0:14400]
trainingdata = trainingdata [14401:]
np.random.shuffle(testdata)
np.random.shuffle(trainingdata)
testdata = testdata.astype('float32')
trainingdata = trainingdata.astype('float32')
testdata = testdata.reshape((len(testdata), np.prod(testdata.shape[1:])))
trainingdata = trainingdata.reshape((len(trainingdata), np.prod(trainingdata.shape[1:])))

in_layer     = Input(shape=(2679,))

encode_layer = Dense(1024, activation='relu')(in_layer)
encode_layer = Dense(512, activation='relu')(encode_layer)
encode_layer = Dense(128, activation='relu')(encode_layer)
encode_layer = Dense(6, activation='relu')(encode_layer)

decode_layer = Dense(128, activation='relu')(encode_layer)
decode_layer = Dense(512, activation='relu')(decode_layer)
decode_layer = Dense(1024, activation='relu')(decode_layer)
out_layer = Dense(2679, activation='sigmoid')(decode_layer)

autoencoder = Model(in_layer, out_layer)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
autoencoder.fit(trainingdata, trainingdata,
                epochs=64,
                batch_size=512,
                shuffle=True,
                validation_data=(testdata, testdata))

autoencoder.summary()

with open('SI-AE-model.pkl', 'wb') as output:
    pickle.dump(autoencoder, output)

features = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_4').output)
features_output = features.predict(testdata)
predicted = autoencoder.predict(testdata)

morefeatures = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_3').output)
morefeatures_output = morefeatures.predict(testdata)

plt.style.use('grayscale')
plt.figure(figsize=(12, 6), dpi=300)
for i in range(10):
    # display original images
    ax = plt.subplot(4, 10, i + 1)
    plt.imshow(testdata[i].reshape(57, 47))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded images
    ax = plt.subplot(4, 10, i + 1 + 10)
    plt.imshow(morefeatures_output[i].reshape(16, 8))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, 10, i + 1 + 20)
    plt.imshow(features_output[i].reshape(2, 3))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstructed images
    ax = plt.subplot(4, 10, i + 1 + 30)
    plt.imshow(predicted[i].reshape(57, 47))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('SIAE_performance.svg')




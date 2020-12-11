import tensorflow as tf
import tensorflow_io as tfio
import datetime
import matplotlib.pyplot as plt
import more_itertools as mit

audio = tfio.audio.AudioIOTensor('/Users/alexanderlenhardt/src/github.com/flipper/tests/data/seg.mp3')
audio_tensor = tf.squeeze(audio[:44100, 1])
tensor = tf.cast(audio_tensor, tf.float32) / tf.reduce_max(tf.abs(audio_tensor))

spectrogram = tfio.experimental.audio.spectrogram(
    tensor, nfft=512, window=512, stride=128)

mel = tfio.experimental.audio.melscale(spectrogram, rate=44100, mels=64, fmin=100, fmax=8000)
db = tfio.experimental.audio.dbscale(spectrogram, top_db=80)

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(9, 3))
ax = axes.flatten()
ax[0].plot(tensor)
ax[0].set_aspect('auto')
img = ax[1].imshow(db.numpy().T, origin='lower')
ax[1].set_aspect('auto')
plt.colorbar(img)
plt.show()

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


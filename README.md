# Keras Ordinal Categorical Crossentropy Loss Function

This is a Keras implementation of a loss function for ordinal datasets, based on the built-in categorical crossentropy loss.

The assumption is that the relationship between any two consecutive categories is uniform, for example, `{[1, 0, 0, 0], [0, 0, 1, 0]}` will be penalised to the same extent as `{[0, 1, 0, 0], [0, 0, 0, 1]}`, where `{x, y}` are the `(true, prediction)` pairs.

## Requirements

* Python 3.6.0
* Keras 2.0.6
* TensorFlow 1.2.1 or Theano 0.9.0

**Note** These are the versions that I used. To my knowledge, there's no reason that this shouldn't work with slightly earlier versions.

## Usage

This a simple drop-in loss function, and can be used exactly the same as any custom loss function. Simply import `ordinal_categorical_crossentropy`:

```
import ordinal_categorical_crossentropy as OCC
```
Then when compiling the model, set the loss function to `OCC.loss`:
```
model.compile(loss=OCC.loss, optimizer='adam', metrics=['accuracy'])
```

## Example

This uses a modified example from the Keras documentation <https://keras.io/getting-started/sequential-model-guide/#examples>:

```
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from ordinal_categorical_crossentropy import OCC

import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=OCC.loss, optimizer=sgd, metrics=['accuracy']) ### This is where you use the loss function.

model.fit(x_train, y_train, epochs=20, batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

```

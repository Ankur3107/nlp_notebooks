<a href="https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/knowledge_distillation_exploration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```


```
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
```

    25000 Training sequences
    25000 Validation sequences



```
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = student
        self.student = teacher

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
```


```
# Create the teacher
# Input for variable-length sequences of integers
inputs = keras.Input(shape=(maxlen,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(vocab_size, 128)(inputs)
# Add 2 bidirectional LSTMs and GRUs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(128))(x)
# Add a classifier
outputs = layers.Dense(2)(x)
teacher = keras.Model(inputs, outputs, name='teacher')
teacher.summary()


# Create the student
# Input for variable-length sequences of integers
inputs = keras.Input(shape=(maxlen,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(vocab_size, 64)(inputs)
# Add 2 bidirectional LSTMs and GRUs
x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(32))(x)
# Add a classifier
outputs = layers.Dense(2)(x)
student = keras.Model(inputs, outputs, name='student')
student.summary()

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)
```

    Model: "teacher"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 200)]             0         
    _________________________________________________________________
    embedding (Embedding)        (None, 200, 128)          2560000   
    _________________________________________________________________
    bidirectional (Bidirectional (None, 200, 128)          98816     
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 256)               198144    
    _________________________________________________________________
    dense (Dense)                (None, 2)                 514       
    =================================================================
    Total params: 2,857,474
    Trainable params: 2,857,474
    Non-trainable params: 0
    _________________________________________________________________
    Model: "student"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 200)]             0         
    _________________________________________________________________
    embedding_1 (Embedding)      (None, 200, 64)           1280000   
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 200, 64)           24832     
    _________________________________________________________________
    bidirectional_3 (Bidirection (None, 64)                18816     
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 130       
    =================================================================
    Total params: 1,323,778
    Trainable params: 1,323,778
    Non-trainable params: 0
    _________________________________________________________________



```
# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate teacher on data.
teacher.fit(x_train, y_train, epochs=3)
teacher.evaluate(x_val, y_val)
```

    Epoch 1/3
    782/782 [==============================] - 43s 55ms/step - loss: 0.3714 - sparse_categorical_accuracy: 0.8351
    Epoch 2/3
    782/782 [==============================] - 43s 55ms/step - loss: 0.2004 - sparse_categorical_accuracy: 0.9232
    Epoch 3/3
    782/782 [==============================] - 42s 54ms/step - loss: 0.1297 - sparse_categorical_accuracy: 0.9519
    782/782 [==============================] - 12s 15ms/step - loss: 0.4922 - sparse_categorical_accuracy: 0.8590





    [0.49218621850013733, 0.8590400218963623]




```
# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
distiller.fit(x_train, y_train, epochs=3)

# Evaluate student on test dataset
distiller.evaluate(x_val, y_val)
```

    Epoch 1/3
    782/782 [==============================] - 53s 68ms/step - sparse_categorical_accuracy: 0.9664 - student_loss: 0.1382 - distillation_loss: 0.0084
    Epoch 2/3
    782/782 [==============================] - 54s 69ms/step - sparse_categorical_accuracy: 0.9781 - student_loss: 0.1119 - distillation_loss: 0.0085
    Epoch 3/3
    782/782 [==============================] - 54s 69ms/step - sparse_categorical_accuracy: 0.9868 - student_loss: 0.0937 - distillation_loss: 0.0088
    782/782 [==============================] - 12s 15ms/step - sparse_categorical_accuracy: 0.8476 - student_loss: 0.3908





    0.8475599884986877




```
# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, y_train, epochs=3)

```

    Epoch 1/3
    782/782 [==============================] - 31s 39ms/step - loss: 0.3796 - sparse_categorical_accuracy: 0.8246
    Epoch 2/3
    782/782 [==============================] - 30s 39ms/step - loss: 0.1918 - sparse_categorical_accuracy: 0.9283
    Epoch 3/3
    782/782 [==============================] - 30s 39ms/step - loss: 0.1123 - sparse_categorical_accuracy: 0.9602



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-7-2e22fcfd29d9> in <module>()
          8 # Train and evaluate student trained from scratch.
          9 student_scratch.fit(x_train, y_train, epochs=3)
    ---> 10 student_scratch.evaluate(x_test, y_test)
    

    NameError: name 'x_test' is not defined



```
student_scratch.evaluate(x_val, y_val)
```

    782/782 [==============================] - 12s 15ms/step - loss: 0.3720 - sparse_categorical_accuracy: 0.8596





    [0.37198370695114136, 0.8595600128173828]




```

```

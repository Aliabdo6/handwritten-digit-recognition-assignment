# Handwritten Digit Recognition Assignment

This repository contains my solutions for the first assignment in **Course 2** of the **Advanced Learning Algorithms** section from the **Machine Learning Specialization** by **Stanford University** and **DeepLearning.AI**.

## Assignment Overview

In this assignment, I implemented a neural network for recognizing handwritten digits from the MNIST dataset using both **TensorFlow** and **NumPy**. This allowed me to explore both the high-level functionality of TensorFlow and the underlying mechanics by implementing certain aspects manually.

### Tasks

1. **Build a Neural Network using TensorFlow**
2. **Implement Forward Propagation for a Dense Layer using NumPy (Single Example)**
3. **Vectorize Forward Propagation for Multiple Examples using NumPy**

---

## Task 1: TensorFlow Neural Network

The goal of the first task was to construct a simple neural network using **TensorFlow**. I defined a sequential model with three dense layers, all using the sigmoid activation function:

```python
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),    # Input layer with 400 features
        Dense(25, activation="sigmoid", name="layer1"),
        Dense(15, activation="sigmoid", name="layer2"),
        Dense(1, activation="sigmoid", name="layer3")  # Output layer
    ], name="my_model"
)

model.summary()
```

- **Input Layer**: 400 features (28x28 images flattened)
- **Hidden Layer 1**: 25 units, sigmoid activation
- **Hidden Layer 2**: 15 units, sigmoid activation
- **Output Layer**: 1 unit, sigmoid activation for binary classification

---

## Task 2: Dense Layer Forward Propagation (NumPy - Single Example)

In this task, I implemented the forward pass for a dense layer using **NumPy**. The function computes the output for a single input example by performing a weighted sum and applying the activation function.

```python
def my_dense(a_in, W, b, g):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out
```

- **Inputs**:
  - `a_in`: Input example (features)
  - `W`: Weight matrix
  - `b`: Bias vector
  - `g`: Activation function (e.g., sigmoid)

---

## Task 3: Vectorized Dense Layer Forward Propagation (NumPy - Multiple Examples)

The third task extended the previous function to work with multiple examples. I used matrix multiplication to efficiently compute the output for a batch of input examples.

```python
def my_dense_v(A_in, W, b, g):
    Z = np.matmul(A_in, W) + b
    A_out = g(Z)
    return A_out
```

- **Inputs**:
  - `A_in`: Input batch (multiple examples)
  - `W`: Weight matrix
  - `b`: Bias vector
  - `g`: Activation function (e.g., sigmoid)

---

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aliabdo6/handwritten-digit-recognition-assignment.git
   ```

2. Install the required dependencies:

   ```bash
   pip install tensorflow numpy
   ```

3. Open the notebook and run the code:

   ```bash
   jupyter notebook assignment.ipynb
   ```

---

## Acknowledgments

This assignment is part of the **Machine Learning Specialization** by **Stanford University** and **DeepLearning.AI**.

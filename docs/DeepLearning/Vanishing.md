<h2 style="color:red;">✅ Vanishing and Exploding Gradients Problems</h2>

<h3 style="color:blue;">📌 What is Vanishing Gradient?</h3>
 
 The vanishing gradient problem is a challenge that emerges during backpropagation when the derivatives or slopes of the activation functions become progressively smaller as we move backward through the layers of a neural network. This phenomenon is particularly prominent in deep networks with many layers, hindering the effective training of the model. The weight updates becomes extremely tiny, or even exponentially small, it can significantly prolong the training time, and in the worst-case scenario, it can halt the training process altogether.

 **Why the Problem Occurs?**

 During backpropagation, the gradients propagate back through the layers of the network, they decrease significantly. This means that as they leave the output layer and return to the input layer, the gradients become progressively smaller. As a result, the weights associated with the initial levels, which accommodate these small gradients, are updated little or not at each iteration of the optimization process.

 The **vanishing gradient problem** is particularly associated with the **sigmoid** and **hyperbolic tangent (tanh)** activation functions because their derivatives fall within the range of 0 to 0.25 and 0 to 1, respectively. Consequently, extreme weights becomes very small, causing the updated weights to closely resemble the original ones. This persistence of small updates contributes to the vanishing gradient issue.

 The sigmoid and tanh functions limit the input values ​​to the ranges [0,1] and [-1,1], so that they saturate at 0 or 1 for sigmoid and -1 or 1 for Tanh. The derivatives at points becomes zero as they are moving. In these regions, especially when inputs are very small or large, the gradients are very close to zero. 


 When training deep neural networks (especially RNNs, LSTMs, and deep feedforward networks), we use backpropagation to update weights.

 The update depends on **gradients** — numbers that tell us how much to change the weights.

 If these gradients become **very small** as they move backward through the layers, they can “vanish” (approach zero).

 When this happens:

 - The earlier layers **barely get updated**.

 - The network **learns very slowly** or **stops learning**.

**Why it Happens?**

In backpropagation, gradients are multiplied many times by the **derivative of the activation function**.

For example:

- If the derivative is small (e.g., 0.1), multiplying it through 50 layers gives:

![alt text](./images/Vanishing1.png)

- Sigmoid and tanh activations **squash** numbers between small ranges, so their derivatives are small, which worsens the problem.


## Real-Life Example: The Whisper Game

Imagine a group of **50 people** standing in a line.

1. **Person 1** whispers a message: “The train leaves at 8.”

2. Each person passes it on **quietly** to the next.

3. By the time it reaches **Person 50**, the message becomes: 
    
    “T...ain...8” (almost lost).

**Why?**

- Every person slightly loses information.

- The **earlier the person**, the more the message is lost before it reaches the end.

**This is the same as vanishing gradients:**

- The “message” = gradient information.

- Passing through people = layers in the neural network.

- Whispering quietly = multiplying by small derivatives (like sigmoid output’s slope).


**Impact on RNNs**

RNNs process sequences step by step (like passing the message from one person to the next).

- If gradients vanish, **early time steps** (earlier words in a sentence) get forgotten.


**How We Solve It**

- Use **ReLU** instead of sigmoid/tanh (avoids tiny derivatives).

- Use **LSTM** or **GRU** (special gates to keep gradients flowing).

- Use Batch Normalization or Residual Connections.
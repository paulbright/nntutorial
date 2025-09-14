#!/usr/bin/env python3
"""
Neural Network Tutorial: Forward and Backward Propagation

This tutorial demonstrates forward and backward propagation in a neural network with:
- 3 inputs: x1 (science grade), x2 (chemistry grade), x3 (study hours)
- 1 hidden layer with 2 neurons (sigmoid activation)
- 1 output neuron (predicting math grade yp)
- Mean Square Error loss function
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset: student number, science grade (x1), chemistry grade (x2), study hours (x3), math grade (ya)
data = np.array([
    [1, 60, 80, 5, 81],
    [2, 70, 75, 7, 94],
    [3, 50, 55, 10, 45],
    [4, 40, 56, 7, 43]
])

X = data[:, 1:4]  # x1, x2, x3
ya = data[:, 4]   # actual math grades

print("Dataset:")
print("Student | x1 (Science) | x2 (Chemistry) | x3 (Hours) | ya (Math)")
print("-" * 60)
for i, row in enumerate(data):
    print(f"   {int(row[0])}    |      {int(row[1])}      |       {int(row[2])}       |     {int(row[3])}     |    {int(row[4])}")

# Initialize weights and biases as specified in the prompt
w1, w2 = 0.1, 0.15   # x1 to hidden neurons 1 and 2
w3, w4 = 0.1, 0.05   # x2 to hidden neurons 1 and 2
w5, w6 = 0.1, -0.2   # x3 to hidden neurons 1 and 2
b1, b2 = -15, -15    # Hidden layer biases
w7, w8 = 12, 9       # Output layer weights
b3 = 20              # Output layer bias
lr = 0.01            # Learning rate

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    return x * (1 - x)

def mse_loss(ya, yp):
    """Mean Square Error loss function"""
    return np.mean((ya - yp) ** 2)

# Storage for tracking progress
loss_history = []
weight_history = []

print(f"\nInitial Weights: w1={w1}, w2={w2}, w3={w3}, w4={w4}, w5={w5}, w6={w6}, w7={w7}, w8={w8}")
print(f"Initial Biases: b1={b1}, b2={b2}, b3={b3}")

# Training for 3 epochs
epochs = 3

print("\n" + "=" * 80)
print("NEURAL NETWORK TRAINING - STEP BY STEP")
print("=" * 80)

for epoch in range(epochs):
    print(f"\n{'='*20} EPOCH {epoch + 1} {'='*20}")
    
    epoch_loss = 0
    
    for i, (x1, x2, x3, target) in enumerate(zip(X[:, 0], X[:, 1], X[:, 2], ya)):
        print(f"\n--- Sample {i+1}: x1={x1}, x2={x2}, x3={x3}, target={target} ---")
        
        # FORWARD PROPAGATION
        print("\n🔄 FORWARD PROPAGATION:")
        
        # Hidden layer calculations
        z1 = w1*x1 + w3*x2 + w5*x3 + b1
        z2 = w2*x1 + w4*x2 + w6*x3 + b2
        print(f"z1 = {w1}×{x1} + {w3}×{x2} + {w5}×{x3} + {b1} = {z1:.4f}")
        print(f"z2 = {w2}×{x1} + {w4}×{x2} + {w6}×{x3} + {b2} = {z2:.4f}")
        
        g1 = sigmoid(z1)
        g2 = sigmoid(z2)
        print(f"g1 = sigmoid({z1:.4f}) = {g1:.4f}")
        print(f"g2 = sigmoid({z2:.4f}) = {g2:.4f}")
        
        # Output layer
        yp = w7*g1 + w8*g2 + b3
        print(f"yp = {w7}×{g1:.4f} + {w8}×{g2:.4f} + {b3} = {yp:.4f}")
        
        # Loss calculation
        loss = (target - yp) ** 2
        epoch_loss += loss
        print(f"Loss = ({target} - {yp:.4f})² = {loss:.4f}")
        
        # BACKWARD PROPAGATION
        print("\n⬅️ BACKWARD PROPAGATION (Chain Rule):")
        
        # Output layer gradients
        dL_dyp = -2 * (target - yp)
        print(f"∂L/∂yp = -2×({target} - {yp:.4f}) = {dL_dyp:.4f}")
        
        dL_dw7 = dL_dyp * g1
        dL_dw8 = dL_dyp * g2
        dL_db3 = dL_dyp
        print(f"∂L/∂w7 = {dL_dyp:.4f} × {g1:.4f} = {dL_dw7:.4f}")
        print(f"∂L/∂w8 = {dL_dyp:.4f} × {g2:.4f} = {dL_dw8:.4f}")
        print(f"∂L/∂b3 = {dL_db3:.4f}")
        
        # Hidden layer gradients
        dL_dg1 = dL_dyp * w7
        dL_dg2 = dL_dyp * w8
        print(f"∂L/∂g1 = {dL_dyp:.4f} × {w7} = {dL_dg1:.4f}")
        print(f"∂L/∂g2 = {dL_dyp:.4f} × {w8} = {dL_dg2:.4f}")
        
        dL_dz1 = dL_dg1 * sigmoid_derivative(g1)
        dL_dz2 = dL_dg2 * sigmoid_derivative(g2)
        print(f"∂L/∂z1 = {dL_dg1:.4f} × {sigmoid_derivative(g1):.4f} = {dL_dz1:.4f}")
        print(f"∂L/∂z2 = {dL_dg2:.4f} × {sigmoid_derivative(g2):.4f} = {dL_dz2:.4f}")
        
        # Input layer gradients
        dL_dw1 = dL_dz1 * x1
        dL_dw2 = dL_dz2 * x1
        dL_dw3 = dL_dz1 * x2
        dL_dw4 = dL_dz2 * x2
        dL_dw5 = dL_dz1 * x3
        dL_dw6 = dL_dz2 * x3
        dL_db1 = dL_dz1
        dL_db2 = dL_dz2
        
        print(f"∂L/∂w1 = {dL_dz1:.4f} × {x1} = {dL_dw1:.4f}")
        print(f"∂L/∂w2 = {dL_dz2:.4f} × {x1} = {dL_dw2:.4f}")
        print(f"∂L/∂w3 = {dL_dz1:.4f} × {x2} = {dL_dw3:.4f}")
        print(f"∂L/∂w4 = {dL_dz2:.4f} × {x2} = {dL_dw4:.4f}")
        print(f"∂L/∂w5 = {dL_dz1:.4f} × {x3} = {dL_dw5:.4f}")
        print(f"∂L/∂w6 = {dL_dz2:.4f} × {x3} = {dL_dw6:.4f}")
        
        # GRADIENT DESCENT UPDATE
        print("\n📈 GRADIENT DESCENT UPDATE:")
        old_weights = [w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3]
        
        # Update weights and biases
        w1 -= lr * dL_dw1
        w2 -= lr * dL_dw2
        w3 -= lr * dL_dw3
        w4 -= lr * dL_dw4
        w5 -= lr * dL_dw5
        w6 -= lr * dL_dw6
        w7 -= lr * dL_dw7
        w8 -= lr * dL_dw8
        b1 -= lr * dL_db1
        b2 -= lr * dL_db2
        b3 -= lr * dL_db3
        
        new_weights = [w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3]
        weight_names = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'b1', 'b2', 'b3']
        
        print("Weight updates:")
        for name, old, new in zip(weight_names, old_weights, new_weights):
            change = new - old
            print(f"{name}: {old:.4f} → {new:.4f} (Δ{change:+.4f})")
    
    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(X)
    loss_history.append(avg_loss)
    
    weight_history.append({
        'epoch': epoch + 1,
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6,
        'w7': w7, 'w8': w8, 'b1': b1, 'b2': b2, 'b3': b3, 'loss': avg_loss
    })
    
    print(f"\n🎯 EPOCH {epoch + 1} SUMMARY:")
    print(f"Average Loss: {avg_loss:.4f}")

print("\n" + "=" * 80)
print("TRAINING COMPLETED!")
print("=" * 80)

# Display weight and bias changes table
df = pd.DataFrame(weight_history)
print("\n📊 WEIGHT AND BIAS PROGRESSION TABLE:")
print("=" * 100)
print(df.round(4))

# Plot gradient descent progress
plt.figure(figsize=(12, 8))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(range(1, epochs + 1), loss_history, 'ro-', linewidth=3, markersize=10)
plt.title('Loss During Training\n(Gradient Descent Progress)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Square Error', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, epochs + 1))
for i, loss in enumerate(loss_history):
    plt.annotate(f'{loss:.2f}', (i+1, loss), textcoords="offset points", 
                xytext=(0,15), ha='center', fontsize=10, fontweight='bold')

# Weight evolution plots
plt.subplot(2, 2, 2)
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, [h['w7'] for h in weight_history], 'b-o', label='w7', linewidth=2)
plt.plot(epochs_range, [h['w8'] for h in weight_history], 'g-o', label='w8', linewidth=2)
plt.title('Output Layer Weights', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Weight Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(epochs_range, [h['b1'] for h in weight_history], 'r-o', label='b1', linewidth=2)
plt.plot(epochs_range, [h['b2'] for h in weight_history], 'm-o', label='b2', linewidth=2)
plt.plot(epochs_range, [h['b3'] for h in weight_history], 'c-o', label='b3', linewidth=2)
plt.title('Bias Evolution', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Bias Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
hidden_weights = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow']
for i, weight in enumerate(hidden_weights):
    values = [h[weight] for h in weight_history]
    plt.plot(epochs_range, values, f'{colors[i][0]}-', marker='o', label=weight, linewidth=2)
plt.title('Hidden Layer Weights', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Weight Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final predictions and analysis
print("\n🎯 FINAL PREDICTIONS AND ANALYSIS:")
print("=" * 60)

total_error = 0
for i, (x1, x2, x3, target) in enumerate(zip(X[:, 0], X[:, 1], X[:, 2], ya)):
    # Forward pass with final weights
    z1 = w1*x1 + w3*x2 + w5*x3 + b1
    z2 = w2*x1 + w4*x2 + w6*x3 + b2
    g1 = sigmoid(z1)
    g2 = sigmoid(z2)
    yp = w7*g1 + w8*g2 + b3
    
    error = abs(target - yp)
    total_error += error
    
    print(f"Student {i+1}: Actual={target:2d}, Predicted={yp:6.2f}, Error={error:5.2f}")
    print(f"           z1={z1:6.2f}, z2={z2:6.2f}, g1={g1:6.4f}, g2={g2:6.4f}")

avg_error = total_error / len(X)
print(f"\nAverage Absolute Error: {avg_error:.2f}")
print(f"Final Loss: {loss_history[-1]:.4f}")

if __name__ == "__main__":
    print("\nScript completed successfully!")

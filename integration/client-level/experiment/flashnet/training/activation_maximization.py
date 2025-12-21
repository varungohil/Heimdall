#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def activation_maximization_neuron(model, layer_idx, neuron_idx, input_shape, feature_means=None, iterations=500, learning_rate=1.0, regularization=0.01, other_neurons_regularization=0.1, feature_0_value=0.0):
    """
    Generate an input that maximally activates a specific neuron in a Dense layer.
    
    This function ONLY works with Dense layers. It maximizes the output of the specified
    Dense layer neuron. For models with separate Dense and Activation layers, this maximizes
    the pre-activation values (linear transformation outputs before the activation function).
    
    Args:
        model: The trained Keras model
        layer_idx: Index of the Dense layer (must be a Dense layer)
        neuron_idx: Index of the neuron in that layer
        input_shape: Shape of the input (without batch dimension)
        feature_means: Mean values of features from training data (optional)
        iterations: Number of gradient ascent iterations
        learning_rate: Learning rate for gradient ascent
        regularization: L2 regularization weight to keep inputs reasonable
        other_neurons_regularization: Weight for regularization term that penalizes activation of other neurons in the same layer
        feature_0_value: Fixed value for Feature 0 (0.0 or 1.0)
    
    Returns:
        Tuple of (maximizing_input, loss_history)
        - maximizing_input: The input that maximally activates the neuron
        - loss_history: List of loss values at each iteration
    
    Raises:
        ValueError: If the specified layer is not a Dense layer
    """
    # Validate that the layer is a Dense layer
    layer = model.layers[layer_idx]
    if 'Dense' not in str(type(layer)):
        raise ValueError(f"Layer {layer_idx} ({layer.name}) is not a Dense layer. Activation maximization only works with Dense layers.")
    
    # Create a model that outputs the Dense layer's output
    layer_output = model.layers[layer_idx].output
    activation_model = keras.Model(inputs=model.input, outputs=layer_output)
    
    # Initialize with mean values from training data if available, otherwise random
    if feature_means is not None:
        initial_input = feature_means.reshape(1, -1).copy()
    else:
        initial_input = np.random.random((1, input_shape)) * 0.5 + 0.25
    
    # Ensure no feature is zero; if any feature is 0, set it to 0.5
    initial_input[initial_input == 0.0] = 0.5
    # Set Feature 0 to the fixed value
    initial_input[0, 0] = feature_0_value
    input_data = tf.Variable(initial_input, dtype=tf.float32)
    
    # Track loss history
    loss_history = []
    
    # Gradient ascent loop
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            # Apply constraints:
            # 1. No negative values (clip to >= 0)
            input_data.assign(tf.maximum(input_data, 0.0))
            # 2. Feature 0 is fixed to the specified value
            constrained_input = input_data.numpy()
            constrained_input[0, 0] = feature_0_value
            input_data.assign(constrained_input)   
   
            activation = activation_model(input_data)
            
            # Extract the specific neuron's activation from the Dense layer output
            # Dense layers output 2D tensors: [batch, neurons]
            if len(activation.shape) == 2:  # Dense layer: [batch, neurons]
                neuron_activation = activation[0, neuron_idx]
            elif len(activation.shape) == 1:  # Single neuron Dense layer: [batch]
                if neuron_idx != 0:
                    raise ValueError(f"Neuron index {neuron_idx} is invalid for a single-neuron Dense layer")
                neuron_activation = activation[0]
            else:
                # Flatten and extract (shouldn't happen for Dense layers, but handle gracefully)
                flat_activation = tf.reshape(activation, [-1])
                neuron_activation = flat_activation[neuron_idx]
            
            # Compute activations of other neurons in the same Dense layer as regularizer
            # We want to minimize the activations of other neurons while maximizing the target neuron
            # Dense layers output 2D tensors: [batch, neurons]
            if len(activation.shape) == 2:  # Standard Dense layer: [batch, neurons]
                num_neurons = activation.shape[-1]
                activation_flat = activation[0]
            elif len(activation.shape) == 1:  # Single neuron Dense layer: [batch]
                num_neurons = activation.shape[0]
                activation_flat = activation
            else:
                # Flatten (shouldn't happen for Dense layers, but handle gracefully)
                activation_flat = tf.reshape(activation, [-1])
                num_neurons = activation_flat.shape[0]
            
            if num_neurons > 1:
                # Create a mask to select all neurons except the target neuron
                mask = tf.ones(num_neurons, dtype=tf.float32)
                mask = tf.tensor_scatter_nd_update(mask, [[neuron_idx]], [0.0])
                
                # Get activations of other neurons
                other_neurons_activation = activation_flat * mask
                
                # Regularization term: L1 norm (sum of absolute values) of other neurons' activations
                other_neurons_loss = tf.reduce_sum(tf.abs(other_neurons_activation))
            else:
                other_neurons_loss = 0.0
            
            # Loss is negative activation (we want to maximize, so minimize negative)
            # Add regularization for other neurons and L2 regularization for inputs
            loss = -neuron_activation + other_neurons_regularization * other_neurons_loss #+ regularization * tf.reduce_sum(tf.square(input_data))
        
        # Store loss value
        loss_history.append(float(loss.numpy()))
        
        # Compute gradients
        gradients = tape.gradient(loss, input_data)
        
        # Update input using gradient descent (negative because we minimize negative activation)
        input_data.assign_add(-learning_rate * gradients)
        

    
    return input_data.numpy()[0], loss_history

def get_all_layer_activations(model, input_data):
    """
    Run inference and extract activations from all layers.
    
    Args:
        model: The trained Keras model
        input_data: Input data (numpy array with batch dimension)
    
    Returns:
        Dictionary mapping layer indices to their activation values
    """
    activations = {}
    
    # Create models that output each layer's activations
    for layer_idx, layer in enumerate(model.layers):
        # Only get activations for Dense layers
        if 'Dense' in str(type(layer)):
            layer_model = keras.Model(inputs=model.input, outputs=layer.output)
            layer_activation = layer_model(input_data).numpy()
            activations[layer_idx] = layer_activation[0]  # Remove batch dimension
    
    return activations

def generate_filename_suffix(other_neurons_regularization, iterations, learning_rate, initial_point, feature_0_value):
    """
    Generate a filename suffix that includes all command-line arguments.
    
    Args:
        other_neurons_regularization: Weight for regularization term
        iterations: Number of gradient ascent iterations
        learning_rate: Learning rate for gradient ascent
        initial_point: Initialization method
        feature_0_value: Fixed value for Feature 0
    
    Returns:
        String suffix to append to filenames
    """
    # Format: onr<value>_iter<value>_lr<value>_init<value>_f0<value>
    suffix = f"onr{other_neurons_regularization}_iter{iterations}_lr{learning_rate}_init{initial_point}_f0{int(feature_0_value)}"
    return suffix

def run_activation_maximization(model_path, dataset_path=None, other_neurons_regularization=0.1, iterations=5000, learning_rate=1.0, initial_point="random", feature_0_value=0.0):
    """
    Load a trained model and run activation maximization for all neurons.
    
    Args:
        model_path: Path to the saved model.keras file
        dataset_path: Path to the training dataset CSV file (optional, required if initial_point="means")
        other_neurons_regularization: Weight for regularization term that penalizes activation of other neurons in the same layer
        iterations: Number of gradient ascent iterations
        learning_rate: Learning rate for gradient ascent
        initial_point: Initialization method - "random" for random initialization, "means" for feature means from dataset
        feature_0_value: Fixed value for Feature 0 (0.0 or 1.0)
    """
    # Generate filename suffix from all parameters
    filename_suffix = generate_filename_suffix(other_neurons_regularization, iterations, learning_rate, initial_point, feature_0_value)
    # Load the model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Get input shape from the model
    input_shape = model.input_shape[1]  # Exclude batch dimension
    print(f"Model input shape: {input_shape}")
    
    # Print model summary
    model.summary()
    
    # Load training data and compute normalized feature means based on initial_point setting
    feature_means = None
    if initial_point == "means":
        if dataset_path is None:
            raise ValueError("initial_point='means' requires -dataset to be provided")
        print(f"\nLoading training data from: {dataset_path}")
        try:
            dataset = pd.read_csv(dataset_path)
            
            # Apply same preprocessing as in training (from nnK.py)
            # Remove reject and latency columns
            x = dataset.copy(deep=True).drop(columns=["reject"], axis=1)
            if "latency" in x.columns:
                x = x.drop(columns=["latency"], axis=1)
            y = dataset['reject']
            
            # Split train/test using the same approach as training (50/50 split with random_state=42)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
            
            # Fit scaler on training data and normalize (same as in nnK.py)
            scaler = MinMaxScaler()
            scaler.fit(x_train)
            
            # Compute means from training data and normalize them
            feature_means_raw = x_train.mean().values
            feature_means = scaler.transform(feature_means_raw.reshape(1, -1))[0]
            
            print(f"Computed normalized feature means from training data: shape {feature_means.shape}")
            print(f"Feature means (raw): {feature_means_raw}")
            print(f"Feature means (normalized): {feature_means}")
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
            print("Will use random initialization instead.")
            feature_means = None
    else:  # initial_point == "random"
        print("\nUsing random initialization.")
    
    # Create output directory (same as model directory)
    output_dir = os.path.dirname(model_path)
    activation_output_dir = os.path.join(output_dir, "activation_maximization")
    os.makedirs(activation_output_dir, exist_ok=True)
    print(f"Output directory: {activation_output_dir}")
    
    # Dictionary to store all maximizing inputs
    all_maximizing_inputs = {}
    
    # Iterate through each layer
    # IMPORTANT: Only processes Dense layers. Activation maximization maximizes the outputs
    # of Dense layers (pre-activation values for models with separate Activation layers).
    for layer_idx, layer in enumerate(model.layers):
        layer_config = layer.get_config()
        layer_name = layer.name
        
        # Only process Dense layers with more than 0 units
        if 'Dense' in str(type(layer)):
            num_neurons = layer.units
            print(f"\nProcessing Layer {layer_idx}: {layer_name} with {num_neurons} neurons")
            
            layer_results = []
            
            # Iterate through each neuron in the layer
            for neuron_idx in range(num_neurons):
                print(f"  Neuron {neuron_idx}/{num_neurons-1}...", end='')
                
                # Run activation maximization
                maximizing_input, loss_history = activation_maximization_neuron(
                    model=model,
                    layer_idx=layer_idx,
                    neuron_idx=neuron_idx,
                    input_shape=input_shape,
                    feature_means=feature_means,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    regularization=0.01,
                    other_neurons_regularization=other_neurons_regularization,
                    feature_0_value=feature_0_value
                )
                
                layer_results.append(maximizing_input)
                
                # Save individual neuron's maximizing input
                neuron_csv_filename = os.path.join(activation_output_dir, f"layer_{layer_idx}_neuron_{neuron_idx}_maximizing_input_{filename_suffix}.csv")
                np.savetxt(neuron_csv_filename, maximizing_input.reshape(1, -1), delimiter=',')
                
                # Save loss history with iteration numbers
                loss_csv_filename = os.path.join(activation_output_dir, f"layer_{layer_idx}_neuron_{neuron_idx}_loss_history_{filename_suffix}.csv")
                loss_data = np.column_stack([np.arange(len(loss_history)), loss_history])
                np.savetxt(loss_csv_filename, loss_data, delimiter=',', header='iteration,loss', comments='')
                
                # Run inference with the maximizing input and log all activations
                input_batch = maximizing_input.reshape(1, -1)  # Add batch dimension
                all_activations = get_all_layer_activations(model, input_batch)
                
                # Save activations for all layers
                activations_csv_filename = os.path.join(activation_output_dir, f"layer_{layer_idx}_neuron_{neuron_idx}_all_activations_{filename_suffix}.csv")
                with open(activations_csv_filename, 'w') as f:
                    f.write("layer_idx,neuron_idx,activation_value\n")
                    for act_layer_idx, act_values in all_activations.items():
                        for act_neuron_idx, act_value in enumerate(act_values):
                            f.write(f"{act_layer_idx},{act_neuron_idx},{act_value}\n")
                
                print(f" Saved to {os.path.basename(neuron_csv_filename)} (+ loss history + activations)")
            
            # Convert to numpy array and save combined file for the layer
            layer_results_array = np.array(layer_results)
            all_maximizing_inputs[f"layer_{layer_idx}"] = layer_results_array
            
            # Save combined CSV file for all neurons in this layer
            csv_filename = os.path.join(activation_output_dir, f"layer_{layer_idx}_all_neurons_maximizing_inputs_{filename_suffix}.csv")
            np.savetxt(csv_filename, layer_results_array, delimiter=',')
            print(f"Saved layer {layer_idx} combined results to: {csv_filename}")
            print(f"Shape: {layer_results_array.shape}")
    
    # Create a summary visualization
    print("\n=== Activation Maximization Summary ===")
    for layer_name, inputs in all_maximizing_inputs.items():
        print(f"{layer_name}: {inputs.shape[0]} neurons, input dimension: {inputs.shape[1]}")
    
    # Save a combined summary file
    summary_file = os.path.join(activation_output_dir, f"summary_{filename_suffix}.txt")
    with open(summary_file, 'w') as f:
        f.write("Activation Maximization Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Input dimension: {input_shape}\n")
        f.write(f"\nParameters:\n")
        f.write(f"  other_neurons_regularization: {other_neurons_regularization}\n")
        f.write(f"  iterations: {iterations}\n")
        f.write(f"  learning_rate: {learning_rate}\n")
        f.write(f"  initial_point: {initial_point}\n")
        f.write(f"  feature_0_value: {feature_0_value}\n")
        f.write(f"\n")
        for layer_name, inputs in all_maximizing_inputs.items():
            f.write(f"{layer_name}: {inputs.shape[0]} neurons\n")
        f.write("\n")
        f.write("Files generated:\n")
        for layer_idx, (layer_name, inputs) in enumerate(all_maximizing_inputs.items()):
            f.write(f"  Layer {layer_idx}:\n")
            f.write(f"    - {layer_name}_all_neurons_maximizing_inputs_{filename_suffix}.csv (combined)\n")
            for neuron_idx in range(inputs.shape[0]):
                f.write(f"    - layer_{layer_idx}_neuron_{neuron_idx}_maximizing_input_{filename_suffix}.csv\n")
                f.write(f"    - layer_{layer_idx}_neuron_{neuron_idx}_loss_history_{filename_suffix}.csv\n")
                f.write(f"    - layer_{layer_idx}_neuron_{neuron_idx}_all_activations_{filename_suffix}.csv\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"All results saved to: {activation_output_dir}")
    
    return all_maximizing_inputs

def visualize_neuron_preferences(model_path, feature_names=None, filename_suffix=None):
    """
    Optional: Visualize which features each neuron prefers.
    
    Args:
        model_path: Path to the saved model
        feature_names: List of feature names (optional)
        filename_suffix: Suffix to match files with specific parameters (optional)
    """
    output_dir = os.path.dirname(model_path)
    activation_output_dir = os.path.join(output_dir, "activation_maximization")
    
    # Check if activation maximization has been run
    if not os.path.exists(activation_output_dir):
        print("Run activation maximization first!")
        return
    
    # Load results and create visualizations
    # Only load the combined maximizing input files, not loss history files
    if filename_suffix:
        pattern = f'_all_neurons_maximizing_inputs_{filename_suffix}.csv'
        layer_files = sorted([f for f in os.listdir(activation_output_dir) 
                              if f.endswith(pattern)])
    else:
        # Fallback: try to find files with any suffix or without suffix
        layer_files = sorted([f for f in os.listdir(activation_output_dir) 
                              if '_all_neurons_maximizing_inputs' in f and f.endswith('.csv')])
    
    for layer_file in layer_files:
        layer_data = np.loadtxt(os.path.join(activation_output_dir, layer_file), delimiter=',')
        
        if len(layer_data.shape) == 1:
            layer_data = layer_data.reshape(1, -1)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(4, layer_data.shape[0] // 10)))
        im = ax.imshow(layer_data, aspect='auto', cmap='viridis')
        ax.set_xlabel('Input Features')
        ax.set_ylabel('Neurons')
        ax.set_title(f'Activation Maximization: {layer_file.replace(".csv", "")}')
        plt.colorbar(im, ax=ax, label='Normalized Input Value')
        
        # Save figure with same suffix
        if filename_suffix:
            fig_path = os.path.join(activation_output_dir, layer_file.replace('.csv', '.png'))
        else:
            # Remove suffix if present, or just replace .csv with .png
            base_name = layer_file.replace('.csv', '')
            if '_all_neurons_maximizing_inputs_' in base_name:
                base_name = base_name.split('_all_neurons_maximizing_inputs_')[0] + '_all_neurons_maximizing_inputs'
            fig_path = os.path.join(activation_output_dir, base_name + '.png')
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved visualization: {fig_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run activation maximization on a trained FlashNet model')
    parser.add_argument("-model", help="Path to the saved model.keras file", type=str, required=True)
    parser.add_argument("-dataset", help="Path to the training dataset CSV file (required if initial_point=means)", type=str, default=None)
    parser.add_argument("-visualize", help="Also create visualizations", action='store_true')
    parser.add_argument("-other_neurons_regularization", help="Weight for regularization term that penalizes activation of other neurons in the same layer", type=float, default=0.1)
    parser.add_argument("-iterations", help="Number of gradient ascent iterations", type=int, default=5000)
    parser.add_argument("-learning_rate", help="Learning rate for gradient ascent", type=float, default=1.0)
    parser.add_argument("-initial_point", help="Initialization method: 'random' for random initialization, 'means' for feature means from dataset", type=str, choices=["random", "means"], default="random")
    parser.add_argument("-feature_0", help="Fixed value for Feature 0 (0 or 1)", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    
    if not args.model:
        print("ERROR: You must provide -model <path to model.keras>")
        sys.exit(-1)
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(-1)
    
    if args.initial_point == "means" and args.dataset is None:
        print("ERROR: initial_point='means' requires -dataset to be provided")
        sys.exit(-1)
    
    if args.dataset and not os.path.exists(args.dataset):
        if args.initial_point == "means":
            print(f"ERROR: Dataset file not found: {args.dataset}")
            print("Cannot proceed with initial_point='means' without a valid dataset.")
            sys.exit(-1)
        else:
            print(f"WARNING: Dataset file not found: {args.dataset}")
            print("Will use random initialization instead.")
            args.dataset = None
    
    # Generate filename suffix for consistency
    filename_suffix = generate_filename_suffix(
        args.other_neurons_regularization,
        args.iterations,
        args.learning_rate,
        args.initial_point,
        float(args.feature_0)
    )
    
    # Run activation maximization
    results = run_activation_maximization(args.model, dataset_path=args.dataset, other_neurons_regularization=args.other_neurons_regularization, iterations=args.iterations, learning_rate=args.learning_rate, initial_point=args.initial_point, feature_0_value=float(args.feature_0))
    
    # Optionally create visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        visualize_neuron_preferences(args.model, filename_suffix=filename_suffix)
    
    print("\n=== Activation Maximization Complete ===")


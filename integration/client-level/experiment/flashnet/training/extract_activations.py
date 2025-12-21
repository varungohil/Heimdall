#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def extract_all_activations(model_path, dataset_path, train_eval_split):
    """
    Load a trained model and extract activations of all neurons for all training samples.
    
    Args:
        model_path: Path to the saved model.keras file
        dataset_path: Path to the training dataset CSV file
        train_eval_split: Train/eval split ratio (e.g., "80_20")
    """
    # Parse split ratio
    ratios = train_eval_split.split("_")
    percent_data_for_training = int(ratios[0])
    percent_data_for_eval = int(ratios[1])
    assert(percent_data_for_training + percent_data_for_eval == 100)
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Load and preprocess the dataset (same as in nnK.py training)
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = pd.read_csv(dataset_path)
    
    # Reorder columns to put "latency" at the end
    reordered_cols = [col for col in dataset.columns if col != "latency"] + ["latency"]
    dataset = dataset[reordered_cols]
    
    # Split test and training set (using same random state as training)
    x = dataset.copy(deep=True).drop(columns=["reject"], axis=1)
    y = dataset['reject']
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=percent_data_for_eval/100, random_state=42
    )
    
    # Remove the latency column from the input features
    x_train = x_train.drop(columns=["latency"], axis=1)
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Input features: {x_train.shape[1]}")
    
    # Apply the same normalization as during training
    # Note: We fit on training data to match what was done during training
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_norm = scaler.transform(x_train)
    
    print("\nExtracting activations from all layers...")
    
    # Create a model that outputs activations from all layers
    # We want the output of each Dense layer
    layer_outputs = []
    layer_names = []
    
    for i, layer in enumerate(model.layers):
        if 'Dense' in str(type(layer)):
            layer_outputs.append(layer.output)
            layer_names.append(f"layer_{i}")
            print(f"  Layer {i}: {layer.name} ({layer.units} neurons)")
    
    # Create the activation extraction model
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Run inference on all training samples
    print("\nRunning inference on all training samples...")
    activations = activation_model.predict(x_train_norm, verbose=1)
    
    # Build column names for the CSV
    # Format: layer_<layer_idx>_neuron_<neuron_idx>
    columns = []
    for layer_idx, layer_name in enumerate(layer_names):
        layer_activations = activations[layer_idx]
        num_neurons = layer_activations.shape[1] if len(layer_activations.shape) > 1 else 1
        
        if num_neurons == 1:
            columns.append(f"layer_{layer_idx}_neuron_0")
        else:
            for neuron_idx in range(num_neurons):
                columns.append(f"layer_{layer_idx}_neuron_{neuron_idx}")
    
    print(f"\nTotal neurons across all layers: {len(columns)}")
    print(f"Total samples: {x_train_norm.shape[0]}")
    
    # Combine all activations into a single array
    # Each row is a sample, each column is a neuron's activation
    all_activations = []
    
    for layer_idx, layer_activations in enumerate(activations):
        if len(layer_activations.shape) == 1:
            # Single neuron output
            all_activations.append(layer_activations.reshape(-1, 1))
        else:
            all_activations.append(layer_activations)
    
    # Concatenate all layer activations horizontally
    combined_activations = np.hstack(all_activations)
    
    print(f"Combined activations shape: {combined_activations.shape}")
    
    # Create DataFrame with proper column names
    activations_df = pd.DataFrame(combined_activations, columns=columns)
    
    # Optionally, add the input features and ground truth labels
    # Create DataFrame from training data
    x_train_df = pd.DataFrame(x_train.values, columns=x_train.columns)
    x_train_df.reset_index(drop=True, inplace=True)
    
    # Add ground truth labels
    y_train_series = pd.Series(y_train.values, name='ground_truth_reject')
    y_train_series.reset_index(drop=True, inplace=True)
    
    # Combine everything: input features, activations, and ground truth
    full_df = pd.concat([x_train_df, activations_df, y_train_series], axis=1)
    
    # Determine output path (same directory as model)
    output_dir = os.path.dirname(model_path)
    output_csv_path = os.path.join(output_dir, "neuron_activations.csv")
    
    # Save to CSV
    print(f"\nSaving activations to: {output_csv_path}")
    full_df.to_csv(output_csv_path, index=False)
    
    print(f"✓ Successfully saved {full_df.shape[0]} samples with {full_df.shape[1]} columns")
    print(f"  - Input features: {x_train.shape[1]} columns")
    print(f"  - Neuron activations: {len(columns)} columns")
    print(f"  - Ground truth: 1 column")
    
    # Also save a summary file
    summary_path = os.path.join(output_dir, "neuron_activations_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Neuron Activations Extraction Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Train/Eval Split: {train_eval_split}\n")
        f.write(f"\nTraining samples: {x_train.shape[0]}\n")
        f.write(f"Input features: {x_train.shape[1]}\n")
        f.write(f"\nModel Architecture:\n")
        for i, layer_name in enumerate(layer_names):
            layer_activations = activations[i]
            num_neurons = layer_activations.shape[1] if len(layer_activations.shape) > 1 else 1
            f.write(f"  Layer {i}: {num_neurons} neurons\n")
        f.write(f"\nTotal neurons: {len(columns)}\n")
        f.write(f"\nOutput CSV structure:\n")
        f.write(f"  - Rows: {full_df.shape[0]} (training samples)\n")
        f.write(f"  - Columns: {full_df.shape[1]} total\n")
        f.write(f"    * Input features: {x_train.shape[1]}\n")
        f.write(f"    * Neuron activations: {len(columns)}\n")
        f.write(f"    * Ground truth: 1\n")
        f.write(f"\nColumn naming convention:\n")
        f.write(f"  - Input features: original names from dataset\n")
        f.write(f"  - Neuron activations: layer_<idx>_neuron_<idx>\n")
        f.write(f"  - Ground truth: ground_truth_reject\n")
        f.write(f"\nOutput file: {output_csv_path}\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    
    # Print first few column names as a sample
    print(f"\nSample column names (first 10 activation columns):")
    activation_cols = [col for col in full_df.columns if col.startswith('layer_')]
    for col in activation_cols[:10]:
        print(f"  - {col}")
    if len(activation_cols) > 10:
        print(f"  ... and {len(activation_cols) - 10} more")
    
    return full_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract neuron activations from a trained FlashNet model for all training samples'
    )
    parser.add_argument(
        "-model", 
        help="Path to the saved model.keras file", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "-dataset", 
        help="Path to the training dataset CSV file", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "-train_eval_split", 
        help="Train/eval split ratio (e.g., '80_20') - must match the split used during training", 
        type=str, 
        required=True
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(-1)
    
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset file not found: {args.dataset}")
        sys.exit(-1)
    
    # Run activation extraction
    try:
        extract_all_activations(args.model, args.dataset, args.train_eval_split)
        print("\n=== Activation Extraction Complete ===")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(-1)


#!/bin/bash

# Script to run activation maximization on saved Keras models
# Usage Option 1: ./run_activation_maximization_all.sh device0 device1 dir1 [dir2 ...]
# Usage Option 2: ./run_activation_maximization_all.sh --auto [data_directory]

echo "=========================================="
echo "Activation Maximization - Batch Runner"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define parameter arrays for cross product
ITERATIONS=(500 5000)
OTHER_NEURONS_REG=(0.1 0.5 1)
LEARNING_RATES=(0.1 0.01 1)
INITIAL_POINTS=("means" "random")
FEATURE_0_VALUES=(0 1)

# Function to run activation maximization with all parameter combinations
run_activation_maximization_all_combinations() {
    local model_path=$1
    local dataset_path=$2
    
    local total_combinations=$((${#ITERATIONS[@]} * ${#OTHER_NEURONS_REG[@]} * ${#LEARNING_RATES[@]} * ${#INITIAL_POINTS[@]} * ${#FEATURE_0_VALUES[@]}))
    local current_combination=0
    local success_count=0
    local fail_count=0
    
    echo "Running activation maximization with $total_combinations parameter combinations..."
    echo ""
    
    for iterations in "${ITERATIONS[@]}"; do
        for other_neurons_reg in "${OTHER_NEURONS_REG[@]}"; do
            for learning_rate in "${LEARNING_RATES[@]}"; do
                for initial_point in "${INITIAL_POINTS[@]}"; do
                    for feature_0 in "${FEATURE_0_VALUES[@]}"; do
                        current_combination=$((current_combination + 1))
                        
                        echo "  Combination $current_combination/$total_combinations:"
                        echo "    iterations=$iterations, other_neurons_reg=$other_neurons_reg, learning_rate=$learning_rate, initial_point=$initial_point, feature_0=$feature_0"
                        
                        # Skip if initial_point is "means" but dataset is not available
                        if [ "$initial_point" == "means" ] && [ -z "$dataset_path" ]; then
                            echo "    ⚠ Skipped (initial_point='means' requires dataset, but dataset not found)"
                            fail_count=$((fail_count + 1))
                            echo ""
                            continue
                        fi
                        
                        # Build command
                        local cmd="python3 \"$SCRIPT_DIR/activation_maximization.py\" -visualize -model \"$model_path\""
                        cmd="$cmd -iterations $iterations"
                        cmd="$cmd -other_neurons_regularization $other_neurons_reg"
                        cmd="$cmd -learning_rate $learning_rate"
                        cmd="$cmd -initial_point $initial_point"
                        cmd="$cmd -feature_0 $feature_0"
                        
                        # Add dataset only if initial_point is "means" and dataset_path is provided
                        if [ "$initial_point" == "means" ] && [ -n "$dataset_path" ] && [ -f "$dataset_path" ]; then
                            cmd="$cmd -dataset \"$dataset_path\""
                        fi
                        
                        # Run the command
                        eval $cmd
                        
                        if [ $? -eq 0 ]; then
                            success_count=$((success_count + 1))
                        else
                            fail_count=$((fail_count + 1))
                            echo "    ✗ Failed"
                        fi
                        echo ""
                    done
                done
            done
        done
    done
    
    echo "  Summary for this model:"
    echo "    Total combinations: $total_combinations"
    echo "    Success: $success_count"
    echo "    Failed: $fail_count"
    echo ""
    
    # Return 0 if all succeeded, 1 if any failed
    if [ $fail_count -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Check if running in auto mode or manual mode
if [ "$#" -eq 0 ]; then
    echo "ERROR: No arguments provided"
    echo ""
    echo "Usage Option 1 (Manual): $0 device0 device1 dir1 [dir2 ...]"
    echo "  Example: $0 nvme0n1 nvme1n1 /path/to/trace1 /path/to/trace2"
    echo ""
    echo "Usage Option 2 (Auto): $0 --auto [data_directory]"
    echo "  Example: $0 --auto"
    echo "  Example: $0 --auto /path/to/data"
    exit 1
fi

# Auto mode - search entire data directory
if [ "$1" == "--auto" ]; then
    echo "Mode: AUTO - Searching entire data directory"
    echo ""
    
    # Set data directory
    if [ "$#" -eq 2 ]; then
        DATA_DIR="$2"
    else
        # Default to the standard data directory relative to this script
        DATA_DIR="$SCRIPT_DIR/../../../data"
    fi
    
    # Resolve to absolute path
    DATA_DIR=$(realpath "$DATA_DIR")
    echo "Data Directory: $DATA_DIR"
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    # Find all model.keras files
    echo "Searching for saved models..."
    model_files=$(find "$DATA_DIR" -name "model.keras" -type f)
    
    if [ -z "$model_files" ]; then
        echo "ERROR: No model.keras files found in $DATA_DIR"
        exit 1
    fi
    
    # Count and list models
    model_count=$(echo "$model_files" | wc -l)
    echo "Found $model_count model(s)"
    echo ""
    
    # Process each model
    current=0
    success_count=0
    fail_count=0
    
    for model_path in $model_files; do
        current=$((current + 1))
        echo "=========================================="
        echo "Processing model $current of $model_count"
        echo "=========================================="
        echo "Model: $model_path"
        
        # Extract the model directory
        model_dir=$(dirname "$model_path")
        
        # Determine dataset path
        dataset_path=""
        if [[ "$model_path" == *"/mldrive0/"* ]]; then
            training_results_dir=$(echo "$model_path" | sed 's|/mldrive0/.*||')
            dataset_path="$training_results_dir/mldrive0.csv"
        elif [[ "$model_path" == *"/mldrive1/"* ]]; then
            training_results_dir=$(echo "$model_path" | sed 's|/mldrive1/.*||')
            dataset_path="$training_results_dir/mldrive1.csv"
        fi
        
        # Check if dataset exists
        if [ -n "$dataset_path" ] && [ -f "$dataset_path" ]; then
            echo "Dataset: $dataset_path"
        else
            echo "WARNING: Dataset not found"
            dataset_path=""
        fi
        
        # Run activation maximization with all parameter combinations
        echo ""
        run_activation_maximization_all_combinations "$model_path" "$dataset_path"
        
        if [ $? -eq 0 ]; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
        echo ""
    done
    
    echo "=========================================="
    echo "Batch Processing Complete"
    echo "=========================================="
    echo "Total models processed:  $model_count"
    echo "Models with all combinations successful: $success_count"
    echo "Models with some failures: $fail_count"
    echo ""
    echo "Note: Each model was run with 72 parameter combinations"
    echo "      (2 iterations × 3 regularization × 3 learning rates × 2 initial points × 2 feature_0 values)"
    echo ""
    
else
    # Manual mode - process specific directories
    echo "Mode: MANUAL - Processing specified directories"
    echo ""
    
    # Check minimum arguments
    if [ "$#" -lt 3 ]; then
        echo "ERROR: Insufficient arguments for manual mode"
        echo "Usage: $0 device0 device1 dir1 [dir2 ...]"
        exit 1
    fi
    
    device0=$1
    device1=$2
    directories=("${@:3}")
    
    echo "Device 0: $device0"
    echo "Device 1: $device1"
    echo "Directories to process: ${#directories[@]}"
    echo ""
    
    idx=1
    total_dirs=${#directories[@]}
    success_count=0
    fail_count=0
    
    for dir in "${directories[@]}"; do
        echo "=========================================="
        echo "Processing directory $idx of $total_dirs"
        echo "=========================================="
        echo "Trace directory: $dir"
        
        # Check if directory exists
        if [ ! -d "$dir" ]; then
            echo "ERROR: Directory not found: $dir"
            fail_count=$((fail_count + 1))
            idx=$((idx + 1))
            echo ""
            continue
        fi
        
        # Define training results directory
        training_result_dir="$dir/$device0...$device1/flashnet/training_results"
        
        # Check if training results exist
        if [ ! -d "$training_result_dir" ]; then
            echo "WARNING: Training results not found: $training_result_dir"
            echo "Skipping this directory..."
            fail_count=$((fail_count + 1))
            idx=$((idx + 1))
            echo ""
            continue
        fi
        
        echo "Training results: $training_result_dir"
        echo ""
        
        # Process each mldrive model (typically mldrive0 and mldrive1)
        for mldrive_idx in 0 1; do
            model_path="$training_result_dir/mldrive${mldrive_idx}/nnK/model.keras"
            dataset_path="$training_result_dir/mldrive${mldrive_idx}.csv"
            
            if [ ! -f "$model_path" ]; then
                echo "  Model not found: $model_path (skipping)"
                continue
            fi
            
            echo "  Processing mldrive${mldrive_idx}..."
            echo "    Model:   $model_path"
            
            if [ -f "$dataset_path" ]; then
                echo "    Dataset: $dataset_path"
            else
                echo "    Dataset: Not found"
                dataset_path=""
            fi
            
            # Run activation maximization with all parameter combinations
            echo ""
            run_activation_maximization_all_combinations "$model_path" "$dataset_path"
            
            if [ $? -eq 0 ]; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
            echo ""
        done
        
        idx=$((idx + 1))
    done
    
    echo "=========================================="
    echo "Batch Processing Complete"
    echo "=========================================="
    echo "Directories processed: $total_dirs"
    echo "Models with all combinations successful: $success_count"
    echo "Models with some failures: $fail_count"
    echo ""
    echo "Note: Each model was run with 72 parameter combinations"
    echo "      (2 iterations × 3 regularization × 3 learning rates × 2 initial points × 2 feature_0 values)"
    echo ""
fi

echo "Results are saved in activation_maximization/ subdirectories"
echo "within each model's directory."

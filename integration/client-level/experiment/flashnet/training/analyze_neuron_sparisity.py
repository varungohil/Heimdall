#!/usr/bin/env python3
"""
Script to analyze neuron activation sparsity in the neuron_activations.csv file.
Reports the percentage of zero neurons for each training sample and aggregate statistics.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path


def analyze_neuron_sparsity(csv_file, output_file=None, chunk_size=10000):
    """
    Analyze neuron activation sparsity from the CSV file.
    
    Args:
        csv_file: Path to the neuron_activations.csv file
        output_file: Optional path to save detailed per-sample results
        chunk_size: Number of rows to process at a time for memory efficiency
    """
    print(f"Analyzing neuron activations from: {csv_file}")
    print(f"Processing in chunks of {chunk_size} rows...\n")
    
    # First, read the header to identify neuron columns
    df_header = pd.read_csv(csv_file, nrows=0)
    columns = df_header.columns.tolist()
    
    # Identify neuron columns (those starting with 'layer_')
    neuron_columns = [col for col in columns if col.startswith('layer_')]
    total_neurons = len(neuron_columns)
    
    print(f"Total neurons detected: {total_neurons}")
    print(f"Neuron columns: {neuron_columns[:5]}...{neuron_columns[-5:]}\n")
    
    # Process the file in chunks for memory efficiency
    all_zero_percentages = []
    sample_count = 0
    
    # Open output file if specified
    if output_file:
        output_fh = open(output_file, 'w')
        output_fh.write("sample_id,zero_neuron_count,zero_neuron_percentage\n")
    
    chunk_iterator = pd.read_csv(csv_file, chunksize=chunk_size)
    
    for chunk_id, chunk_df in enumerate(chunk_iterator):
        # Extract only neuron activation columns
        neuron_data = chunk_df[neuron_columns].values
        
        # Count zeros for each sample (row)
        zero_counts = (neuron_data == 0.0).sum(axis=1)
        
        # Calculate percentage of zeros for each sample
        zero_percentages = (zero_counts / total_neurons) * 100.0
        
        all_zero_percentages.extend(zero_percentages)
        
        # Write per-sample results if output file is specified
        if output_file:
            for i, (zero_count, zero_pct) in enumerate(zip(zero_counts, zero_percentages)):
                sample_id = sample_count + i
                output_fh.write(f"{sample_id},{zero_count},{zero_pct:.4f}\n")
        
        sample_count += len(chunk_df)
        
        # Progress update
        if (chunk_id + 1) % 10 == 0:
            print(f"Processed {sample_count} samples...")
    
    if output_file:
        output_fh.close()
        print(f"\nDetailed results saved to: {output_file}")
    
    # Calculate aggregate statistics
    zero_percentages_array = np.array(all_zero_percentages)
    
    print(f"\n{'='*70}")
    print("AGGREGATE STATISTICS - Zero Neuron Percentages")
    print(f"{'='*70}")
    print(f"Total samples analyzed: {sample_count}")
    print(f"Total neurons per sample: {total_neurons}")
    print()
    print(f"Mean:                   {zero_percentages_array.mean():.4f}%")
    print(f"Median:                 {np.median(zero_percentages_array):.4f}%")
    print(f"Standard Deviation:     {zero_percentages_array.std():.4f}%")
    print(f"Minimum:                {zero_percentages_array.min():.4f}%")
    print(f"Maximum:                {zero_percentages_array.max():.4f}%")
    print()
    print("Percentiles:")
    print(f"  25th percentile:      {np.percentile(zero_percentages_array, 25):.4f}%")
    print(f"  50th percentile:      {np.percentile(zero_percentages_array, 50):.4f}%")
    print(f"  75th percentile:      {np.percentile(zero_percentages_array, 75):.4f}%")
    print(f"  90th percentile:      {np.percentile(zero_percentages_array, 90):.4f}%")
    print(f"  95th percentile:      {np.percentile(zero_percentages_array, 95):.4f}%")
    print(f"  99th percentile:      {np.percentile(zero_percentages_array, 99):.4f}%")
    print()
    
    # Distribution of sparsity levels
    print("Distribution of Zero Neuron Percentages:")
    ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), 
              (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    
    for low, high in ranges:
        count = np.sum((zero_percentages_array >= low) & (zero_percentages_array < high))
        percentage = (count / sample_count) * 100.0
        print(f"  {low:3d}% - {high:3d}%:  {count:8d} samples ({percentage:6.2f}%)")
    
    # Samples with 100% zeros
    count_100 = np.sum(zero_percentages_array == 100.0)
    percentage_100 = (count_100 / sample_count) * 100.0
    print(f"  100%:         {count_100:8d} samples ({percentage_100:6.2f}%)")
    
    print(f"{'='*70}\n")
    
    return {
        'total_samples': sample_count,
        'total_neurons': total_neurons,
        'mean': zero_percentages_array.mean(),
        'median': np.median(zero_percentages_array),
        'std': zero_percentages_array.std(),
        'min': zero_percentages_array.min(),
        'max': zero_percentages_array.max(),
        'percentiles': {
            25: np.percentile(zero_percentages_array, 25),
            50: np.percentile(zero_percentages_array, 50),
            75: np.percentile(zero_percentages_array, 75),
            90: np.percentile(zero_percentages_array, 90),
            95: np.percentile(zero_percentages_array, 95),
            99: np.percentile(zero_percentages_array, 99),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze neuron activation sparsity from CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze the CSV file and print aggregate statistics
  python analyze_neuron_sparsity.py neuron_activations.csv
  
  # Save per-sample results to a file
  python analyze_neuron_sparsity.py neuron_activations.csv -o sparsity_results.csv
  
  # Use larger chunk size for faster processing (if you have more memory)
  python analyze_neuron_sparsity.py neuron_activations.csv --chunk-size 50000
        """
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the neuron_activations.csv file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Optional: Path to save per-sample zero neuron percentages'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Number of rows to process at once (default: 10000)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
    
    # Run the analysis
    try:
        stats = analyze_neuron_sparsity(
            args.csv_file,
            output_file=args.output,
            chunk_size=args.chunk_size
        )
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


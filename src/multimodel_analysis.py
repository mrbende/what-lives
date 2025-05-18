import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import asyncio

def multi_model_analysis(whatLives_instance):
    """
    Load correlation matrices from multiple models, average them, and run clustering analysis.
    
    Args:
        whatLives_instance: An initialized WhatLives object that has been set up
        
    Returns:
        tuple: (whatLives_instance, avg_matrix, cluster_assignments, color_map)
    """
    # Store original model to restore it later
    original_model = whatLives_instance.model
    
    # Create a new directory for the multi-model analysis
    multi_model_dir = os.path.join(whatLives_instance.data_dir, "results", "Multi-Model-Average")
    os.makedirs(multi_model_dir, exist_ok=True)
    
    # List of models whose correlation matrices we want to average
    models = [
        "GPT-4o",
        "Llama 3.3 70B Instruct", 
        "Anthropic Claude 3.7 Sonnet"
    ]
    
    print(f"Loading and averaging correlation matrices from: {', '.join(models)}")
    
    # Load the correlation matrices
    correlation_matrices = []
    
    for model in models:
        matrix_path = os.path.join(whatLives_instance.data_dir, "results", model, f"m_correlation_{model}.npy")
        if os.path.exists(matrix_path):
            matrix = np.load(matrix_path)
            correlation_matrices.append(matrix)
            print(f"Loaded matrix for {model} with shape {matrix.shape}")
        else:
            print(f"Warning: Could not find correlation matrix for {model} at {matrix_path}")
    
    if not correlation_matrices:
        raise ValueError("No correlation matrices found. Cannot proceed with analysis.")
    
    # Average the matrices
    avg_matrix = np.mean(correlation_matrices, axis=0)
    print(f"Created averaged matrix with shape {avg_matrix.shape}")
    
    # Create a copy of the WhatLives instance to preserve the original
    # wl_temp = deepcopy(whatLives_instance)
    wl_temp = whatLives_instance
    
    # Set the new output directory and model name
    # wl_temp.model = "Multi-Model-Average"
    wl_temp.output_dir = multi_model_dir
    wl_temp.make_out_dir()
    
    # Save the averaged correlation matrix
    matrix_path = os.path.join(multi_model_dir, f"m_correlation_Multi-Model-Average.npy")
    np.save(matrix_path, avg_matrix)
    print(f"Saved averaged correlation matrix to {matrix_path}")
    
    # Plot the correlation matrix
    wl_temp.plot_correlation_matrix(
        avg_matrix, 
        title=f'Definition Correlation - Multi-Model Average'
    )
    
    # Generate clustered correlation heatmap
    print("Generating clustered correlation heatmap...")
    fig, ax_heatmap, ax_dendrogram, cluster_assignments, reordered_idx, color_map = wl_temp.plot_clustered_correlation_heatmap(
        avg_matrix, 
        wl_temp.definitions, 
        filename='clustered_correlations.png'
    )
    plt.close(fig)  # Close the figure after saving
    
    # Analyze clusters
    print("Performing cluster analysis...")
    stats = wl_temp.analyze_clusters(cluster_assignments, wl_temp.definitions)
    print(f"Found {stats['n_clusters']} clusters with average size {stats['avg_cluster_size']:.1f}")
    
    # Run semantic analysis on the clusters
    print("Running semantic analysis on clusters...")
    analysis = asyncio.run(wl_temp.get_cluster_analysis(cluster_assignments, wl_temp.definitions))
    wl_temp.print_cluster_analysis(analysis, markdown_filename="multi_model_cluster_analysis.md")
    
    # Project data into semantic space using different methods
    print("Projecting into embedding space...")
    projections = wl_temp.project_and_visualize_embeddings(
        embedding_type="bedrock",
        cluster_assignments=cluster_assignments,
        color_map=color_map,
        force_recompute=True
    )
    
    # Project correlation matrix directly to 2D/3D space
    print("Projecting correlation matrix directly...")
    corr_projections = wl_temp.correlation_based_projections_with_testing(
        correlation_matrix=avg_matrix,
        cluster_assignments=cluster_assignments,
        color_map=color_map,
        force_recompute=True
    )
    
    # Generate comparisons between the multi-model average and individual models
    print("Generating model comparison plots...")
    generate_model_comparisons(whatLives_instance, models, multi_model_dir)
    
    # Restore original model setting
    whatLives_instance.model = original_model
    
    print(f"\nMulti-model analysis complete! Results saved to {multi_model_dir}")
    return wl_temp, avg_matrix, cluster_assignments, color_map


def generate_model_comparisons(whatLives_instance, models, multi_model_dir):
    """
    Generate comparison plots between the multi-model average and individual models.
    
    Args:
        whatLives_instance: WhatLives instance
        models: List of model names
        multi_model_dir: Output directory for multi-model analysis
    """
    # Load the multi-model average correlation matrix
    multi_model_matrix = np.load(os.path.join(multi_model_dir, "m_correlation_Multi-Model-Average.npy"))
    
    # Create a figure for comparing correlation matrices
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5), dpi=150)
    
    # Handle the case of just one model
    if n_models == 1:
        axes = [axes]
    
    # Plot the multi-model average first
    ax = axes[0]
    im = ax.imshow(multi_model_matrix, vmin=-1, vmax=1, cmap='RdYlGn')
    ax.set_title("Multi-Model Average", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Plot each individual model's matrix
    for i, model in enumerate(models):
        matrix_path = os.path.join(whatLives_instance.data_dir, "results", model, f"m_correlation_{model}.npy")
        if os.path.exists(matrix_path):
            matrix = np.load(matrix_path)
            ax = axes[i+1]
            im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdYlGn')
            ax.set_title(model, fontsize=12, fontweight='bold')
            ax.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('Correlation', fontsize=12, fontweight='bold')
    
    # Add overall title
    plt.suptitle("Comparison of Correlation Matrices", fontsize=16, fontweight='bold', y=0.95)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(multi_model_dir, "model_correlation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Calculate and save matrix differences
    print("Calculating matrix differences between models...")
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), dpi=150)
    if n_models == 1:
        axes = [axes]  # Handle the case of just one model
        
    for i, model in enumerate(models):
        matrix_path = os.path.join(whatLives_instance.data_dir, "results", model, f"m_correlation_{model}.npy")
        if os.path.exists(matrix_path):
            matrix = np.load(matrix_path)
            diff = multi_model_matrix - matrix
            
            # Plot difference
            ax = axes[i]
            im = ax.imshow(diff, vmin=-0.5, vmax=0.5, cmap='coolwarm')
            ax.set_title(f"Avg - {model}", fontsize=12, fontweight='bold')
            ax.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('Difference', fontsize=12, fontweight='bold')
    
    # Add overall title
    plt.suptitle("Differences from Multi-Model Average", fontsize=16, fontweight='bold', y=0.95)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(multi_model_dir, "model_correlation_differences.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Calculate statistics on model differences
    differences = {}
    agreement = {}
    
    for model in models:
        matrix_path = os.path.join(whatLives_instance.data_dir, "results", model, f"m_correlation_{model}.npy")
        if os.path.exists(matrix_path):
            matrix = np.load(matrix_path)
            diff = multi_model_matrix - matrix
            
            # Get difference statistics
            differences[model] = {
                'mean_abs_diff': np.mean(np.abs(diff)),
                'max_abs_diff': np.max(np.abs(diff)),
                'std_diff': np.std(diff)
            }
            
            # Calculate agreement between models (cells with similar correlation values)
            # We define agreement as cells where the difference is less than 0.2
            agreement_mask = np.abs(diff) < 0.2
            agreement[model] = np.mean(agreement_mask) * 100  # Percentage of cells in agreement
    
    # Save statistics to file
    with open(os.path.join(multi_model_dir, "model_agreement_statistics.txt"), 'w') as f:
        f.write("Model Agreement Statistics\n")
        f.write("=========================\n\n")
        
        f.write("Differences from Multi-Model Average:\n")
        for model, stats in differences.items():
            f.write(f"{model}:\n")
            f.write(f"  Mean absolute difference: {stats['mean_abs_diff']:.4f}\n")
            f.write(f"  Max absolute difference: {stats['max_abs_diff']:.4f}\n")
            f.write(f"  Standard deviation of difference: {stats['std_diff']:.4f}\n")
            f.write(f"  Agreement with average: {agreement[model]:.1f}%\n")
            f.write("\n")
        
        # Calculate pairwise agreement between models
        f.write("Pairwise Model Agreements:\n")
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Only compute each pair once
                    path1 = os.path.join(whatLives_instance.data_dir, "results", model1, f"m_correlation_{model1}.npy")
                    path2 = os.path.join(whatLives_instance.data_dir, "results", model2, f"m_correlation_{model2}.npy")
                    
                    if os.path.exists(path1) and os.path.exists(path2):
                        mat1 = np.load(path1)
                        mat2 = np.load(path2)
                        pair_diff = mat1 - mat2
                        
                        # Calculate agreement statistics
                        pair_agreement = np.mean(np.abs(pair_diff) < 0.2) * 100
                        pair_corr = np.corrcoef(mat1.flatten(), mat2.flatten())[0, 1]
                        
                        f.write(f"{model1} vs {model2}:\n")
                        f.write(f"  Agreement: {pair_agreement:.1f}%\n")
                        f.write(f"  Correlation: {pair_corr:.4f}\n")
                        f.write("\n")


# Example usage
def run_multi_model_analysis():
    """
    Main function to run the multi-model analysis after importing the WhatLives class
    """
    from inference import Inference
    from whatLives import WhatLives
    
    # Initialize the base WhatLives object with any model
    inference = Inference()
    wl = WhatLives(inference=inference, model="Anthropic Claude 3.7 Sonnet")
    
    # Run the multi-model analysis
    wl_multi, avg_matrix, cluster_assignments, color_map = multi_model_analysis(wl)
    
    return wl_multi, avg_matrix, cluster_assignments, color_map


# Execute the script if run directly
if __name__ == "__main__":
    run_multi_model_analysis()
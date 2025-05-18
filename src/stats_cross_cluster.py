import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# from whatLives import WhatLives

def analyze_model_clusters(data_dir="/workspace/what-lives/data", n_clusters=8):
    """
    Analyze clustering consistency across models using WhatLives's exact clustering method
    """
    # Define models
    models = [
        "Anthropic Claude 3.7 Sonnet",
        "Llama 3.3 70B Instruct", 
        "GPT-4o",
        "Multi-Model-Average"
    ]
    
    # Initialize WhatLives instance without specifying a model
    # (We'll manually set the output directory for each model)
    wl = WhatLives(model=None)
    
    # Dictionary to store cluster assignments for each model
    model_clusters = {}
    model_stats = {}
    
    # Dictionary to store intra and inter cluster correlation metrics
    intra_inter_stats = {}
    
    # Process each model
    for model in models:
        print(f"\nProcessing {model}...")
        
        # Set correct output directory for this model
        wl.output_dir = os.path.join(data_dir, "results", model)
        
        # Load correlation matrix
        matrix_path = os.path.join(wl.output_dir, f"m_correlation_{model if model != 'Multi-Model-Average' else 'Multi-Model-Average'}.npy")

        if os.path.exists(matrix_path):
            # Load correlation matrix
            correlation_matrix = np.load(matrix_path)
            
            # Calculate basic statistics
            flat_matrix = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            model_stats[model] = {
                "range": (np.min(flat_matrix), np.max(flat_matrix)),
                "mean": np.mean(flat_matrix),
                "median": np.median(flat_matrix),
                "std": np.std(flat_matrix)
            }
            print(f"  Matrix stats: Range [{model_stats[model]['range'][0]:.2f}, {model_stats[model]['range'][1]:.2f}], "
                  f"Mean: {model_stats[model]['mean']:.2f}, Median: {model_stats[model]['median']:.2f}")
            
            # Use the exact same clustering method from the WhatLives class
            # This will ensure consistency with your established methodology
            fig, ax_heatmap, ax_dendrogram, cluster_assignments, reordered_idx, color_map = wl.plot_clustered_correlation_heatmap(
                correlation_matrix, 
                wl.definitions,
                filename=None  # Don't save a new file
            )
            
            # Close the figure to save memory
            plt.close(fig)
            
            # Store cluster assignments
            model_clusters[model] = cluster_assignments
            
            # Count members per cluster
            cluster_counts = {}
            for name, cluster in cluster_assignments.items():
                if cluster not in cluster_counts:
                    cluster_counts[cluster] = 0
                cluster_counts[cluster] += 1
            
            print(f"  Cluster sizes: {cluster_counts}")
            
            # Calculate intra-cluster and inter-cluster correlations
            definition_names = [d['Name'] for d in wl.definitions]
            name_to_idx = {name: i for i, name in enumerate(definition_names)}
            
            # Group indices by cluster
            cluster_indices = {}
            for name, cluster in cluster_assignments.items():
                if cluster not in cluster_indices:
                    cluster_indices[cluster] = []
                cluster_indices[cluster].append(name_to_idx[name])
            
            # Calculate intra-cluster correlations
            intra_cluster_corrs = {}
            all_intra_corrs = []
            
            for cluster, indices in cluster_indices.items():
                if len(indices) > 1:  # Need at least 2 items for correlation
                    corrs = []
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            corrs.append(correlation_matrix[indices[i], indices[j]])
                    
                    mean_corr = np.mean(corrs)
                    std_corr = np.std(corrs)
                    all_intra_corrs.extend(corrs)
                    
                    intra_cluster_corrs[cluster] = {
                        'mean': mean_corr,
                        'std': std_corr,
                        'min': np.min(corrs),
                        'max': np.max(corrs),
                        'size': len(indices)
                    }
                else:
                    # For singleton clusters, we define self-correlation as 1.0
                    intra_cluster_corrs[cluster] = {
                        'mean': 1.0,
                        'std': 0.0,
                        'min': 1.0,
                        'max': 1.0,
                        'size': 1
                    }
            
            # Calculate inter-cluster correlations
            inter_cluster_corrs = {}
            all_inter_corrs = []
            
            unique_clusters = sorted(cluster_indices.keys())
            for i, cluster1 in enumerate(unique_clusters):
                for cluster2 in unique_clusters[i+1:]:
                    corrs = []
                    for idx1 in cluster_indices[cluster1]:
                        for idx2 in cluster_indices[cluster2]:
                            corrs.append(correlation_matrix[idx1, idx2])
                    
                    if corrs:
                        mean_corr = np.mean(corrs)
                        std_corr = np.std(corrs)
                        all_inter_corrs.extend(corrs)
                        
                        pair_key = f"{cluster1}-{cluster2}"
                        inter_cluster_corrs[pair_key] = {
                            'mean': mean_corr,
                            'std': std_corr,
                            'min': np.min(corrs),
                            'max': np.max(corrs),
                            'size1': len(cluster_indices[cluster1]),
                            'size2': len(cluster_indices[cluster2])
                        }
            
            # Calculate overall means
            intra_mean = np.mean(all_intra_corrs) if all_intra_corrs else 1.0
            intra_std = np.std(all_intra_corrs) if all_intra_corrs else 0.0
            inter_mean = np.mean(all_inter_corrs) if all_inter_corrs else 0.0
            inter_std = np.std(all_inter_corrs) if all_inter_corrs else 0.0
            contrast = intra_mean - inter_mean
            
            intra_inter_stats[model] = {
                'intra_cluster': {
                    'overall_mean': intra_mean,
                    'overall_std': intra_std,
                    'per_cluster': intra_cluster_corrs
                },
                'inter_cluster': {
                    'overall_mean': inter_mean,
                    'overall_std': inter_std,
                    'per_pair': inter_cluster_corrs
                },
                'contrast': contrast
            }
            
            print(f"  Intra-cluster correlation: {intra_mean:.3f} ± {intra_std:.3f}")
            print(f"  Inter-cluster correlation: {inter_mean:.3f} ± {inter_std:.3f}")
            print(f"  Contrast (intra-inter): {contrast:.3f}")
            
            # Create a visualization of the cluster correlation matrix
            plt.figure(figsize=(10, 8))
            n_clusters = len(unique_clusters)
            
            # Create matrix to hold correlations
            corr_matrix = np.zeros((n_clusters, n_clusters))
            
            # Fill diagonal with intra-cluster correlations
            for i, cluster in enumerate(unique_clusters):
                corr_matrix[i, i] = intra_cluster_corrs[cluster]['mean']
            
            # Fill off-diagonal with inter-cluster correlations
            for pair, stats in inter_cluster_corrs.items():
                c1, c2 = map(int, pair.split('-'))
                idx1 = unique_clusters.index(c1)
                idx2 = unique_clusters.index(c2)
                corr_matrix[idx1, idx2] = stats['mean']
                corr_matrix[idx2, idx1] = stats['mean']  # Make symmetric
            
            # Plot heatmap
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1,
                       xticklabels=unique_clusters,
                       yticklabels=unique_clusters)
            plt.title(f'{model} - Cluster Correlation Matrix')
            plt.tight_layout()
            
            # Save figure
            heatmap_path = os.path.join(wl.output_dir, "cluster_correlation_matrix.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Cluster correlation matrix saved to {heatmap_path}")
            
        else:
            print(f"  ERROR: Matrix file not found at {matrix_path}")
    
    # Check if we have all models processed
    if len(model_clusters) != len(models):
        print("WARNING: Not all models could be processed")
    
    # Now compare the clusterings between models
    # We'll build a matrix showing how often pairs of definitions are grouped together
    definition_names = [d['Name'] for d in wl.definitions]
    name_pairs = [(name1, name2) for i, name1 in enumerate(definition_names) 
                 for name2 in definition_names[i+1:]]
    
    # Calculate pairwise consistency between models
    consistency_matrix = np.zeros((len(models), len(models)))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i <= j and model1 in model_clusters and model2 in model_clusters:  # Include diagonal and upper triangle
                # Count pairs of definitions that are clustered the same way in both models
                same_cluster_count = 0
                total_pairs = 0
                
                # For each pair of definitions
                for name1, name2 in name_pairs:
                    # Check if they're in the same cluster in both models
                    same_in_model1 = model_clusters[model1][name1] == model_clusters[model1][name2]
                    same_in_model2 = model_clusters[model2][name1] == model_clusters[model2][name2]
                    
                    # If the grouping is the same in both models (either both together or both apart)
                    if same_in_model1 == same_in_model2:
                        same_cluster_count += 1
                    
                    total_pairs += 1
                
                # Calculate consistency percentage
                consistency = (same_cluster_count / total_pairs) * 100
                consistency_matrix[i, j] = consistency
                consistency_matrix[j, i] = consistency  # Make symmetric
                
                if i != j:
                    print(f"Consistency between {model1} and {model2}: {consistency:.1f}%")
    
    # Calculate definition stability across all models
    stable_definitions = []
    for name in definition_names:
        # For each other definition
        stability_score = 0
        for other_name in definition_names:
            if name != other_name:
                # Check if this definition is grouped with other_name consistently across models
                grouping_patterns = []
                for model in models:
                    if model in model_clusters:
                        same_cluster = model_clusters[model][name] == model_clusters[model][other_name]
                        grouping_patterns.append(same_cluster)
                
                # If all models agree on whether these two should be together
                if all(grouping_patterns) or not any(grouping_patterns):
                    stability_score += 1
        
        # Convert to percentage of stable relationships
        stability_pct = (stability_score / (len(definition_names) - 1)) * 100
        stable_definitions.append((name, stability_pct))
    
    # Sort by stability
    stable_definitions.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate overall stability percentage
    overall_stability = np.mean([s[1] for s in stable_definitions])
    print(f"\nOverall definition grouping stability: {overall_stability:.1f}%")
    
    # Print most and least stable definitions
    print("\nMost consistently grouped definitions:")
    for name, stability in stable_definitions[:5]:
        print(f"  {name}: {stability:.1f}%")
    
    print("\nLeast consistently grouped definitions:")
    for name, stability in stable_definitions[-5:]:
        print(f"  {name}: {stability:.1f}%")
    
    # Calculate inter-model correlations for the correlation matrices themselves
    print("\nCorrelation matrix similarities:")
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j and model1 in model_clusters and model2 in model_clusters:
                mat1 = np.load(os.path.join(data_dir, "results", model1, f"m_correlation_{model1}.npy"))
                mat2 = np.load(os.path.join(data_dir, "results", model2, f"m_correlation_{model2}.npy"))
                
                # Flatten upper triangles
                flat1 = mat1[np.triu_indices_from(mat1, k=1)]
                flat2 = mat2[np.triu_indices_from(mat2, k=1)]
                
                # Calculate correlation
                corr = np.corrcoef(flat1, flat2)[0, 1]
                print(f"  {model1} vs {model2}: r = {corr:.4f}")
    
    # Plot consistency matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(consistency_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=[m.split()[-1] for m in models],  # Just use last part of model name
                yticklabels=[m.split()[-1] for m in models])
    plt.title("Cluster Assignment Consistency Between Models (%)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(data_dir, "results", "Multi-Model-Average", "model_cluster_consistency.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConsistency matrix saved to {output_path}")
    
    # Count how many definitions maintain the exact same cluster groupings across all models
    consistent_groups = []
    for i, name1 in enumerate(definition_names):
        for j in range(i+1, len(definition_names)):
            name2 = definition_names[j]
            
            # Check if this pair has consistent grouping across all models
            consistent = True
            first_model = models[0]
            first_grouping = model_clusters[first_model][name1] == model_clusters[first_model][name2]
            
            for model in models[1:]:
                if model in model_clusters:
                    grouping = model_clusters[model][name1] == model_clusters[model][name2]
                    if grouping != first_grouping:
                        consistent = False
                        break
            
            if consistent:
                consistent_groups.append((name1, name2))
    
    consistency_pct = (len(consistent_groups) / len(name_pairs)) * 100
    print(f"\nPercentage of definition pairs with consistent grouping across all models: {consistency_pct:.1f}%")
    
    # Create a comparison bar chart for intra/inter metrics across models
    if len(intra_inter_stats) > 1:
        plt.figure(figsize=(12, 7))
        x = np.arange(len(intra_inter_stats))
        width = 0.25
        
        intra_means = [intra_inter_stats[model]['intra_cluster']['overall_mean'] for model in models if model in intra_inter_stats]
        inter_means = [intra_inter_stats[model]['inter_cluster']['overall_mean'] for model in models if model in intra_inter_stats]
        contrasts = [intra_inter_stats[model]['contrast'] for model in models if model in intra_inter_stats]
        
        model_labels = [m.split()[-1] for m in models if m in intra_inter_stats]
        
        plt.bar(x - width, intra_means, width, label='Intra-Cluster Mean', color='green')
        plt.bar(x, inter_means, width, label='Inter-Cluster Mean', color='blue')
        plt.bar(x + width, contrasts, width, label='Contrast (Intra-Inter)', color='red')
        
        plt.ylabel('Correlation')
        plt.title('Cluster Correlation Comparison Across Models')
        plt.xticks(x, model_labels)
        plt.legend()
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        comparison_path = os.path.join(data_dir, "results", "Multi-Model-Average", "cluster_correlation_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nCluster correlation comparison saved to {comparison_path}")
    
    return {
        "model_stats": model_stats,
        "model_clusters": model_clusters,
        "consistency_matrix": consistency_matrix,
        "stable_definitions": stable_definitions,
        "overall_stability": overall_stability,
        "consistent_pairs_percentage": consistency_pct,
        "intra_inter_stats": intra_inter_stats
    }

# Run the analysis
results = analyze_model_clusters()
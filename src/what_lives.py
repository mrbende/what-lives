import json, re, os
import sys
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import leidenalg as la
import igraph as ig
from sklearn.neighbors import kneighbors_graph
import asyncio
import nest_asyncio
from tqdm.notebook import tqdm
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
import seaborn as sns
import textwrap
from collections import defaultdict
import datetime
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
import umap
from matplotlib.lines import Line2D  
from scipy.stats import gaussian_kde
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties

# nest_asyncio.apply()
sys.path.append("/workspace/what-lives/src")

from inference import Inference

class WhatLives:
    def __init__(self, inference=None, model=None, n_max=None, semaphore_limit=33, n_replicates=3):
        ### LOAD INFERENCE CLASS FOR LANGUAGE QUERY
        if not inference:
            self.Inference = Inference()
        else:
            self.Inference = inference

        ### CONCURRENCY LIMIT
        self.semaphore_limit = semaphore_limit
        
        ### CORRELATION MATRIX HYPERPARAMETERS
        self.n_replicates = n_replicates
        
        ### SOURCE XLSX INGRESS
        self.n_max = n_max   # allow for subset of definitions to be analyzed, or `None`
        self.data_dir = "/workspace/what-lives/data"
        self.embeddings_dir = os.path.join(self.data_dir, "definition-embeddings")
        self.definitions_table_path = os.path.join(self.data_dir, "what_lives_definitions.xlsx")
        self.definitions_all = self.xlsx_to_json(self.definitions_table_path)
        
        ### FILTER DEFINITIONS TO REMOVE SUPPLEMENTAL
        self.definitions = [item for item in self.definitions_all if item.get('Supplemental') == False]
        
        ### PARSE LAST NAMES IN IMPORTED TABLE
        self.add_last_names()
        
        ### SET MODEL TO USE FOR CORRELATION AND SEMANTIC ANALYSIS
        if not model:
            self.model = self.Inference.model_config["default_models"]["global_default"]
        else:
            if model not in self.Inference.all_models:
                raise ValueError(f"Provided model `{model}` is not supported. Please provide any of : {self.Inference.all_models}")
            else:
                self.model = model
        print(f"Initialized model for analysis: {self.model}")
        
        ### SETUP PATHS -- AFTER MODEL HAS BEEN INSTANTIATED
        self.output_dir = os.path.join(self.data_dir, "results", self.model)
        self.make_out_dir()
        
        ### Initialize empty dictionary for cluster titles
        self.cluster_titles = {}
        
    #############    
    ### SETUP ###
    #############
    def make_out_dir(self):
        try:
            os.mkdir(self.output_dir)
        except FileExistsError:
            pass
    
    ############################
    ### READ LIFE DEINITIONS ###
    ############################
    def xlsx_to_json(self,
                     table_name,
                     sheet_name='Sheet1',
                     output_path=None,
                     orient='records'):
        try:
            df = pd.read_excel(table_name, sheet_name=sheet_name)
            df = df.where(pd.notnull(df), None)
            json_data = json.loads(df.to_json(orient=orient, date_format='iso'))
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            if not self.n_max:
                return json_data
            else:
                return json_data[:self.n_max]
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        except ValueError as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")
            
    def add_last_names(self):
        for i in range(len(self.definitions)):
            names = self.definitions[i]["Name"].split()
            self.definitions[i]["Last"] = names[-1]
             
    ############################    
    ### CORRELATION MATRICES ###
    ############################
    ### AVERAGE OF MULTIPLE API CALLS
    async def definition_correlation(self, def1, def2):
        ### FORMAT SYSTEM PROMPT
        prompt = self.Inference._read_prompt_template("definition_correlation")
        prompt = prompt.format(
            def1=def1,
            def2=def2,
        )
        question = "What is the correlation metric between -1.0 and 1.0 for these two definitions? Respond with ONLY a single number!"

        ### CREATE THREE ASYNC COROUTINES
        async def get_single_correlation():
            score, metadata = await self.Inference.acomplete(
                text=question, 
                system_prompt=prompt, 
                numerical=True,
                model=self.model
            )
            return score, metadata["cost"]

        ### EXECUTE THREE CORRELATION ANALYSES CONCURRENTLY
        tasks = [get_single_correlation() for _ in range(self.n_replicates)]
        results = await asyncio.gather(*tasks)
        scores, costs = zip(*results)

        ### RETURN AVERAGE CORRELATION, STD, AND TOTAL COST
        average_score = sum(scores) / len(scores)
        std_score = np.std(scores)
        total_cost = sum(costs)
        return average_score, std_score, total_cost

    async def async_define_correlation_matrix(self, checkpoint_freq=100, resume_from_checkpoint=True):
        ### CREATE PROGRESS FILES FOR CHECKPOINT SAVING
        checkpoint_path = os.path.join(self.output_dir, f"correlation_checkpoint_{self.model}.npz")
        progress_path = os.path.join(self.output_dir, f"correlation_progress_{self.model}.json")

        # Get total number of definitions
        n = len(self.definitions)

        # Initialize matrices and progress tracking
        M = np.zeros((n, n))  # For average correlations
        S = np.zeros((n, n))  # For standard deviations
        costs = []

        # Track which pairs have been computed
        computed_pairs = set()

        # Check if checkpoint exists and load if requested
        if os.path.exists(checkpoint_path) and resume_from_checkpoint:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = np.load(checkpoint_path)
            M = checkpoint['correlation_matrix']
            S = checkpoint['std_matrix']

            if os.path.exists(progress_path):
                with open(progress_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    costs = checkpoint_data.get('costs', [])
                    computed_pairs = set(tuple(pair) for pair in checkpoint_data.get('computed_pairs', []))

            print(f"Resumed with {len(computed_pairs)} pairs already computed")

        # Create list of all pairs to compute
        all_pairs = [(i, j) for i in range(n) for j in range(n)]

        # Filter out already computed pairs
        remaining_pairs = [(i, j) for i, j in all_pairs if (i, j) not in computed_pairs]

        if len(remaining_pairs) == 0:
            print("All correlations already computed. Returning checkpoint data.")
            return M, S

        print(f"Computing {len(remaining_pairs)} remaining correlation pairs...")

        # Set up semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.semaphore_limit)

        async def process_pair(i, j):
            async with semaphore:
                avg, std, cost = await self.definition_correlation(
                    def1=self.definitions[i]['Definition'],
                    def2=self.definitions[j]['Definition']
                )
                return i, j, avg, std, cost

        # Process remaining pairs with checkpointing
        tasks = [process_pair(i, j) for i, j in remaining_pairs]

        # Track how many pairs have been processed since last checkpoint
        pairs_since_checkpoint = 0
        checkpoint_counter = 0

        # Process all tasks with progress tracking
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            i, j, avg, std, cost = await task

            # Update matrices
            M[i, j] = avg
            S[i, j] = std
            costs.append(cost)

            # Track this pair as computed
            computed_pairs.add((i, j))

            # Increment counter for checkpoint
            pairs_since_checkpoint += 1

            # Save checkpoint if needed
            if pairs_since_checkpoint >= checkpoint_freq:
                checkpoint_counter += 1
                # print(f"\nSaving checkpoint {checkpoint_counter}...")

                # Save matrix data
                np.savez(
                    checkpoint_path,
                    correlation_matrix=M,
                    std_matrix=S
                )

                # Save progress metadata
                with open(progress_path, 'w') as f:
                    json.dump({
                        'costs': costs,
                        'computed_pairs': list(map(list, computed_pairs)),
                        'timestamp': str(datetime.datetime.now())
                    }, f)

                pairs_since_checkpoint = 0
                # print(f"Checkpoint saved. Completed {len(computed_pairs)}/{len(all_pairs)} pairs.")

        # Final checkpoint after all processing
        print("Saving final results...")
        np.savez(
            checkpoint_path,
            correlation_matrix=M,
            std_matrix=S
        )

        # Save final progress metadata
        with open(progress_path, 'w') as f:
            json.dump({
                'costs': costs,
                'computed_pairs': list(map(list, computed_pairs)),
                'timestamp': str(datetime.datetime.now()),
                'completed': True
            }, f)

        # Symmetrize, plot, and return
        self.plot_correlation_matrix(M, title=f'Definition Correlations - {self.model} - Raw')
        self.plot_correlation_matrix(S, title=f'Definition Correlations Standard Deviation - {self.model} - Raw', 
                                   is_std=True)

        Ms = (M.transpose() + M) * 0.5
        Ss = (S.transpose() + S) * 0.5

        self.plot_correlation_matrix(Ms, title=f'Definition Correlation - {self.model}')
        self.plot_correlation_matrix(Ss, title=f'Definition Correlations Standard Deviations - {self.model}', 
                                   is_std=True)

        total_cost = sum(costs)
        print("--- COMPLETE ---")
        print(f"total cost: ${total_cost:.2f}")

        # Save the final matrices to their standard locations
        self.save_correlation_matrix(Ms, matrix_name=f'm_correlation_{self.model}')

        return Ms, Ss


    def create_correlation_matrix(self, checkpoint_freq=100, resume_from_checkpoint=True):
        M, S = asyncio.run(self.async_define_correlation_matrix(
            checkpoint_freq=checkpoint_freq,
            resume_from_checkpoint=resume_from_checkpoint
        ))
        
        ### SAVE CORRELATION MATRIX TO FILE
        self.save_correlation_matrix(M, matrix_name=f'm_correlation_{self.model}')
        
        return M, S
    
    ### CHECKPOINT MANAGEMENT ###
    def get_correlation_status(self):
        # Define paths for checkpoint files
        checkpoint_path = os.path.join(self.output_dir, f"correlation_checkpoint_{self.model}.npz")
        progress_path = os.path.join(self.output_dir, f"correlation_progress_{self.model}.json")

        status = {
            "checkpoint_exists": False,
            "progress_exists": False,
            "total_pairs": len(self.definitions) ** 2,
            "computed_pairs": 0,
            "completion_percentage": 0,
            "total_cost": 0,
            "avg_cost_per_pair": 0,
            "estimated_remaining_cost": 0,
            "last_updated": None
        }

        # Check if checkpoint exists
        if os.path.exists(checkpoint_path):
            status["checkpoint_exists"] = True

            # Try to load the checkpoint to verify it's valid
            try:
                checkpoint = np.load(checkpoint_path)
                status["matrix_shape"] = checkpoint['correlation_matrix'].shape
            except Exception as e:
                status["checkpoint_error"] = str(e)

        # Check if progress file exists
        if os.path.exists(progress_path):
            status["progress_exists"] = True

            try:
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)

                status["computed_pairs"] = len(progress_data.get('computed_pairs', []))
                status["completion_percentage"] = (status["computed_pairs"] / status["total_pairs"]) * 100
                status["total_cost"] = sum(progress_data.get('costs', []))
                status["last_updated"] = progress_data.get('timestamp', None)

                if status["computed_pairs"] > 0:
                    status["avg_cost_per_pair"] = status["total_cost"] / status["computed_pairs"]
                    remaining_pairs = status["total_pairs"] - status["computed_pairs"]
                    status["estimated_remaining_cost"] = status["avg_cost_per_pair"] * remaining_pairs

                status["completed"] = progress_data.get('completed', False)

            except Exception as e:
                status["progress_error"] = str(e)

        return status

    def print_correlation_status(self):
        status = self.get_correlation_status()

        print(f"\n=== Correlation Matrix Status for {self.model} ===")

        if not status["checkpoint_exists"] and not status["progress_exists"]:
            print("No checkpoints found. Correlation matrix has not been started.")
            print(f"Total pairs to compute: {status['total_pairs']}")
            return

        print(f"Completion: {status['completion_percentage']:.2f}% ({status['computed_pairs']}/{status['total_pairs']} pairs)")
        print(f"Cost so far: ${status['total_cost']:.2f}")

        if status["computed_pairs"] > 0:
            print(f"Average cost per pair: ${status['avg_cost_per_pair']:.4f}")
            print(f"Estimated remaining cost: ${status['estimated_remaining_cost']:.2f}")

        if status.get("completed", False):
            print("Status: COMPLETED")
        else:
            print("Status: IN PROGRESS")

        if status["last_updated"]:
            print(f"Last updated: {status['last_updated']}")

        if "checkpoint_error" in status or "progress_error" in status:
            print("\nWarnings:")
            if "checkpoint_error" in status:
                print(f"- Checkpoint file error: {status['checkpoint_error']}")
            if "progress_error" in status:
                print(f"- Progress file error: {status['progress_error']}")


    def reset_correlation_calculation(self, confirmation=False):
        # Define paths for checkpoint files
        checkpoint_path = os.path.join(self.output_dir, f"correlation_checkpoint_{self.model}.npz")
        progress_path = os.path.join(self.output_dir, f"correlation_progress_{self.model}.json")

        # Check if files exist
        files_exist = os.path.exists(checkpoint_path) or os.path.exists(progress_path)

        if not files_exist:
            print("No checkpoint files found. Nothing to reset.")
            return False

        # Get confirmation if needed
        if not confirmation:
            status = self.get_correlation_status()
            print(f"\n=== Reset Correlation Matrix for {self.model} ===")
            print(f"Completion: {status['completion_percentage']:.2f}% ({status['computed_pairs']}/{status['total_pairs']} pairs)")
            print(f"Cost invested so far: ${status['total_cost']:.2f}")

            confirm = input("\nThis will delete all progress. Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("Reset cancelled.")
                return False

        # Delete files
        files_deleted = []

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                files_deleted.append(checkpoint_path)
            except Exception as e:
                print(f"Error deleting checkpoint file: {e}")

        if os.path.exists(progress_path):
            try:
                os.remove(progress_path)
                files_deleted.append(progress_path)
            except Exception as e:
                print(f"Error deleting progress file: {e}")

        if files_deleted:
            print(f"Reset successful. Deleted: {', '.join(files_deleted)}")
            return True
        else:
            print("Reset failed. No files were deleted.")
            return False

    def plot_correlation_matrix(self, M, title='Definition Correlations', is_std=False):
        plt.figure(figsize=(11,9))

        # Adjust colormap and range based on what we're plotting
        if is_std:
            vmin, vmax = 0, 0.5  # Adjust max range for std dev as needed
            cmap = 'YlOrRd'  # Different colormap for std dev
            cbar_label = 'Standard Deviation'
        else:
            vmin, vmax = -1, 1
            cmap = 'RdYlGn'
            cbar_label = 'Correlation'

        # Create the heatmap
        im = plt.pcolor(M, vmin=vmin, vmax=vmax, cmap=cmap)

        # Create colorbar with appropriate labels
        cbar = plt.colorbar(im, label=cbar_label, fraction=0.046, pad=0.04)

        if not is_std:
            cbar.ax.text(3.5, 1.0, 'agree', ha='left', va='bottom', fontweight='bold')
            cbar.ax.text(3.5, -1.0, 'disagree', ha='left', va='top', fontweight='bold')
            # cbar.ax.text(3.5, 0.0, 'no relation', ha='left', va='bottom', fontweight='bold')

        ### MAP TO NAMES
        names = [d['Name'] for d in self.definitions]

        # Set tick positions and labels
        tick_positions = np.arange(M.shape[0]) + 0.5
        plt.gca().set_xticks(tick_positions)
        plt.gca().set_yticks(tick_positions)
        plt.gca().set_xticklabels(names, rotation=45, ha='right', fontsize=6)
        plt.gca().set_yticklabels(names, fontsize=6)

        # Set the title
        plt.title(title, fontsize=14, pad=20, fontweight='bold')
        # plt.suptitle(self.model, fontsize=12, y=0.8)

        # Save and display
        plt.tight_layout()
        label_name = re.sub(r'[A-Z\s]+', lambda m: m.group().lower().replace(' ', '_'), title)
        plt.savefig(f'{os.path.join(self.output_dir,label_name)}.png', 
                    dpi=600, bbox_inches='tight')
        plt.show()    
    
#     def create_correlation_matrix(self):
#         M, S = asyncio.run(self.async_define_correlation_matrix())
#         # self.plot_correlation_matrix(M)
        
#         ### SAVE CORRELATION MATRIX TO FILE
#         self.save_correlation_matrix(M, matrix_name=f'm_correlation_{self.model}')
        
#         return M, S  
    
    def save_correlation_matrix(self, matrix, matrix_name='correlation_matrix'):
        file_path = os.path.join(self.output_dir, f"{matrix_name}.npy")
        np.save(file_path, matrix)
        print(f"Correlation matrix saved to {file_path}")
        return file_path
    
    #########################################
    ### AGGLOMERATIVE CLUSTERING ANALYSIS ###
    #########################################
    def find_optimal_clusters(self, linkage_matrix, distance_matrix, min_clusters=2, max_clusters=15):
        # Get the distances at each merge in the linkage matrix
        distances = linkage_matrix[:, 2]

        # Calculate acceleration (second derivative) of distances
        acceleration = np.diff(distances, n=2)

        # Find elbow point (maximum acceleration)
        elbow_idx = np.argmax(acceleration) + 2
        elbow_n_clusters = len(distances) - elbow_idx + 2

        # Constrain elbow suggestion to our bounds
        elbow_n_clusters = max(min_clusters, min(elbow_n_clusters, max_clusters))
        print(f"Elbow Clusters: {elbow_n_clusters}")
        return elbow_n_clusters

    ### PRIMARY PLOT TO SHOW SORTED CORRELATION MATRIX WITH SUPERIMPOSED LINKAGE DENDROGRAM
    def plot_clustered_correlation_heatmap(self, correlation_matrix, definitions, filename=None, figsize=(28, 20)):
        # Extract names for labels
        names = [d['Name'] for d in definitions]

        # # Convert correlation matrix to distance matrix
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))  # Using correlation-based distance
        distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure perfect symmetry
        ### USE SIGMOID TRANSFORMATION ON RESULTING DISTANCE MATRIX AROUND 75TH PERCENTILE AS CENTROID...
        # distance_matrix = self.transform_correlation_to_distance(
        #     correlation_matrix, 
        #     method="sigmoid",
        #     percentile=50  # lower percentile to not have a bunch of isolate vanguard clusters...
        # )

        # Get condensed form for linkage
        condensed_dist = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering using complete linkage
        linkage_matrix = hierarchy.linkage(
            condensed_dist,
            method='complete',  # Complete linkage for clearer cluster separation
            optimal_ordering=True
        )

        # Determine optimal number of clusters
        n_clusters = self.find_optimal_clusters(linkage_matrix, distance_matrix)
        clusters = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Create color palette once for all cluster-related coloring
        unique_clusters = sorted(np.unique(clusters))
        cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))
        color_map = dict(zip(unique_clusters, cluster_colors))

        # Create figure with custom layout
        fig = plt.figure(figsize=figsize)

        # Create a gridspec with proper spacing
        gs = plt.GridSpec(1, 2, width_ratios=[1, 0.3], wspace=0.02, height_ratios=[1])

        # First, compute dendrogram to get the leaf ordering
        dendrogram_info = hierarchy.dendrogram(
            linkage_matrix,
            no_plot=True,  # Just get the ordering information
        )

        # Get the order of leaves
        reordered_idx = dendrogram_info['leaves']

        # Reorder the correlation matrix and names
        reordered_corr = correlation_matrix[reordered_idx][:, reordered_idx]
        reordered_names = [names[i] for i in reordered_idx]
        reordered_clusters = clusters[reordered_idx]

        # Create heatmap (left side)
        ax_heatmap = fig.add_subplot(gs[0, 0])

        # Plot the heatmap with the original ordering first (will reorient later)
        im = ax_heatmap.imshow(
            reordered_corr,
            aspect='auto',  # Force square cells
            cmap='RdYlGn',
            vmin=-1,
            vmax=1
        )

        # Set up axes and labels for heatmap
        ax_heatmap.set_xticks(np.arange(len(reordered_names)))
        ax_heatmap.set_yticks(np.arange(len(reordered_names)))
        ax_heatmap.set_xticklabels(reordered_names, rotation=45, ha='right', fontsize=12)
        ax_heatmap.set_yticklabels(reordered_names, fontsize=12)

        # Flip the y-axis to match dendrogram orientation
        ax_heatmap.invert_yaxis()

        # Define cluster boundaries in reordered coordinates
        cluster_boundaries = {}
        current_cluster = None
        start_idx = 0

        # Find boundaries for clusters in the x-direction (original order)
        for idx, cluster_id in enumerate(reordered_clusters):
            if current_cluster is None:
                current_cluster = cluster_id
                start_idx = idx
            elif cluster_id != current_cluster:
                cluster_boundaries[current_cluster] = (start_idx, idx - 1)
                current_cluster = cluster_id
                start_idx = idx

        # Add the last cluster
        if current_cluster is not None:
            cluster_boundaries[current_cluster] = (start_idx, len(reordered_clusters) - 1)

        # Color code tick labels and add cluster boundaries
        for cluster_id, (start, end) in cluster_boundaries.items():
            color = color_map[cluster_id]

            # Color the x-axis tick labels
            for i in range(start, end + 1):
                ax_heatmap.get_xticklabels()[i].set_color(mcolors.rgb2hex(color))
                ax_heatmap.get_xticklabels()[i].set_fontweight('bold')

            # Color the y-axis tick labels - matching the flipped orientation
            for i in range(start, end + 1):
                ax_heatmap.get_yticklabels()[i].set_color(mcolors.rgb2hex(color))
                ax_heatmap.get_yticklabels()[i].set_fontweight('bold')

            # Add vertical divider lines
            if start > 0:
                ax_heatmap.axvline(x=start-0.5, color='black', linewidth=1.0)

            # Add horizontal divider lines - matching flipped orientation
            if start > 0:
                ax_heatmap.axhline(y=start-0.5, color='black', linewidth=1.0)

            # Draw rectangle around each cluster in the heatmap
            rect = plt.Rectangle(
                (start - 0.5, start - 0.5),  # Position (x_start, y_start) in flipped coordinates
                end - start + 1,              # Width
                end - start + 1,              # Height
                fill=False,
                edgecolor=mcolors.rgb2hex(color),
                linewidth=6,
                zorder=10
            )
            ax_heatmap.add_patch(rect)

        # Add gridlines to heatmap
        ax_heatmap.set_xticks(np.arange(-.5, len(reordered_names), 1), minor=True)
        ax_heatmap.set_yticks(np.arange(-.5, len(reordered_names), 1), minor=True)
        ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

        # Create link colors for dendrogram
        link_cols = {}
        for i, merge in enumerate(linkage_matrix):
            left = int(merge[0])
            right = int(merge[1])

            # Get clusters for merged elements
            if left < len(names):
                left_cluster = clusters[left]
            else:
                left_cluster = link_cols[left - len(names)]['cluster']

            if right < len(names):
                right_cluster = clusters[right]
            else:
                right_cluster = link_cols[right - len(names)]['cluster']

            # If merging within same cluster, use cluster color
            if left_cluster == right_cluster:
                link_cols[i] = {
                    'color': mcolors.rgb2hex(color_map[left_cluster]),
                    'cluster': left_cluster
                }
            else:
                # For between-cluster merges, use dark grey
                link_cols[i] = {
                    'color': '#555555',  # Darker gray for better visibility
                    'cluster': min(left_cluster, right_cluster)
                }

        # Create dendrogram on the right side
        ax_dendrogram = fig.add_subplot(gs[0, 1])

        # Function to determine link colors
        def get_link_color(k):
            if k < len(link_cols):
                return link_cols[k]['color']
            return 'black'  # Black for better visibility

        # Create dendrogram with orientation='right'
        R = hierarchy.dendrogram(
            linkage_matrix,
            orientation='right',       # Orient to right side
            labels=None,               # No labels
            no_labels=True,            # Explicitly no labels
            ax=ax_dendrogram,
            link_color_func=get_link_color,
            above_threshold_color='black'
        )

        # Clean up dendrogram axes
        ax_dendrogram.spines['top'].set_visible(False)
        ax_dendrogram.spines['right'].set_visible(False)
        ax_dendrogram.spines['bottom'].set_visible(False)
        ax_dendrogram.spines['left'].set_visible(False)
        ax_dendrogram.tick_params(bottom=False, left=False)
        ax_dendrogram.set_xticks([])
        ax_dendrogram.set_yticks([])

        # Add title
        plt.suptitle(
            'Clustered Definition Correlations',
            fontsize=20, 
            fontweight='bold',
            y=0.98
        )

        # Final layout adjustment - using subplots_adjust instead of tight_layout
        plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1)

        # Save figure if filename provided
        if filename:
            fout = os.path.join(self.output_dir, filename)
            plt.savefig(fout, dpi=400, bbox_inches='tight')

        # Force complete rendering of the figure
        fig.canvas.draw()

        # Return everything needed for further analysis
        cluster_assignments = {name: cluster for name, cluster in zip(names, clusters)}
        return fig, ax_heatmap, ax_dendrogram, cluster_assignments, reordered_idx, color_map
    
    ##########################################################################
    ### PLOT WITH FEATURE VECTORS DERIVED DIRECTLY FROM CORRELATION MATRIX ###
    ##########################################################################
    def transform_correlation_to_distance(self, correlation_matrix, method="standard", **kwargs):
        # Ensure the matrix is symmetric and values are in expected range
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

        # Standard transformation (baseline)
        if method == "standard":
            distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

        # Power transformation (power < 1 stretches differences in high correlations)
        elif method == "power":
            power = kwargs.get("power", 0.3)  # Lower values give more stretching
            distance_matrix = (2 * (1 - correlation_matrix)) ** power

        # Logarithmic transformation (amplifies small differences)
        elif method == "log":
            epsilon = kwargs.get("epsilon", 1e-10)  # Small value to avoid log(0)
            # Using -log(similarity) to amplify small differences in high correlations
            similarity = (correlation_matrix + 1) / 2  # Map to [0, 1] range
            distance_matrix = -np.log(similarity + epsilon)
            # Normalize to have max distance = 2 (similar to standard method)
            distance_matrix = distance_matrix * (2 / distance_matrix.max())

        # Sigmoid transformation (creates more contrast around a threshold)
        elif method == "sigmoid":
            center = np.percentile(correlation_matrix, kwargs.get("percentile", 75))  # Center point of the sigmoid (in correlation space)
            steepness = kwargs.get("steepness", 15)  # Steepness of the sigmoid curve

            def sigmoid(x, center=center, steepness=steepness):
                # Convert center from correlation to distance
                center_dist = np.sqrt(2 * (1 - center))
                return 1 / (1 + np.exp(-steepness * (x - center_dist)))

            # First get standard distances
            raw_distances = np.sqrt(2 * (1 - correlation_matrix))
            # Apply sigmoid transformation
            distance_matrix = sigmoid(raw_distances)
            # Scale to [0, 2] range for consistency
            distance_matrix = distance_matrix * 2 / distance_matrix.max()

        # Adaptive power transformation (uses different powers for different correlation ranges)
        elif method == "adaptive":
            min_power = kwargs.get("min_power", 0.1)  # For high correlations
            max_power = kwargs.get("max_power", 0.5)  # For low correlations
            threshold_high = kwargs.get("threshold_high", 0.8)
            threshold_low = kwargs.get("threshold_low", 0.4)

            distance_matrix = np.zeros_like(correlation_matrix)
            for i in range(correlation_matrix.shape[0]):
                for j in range(correlation_matrix.shape[1]):
                    corr = correlation_matrix[i, j]
                    # Determine power based on correlation value
                    if corr > threshold_high:
                        # For high correlations, use lower power to stretch differences
                        power = min_power
                    elif corr > threshold_low:
                        # For medium correlations, use interpolated power
                        ratio = (corr - threshold_low) / (threshold_high - threshold_low)
                        power = min_power + (max_power - min_power) * (1 - ratio)
                    else:
                        # For low correlations, use higher power
                        power = max_power

                    # Apply the power transformation
                    distance_matrix[i, j] = (2 * (1 - corr)) ** power

        else:
            raise ValueError(f"Unknown transformation method: {method}")

        # Ensure the matrix is symmetric (may not be necessary for all methods)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        return distance_matrix
    
    def correlation_based_projections_with_testing(self, correlation_matrix, cluster_assignments, color_map=None, force_recompute=False):
        # Run transformation comparison if requested
        ### ! DEPR !
        # if test_transforms:
        #     print("Testing different distance transformation methods...")
        #     distance_matrix = self.apply_best_distance_transformation(
        #         correlation_matrix, 
        #         cluster_assignments=cluster_assignments,
        #         color_map=color_map,
        #         projection_method='mds'
        #     )
        # else:
        # Convert correlation matrix to distance matrix with enhanced separation
        if hasattr(self, 'transform_correlation_to_distance'):
            # Use the enhanced transformation method if available
            distance_matrix = self.transform_correlation_to_distance(
                correlation_matrix, 
                method="standard",  # Options: "standard", "power", "log", "sigmoid", "adaptive"
            )
            #     percentile=75,  # Lower values (0.1-0.4) enhance separation between highly correlated points,
            #     steepness=12
            # )
        else:
            # Fallback to standard transformation
            distance_matrix = np.sqrt(2 * (1 - correlation_matrix))  # Using correlation-based distance
            distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure perfect symmetry

        # Create a dictionary to store all projections
        projections = {
            "distance_matrix": distance_matrix,
        }

        # Generate 2D projections using different methods

        # 1. MDS (equivalent to PCA for distance matrices)
        mds_2d_path = os.path.join(self.output_dir, f"mds_2d_corr.npy")
        if os.path.exists(mds_2d_path) and not force_recompute:
            print(f"Loading existing MDS 2D projection from {mds_2d_path}")
            projections["mds_2d"] = np.load(mds_2d_path)
        else:
            print("Computing MDS 2D projection from correlation matrix...")
            mds_2d = MDS(n_components=2, dissimilarity="precomputed", random_state=33, n_jobs=-1)
            projections["mds_2d"] = mds_2d.fit_transform(distance_matrix)
            np.save(mds_2d_path, projections["mds_2d"])

        # 2. t-SNE with precomputed distances
        tsne_2d_path = os.path.join(self.output_dir, f"tsne_2d_corr.npy")
        if os.path.exists(tsne_2d_path) and not force_recompute:
            print(f"Loading existing t-SNE 2D projection from {tsne_2d_path}")
            projections["tsne_2d_corr"] = np.load(tsne_2d_path)
        else:
            print("Computing t-SNE 2D projection from correlation matrix...")
            tsne_2d = TSNE(n_components=2, metric="precomputed", perplexity=min(30, distance_matrix.shape[0]-1), 
                           init="random", random_state=33)
            projections["tsne_2d_corr"] = tsne_2d.fit_transform(distance_matrix)
            np.save(tsne_2d_path, projections["tsne_2d_corr"])

        # 3. UMAP with precomputed distances
        umap_2d_path = os.path.join(self.output_dir, f"umap_2d_corr.npy")
        if os.path.exists(umap_2d_path) and not force_recompute:
            print(f"Loading existing UMAP 2D projection from {umap_2d_path}")
            projections["umap_2d_corr"] = np.load(umap_2d_path)
        else:
            print("Computing UMAP 2D projection from correlation matrix...")
            umap_2d = umap.UMAP(n_components=2, metric="precomputed", 
                                n_neighbors=min(15, distance_matrix.shape[0]-1),
                                min_dist=0.1, random_state=33)
            projections["umap_2d_corr"] = umap_2d.fit_transform(distance_matrix)
            np.save(umap_2d_path, projections["umap_2d_corr"])

        # 4. Ensemble 2D (combining MDS, t-SNE, and UMAP)
        ensemble_2d_path = os.path.join(self.output_dir, f"ensemble_2d_corr.npy")
        if os.path.exists(ensemble_2d_path) and not force_recompute:
            print(f"Loading existing Ensemble 2D projection from {ensemble_2d_path}")
            projections["ensemble_2d_corr"] = np.load(ensemble_2d_path)
        else:
            print("Computing Ensemble 2D projection from correlation-based projections...")

            # Normalize each projection to [0,1] range for fair combination
            normalized_projections = []

            for key in ["mds_2d", "tsne_2d_corr", "umap_2d_corr"]:
                proj = projections[key]
                # Normalize to [0,1] range
                proj_min = proj.min(axis=0)
                proj_max = proj.max(axis=0)
                # Add small epsilon to avoid division by zero
                norm_proj = (proj - proj_min) / (proj_max - proj_min + 1e-10)  
                normalized_projections.append(norm_proj)

            # Average the normalized projections
            ensemble_proj = np.mean(normalized_projections, axis=0)

            # Scale back to reasonable range for consistency with other plots
            ensemble_proj = (ensemble_proj * 2) - 1

            projections["ensemble_2d_corr"] = ensemble_proj
            np.save(ensemble_2d_path, projections["ensemble_2d_corr"])

        # Generate 3D projections

        # 1. MDS 3D
        mds_3d_path = os.path.join(self.output_dir, f"mds_3d_corr.npy")
        if os.path.exists(mds_3d_path) and not force_recompute:
            print(f"Loading existing MDS 3D projection from {mds_3d_path}")
            projections["mds_3d"] = np.load(mds_3d_path)
        else:
            print("Computing MDS 3D projection from correlation matrix...")
            mds_3d = MDS(n_components=3, dissimilarity="precomputed", random_state=33, n_jobs=-1)
            projections["mds_3d"] = mds_3d.fit_transform(distance_matrix)
            np.save(mds_3d_path, projections["mds_3d"])

        # 2. t-SNE 3D
        tsne_3d_path = os.path.join(self.output_dir, f"tsne_3d_corr.npy")
        if os.path.exists(tsne_3d_path) and not force_recompute:
            print(f"Loading existing t-SNE 3D projection from {tsne_3d_path}")
            projections["tsne_3d_corr"] = np.load(tsne_3d_path)
        else:
            print("Computing t-SNE 3D projection from correlation matrix...")
            tsne_3d = TSNE(n_components=3, metric="precomputed", perplexity=min(30, distance_matrix.shape[0]-1), 
                           init="random", random_state=33)
            projections["tsne_3d_corr"] = tsne_3d.fit_transform(distance_matrix)
            np.save(tsne_3d_path, projections["tsne_3d_corr"])

        # 3. UMAP 3D
        umap_3d_path = os.path.join(self.output_dir, f"umap_3d_corr.npy")
        if os.path.exists(umap_3d_path) and not force_recompute:
            print(f"Loading existing UMAP 3D projection from {umap_3d_path}")
            projections["umap_3d_corr"] = np.load(umap_3d_path)
        else:
            print("Computing UMAP 3D projection from correlation matrix...")
            umap_3d = umap.UMAP(n_components=3, metric="precomputed", 
                               n_neighbors=min(15, distance_matrix.shape[0]-1),
                               min_dist=0.1, random_state=33)
            projections["umap_3d_corr"] = umap_3d.fit_transform(distance_matrix)
            np.save(umap_3d_path, projections["umap_3d_corr"])

        # Create visualizations for each projection with names
        self.plot_2d_projection(projections["mds_2d"], cluster_assignments, "MDS", color_map, 
                              filename=f"mds_2d_corr_with_names", include_names=True)

        self.plot_2d_projection(projections["tsne_2d_corr"], cluster_assignments, "t-SNE", color_map, 
                              filename=f"tsne_2d_corr_with_names", include_names=True)

        self.plot_2d_projection(projections["umap_2d_corr"], cluster_assignments, "UMAP", color_map, 
                              filename=f"umap_2d_corr_with_names", include_names=True)

        # Create visualization for ensemble projection
        self.plot_2d_projection(projections["ensemble_2d_corr"], cluster_assignments, "Ensemble", color_map, 
                              filename=f"ensemble_2d_corr_with_names", include_names=True)

        # Create 3D visualizations
        self.plot_3d_projection(projections["mds_3d"], cluster_assignments, "MDS", color_map, 
                              filename=f"mds_3d_corr_with_names", include_names=True)

        self.plot_3d_projection(projections["tsne_3d_corr"], cluster_assignments, "t-SNE", color_map, 
                              filename=f"tsne_3d_corr_with_names", include_names=True)

        self.plot_3d_projection(projections["umap_3d_corr"], cluster_assignments, "UMAP", color_map, 
                              filename=f"umap_3d_corr_with_names", include_names=True)

        # Create a panel visualization for correlation-based projections
        self.plot_correlation_projection_panel(projections, cluster_assignments, color_map, 
                                             filename="correlation_based_projections_panel")

        return projections

    def plot_correlation_projection_panel(self, projections, cluster_assignments, color_map=None, 
                                         figsize=(18, 16), filename=None, dpi=300, include_density=True):
        # Extract names and ensure order matches projections
        names = [d['Name'] for d in self.definitions]

        # Get cluster IDs in the same order as projections
        clusters = [cluster_assignments[name] for name in names]

        # Create color map if not provided
        if color_map is None:
            unique_clusters = sorted(set(clusters))
            cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))
            color_map = dict(zip(unique_clusters, cluster_colors))

        # Create figure and 2x2 grid
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2, 2, figure=fig, wspace=0.15, hspace=0.15)

        # Define methods to plot (2D projections)
        methods_2d = [
            ("mds_2d", "MDS", 0, 0),
            ("tsne_2d_corr", "t-SNE", 0, 1),
            ("umap_2d_corr", "UMAP", 1, 0),
            ("ensemble_2d_corr", "Ensemble", 1, 1),  # Now using ensemble method
        ]

        # Marker styling
        marker_size = 90
        edge_width = 1.0
        alpha = 0.9

        # Plot each 2D projection in its panel
        for key, method_name, row, col in methods_2d:
            if key is None:  # Skip placeholder
                continue

            ax = fig.add_subplot(gs[row, col])

            # Get projection data
            projection = projections[key]

            # Set panel styling
            ax.set_facecolor('#f8f8f8')
            ax.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)

            for spine in ax.spines.values():
                spine.set(color='#cccccc', linewidth=1)

            # Extract coordinates
            x = projection[:, 0]
            y = projection[:, 1]

            # Add density visualization if requested
            if include_density:
                # Calculate density estimation
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                z = kde(xy)

                # Create a fine grid for contour plotting
                x_min, x_max = x.min() - 0.1 * np.ptp(x), x.max() + 0.1 * np.ptp(x)
                y_min, y_max = y.min() - 0.1 * np.ptp(y), y.max() + 0.1 * np.ptp(y)
                xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xi.ravel(), yi.ravel()])

                # Calculate density on the grid
                zi = kde(positions).reshape(xi.shape)

                # Plot density as filled contours with a light alpha
                contour = ax.contourf(xi, yi, zi, 15, cmap='viridis', alpha=0.10, zorder=1)

                # Add contour lines
                ax.contour(xi, yi, zi, 8, colors='black', linewidths=0.5, alpha=0.25, zorder=2)

            # Plot each point with its cluster color
            for j, (x_j, y_j) in enumerate(projection):
                cluster = clusters[j]
                color = color_map[cluster]
                ax.scatter(x_j, y_j, c=[color], s=marker_size, alpha=alpha, 
                         edgecolors='black', linewidths=edge_width, zorder=10)

            # Set title and labels
            ax.set_title(f"{method_name}", fontsize=14, fontweight='bold')
            # ax.set_xlabel(f"Dimension 1", fontsize=10, fontweight='bold')
            # ax.set_ylabel(f"Dimension 2", fontsize=10, fontweight='bold')

        # Add a common legend at the bottom with cluster titles
        try:
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[cluster], 
                                     markeredgecolor='black',
                                     markeredgewidth=1,
                                     markersize=10, 
                                     label=f'{self.get_cluster_label(cluster)}') 
                              for cluster in sorted(color_map.keys())]
        except AttributeError:
            # Fall back to cluster numbers if get_cluster_label is not defined
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[cluster], 
                                     markeredgecolor='black',
                                     markeredgewidth=1,
                                     markersize=10, 
                                     label=f'Cluster {cluster}') 
                              for cluster in sorted(color_map.keys())]

        # Place legend below the subplots
        fig.legend(handles=legend_elements, title='Clusters', 
                  title_fontsize=12, fontsize=10,
                  loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=min(5, len(color_map.keys())), 
                  framealpha=0.9, 
                  edgecolor='#cccccc')

        # Add overall title
        plt.suptitle("Correlation Feature Projection - What is Life", 
                    fontsize=18, fontweight='bold', y=0.98)

        # Save figure if filename provided
        if filename:
            fout = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(fout, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {fout}")

        return fig
    
    #############################
    ### SEMANTIC LLM ANALYSIS ###
    #############################
    def print_cluster_definitions(self, cluster_assignments, definitions, color_map=None):
        # Create a reverse mapping of clusters to names
        clusters_to_names = {}
        for name, cluster in cluster_assignments.items():
            if cluster not in clusters_to_names:
                clusters_to_names[cluster] = []
            clusters_to_names[cluster] = clusters_to_names[cluster] + [name]

        # Create a name to definition mapping for quick lookup
        name_to_def = {d['Name']: d['Definition'] for d in definitions}

        # Print each cluster with its definitions
        print("\nCLUSTER ANALYSIS OF LIFE DEFINITIONS")
        print("====================================\n")

        for cluster_num in sorted(clusters_to_names.keys()):
            # Print cluster header
            print(f"\nCluster {cluster_num}")
            print("-" * 50)

            # Print each definition in the cluster
            for name in sorted(clusters_to_names[cluster_num]):
                print(f"\n{name}:")
                # Wrap the definition text for better readability
                definition = name_to_def[name]
                # Add indentation to wrapped text
                wrapped_def = "\n".join(
                    "    " + line 
                    for line in textwrap.wrap(definition, width=80)
                )
                print(wrapped_def)

            print("\n" + "=" * 50)  # Separator between clusters

    def analyze_clusters(self, cluster_assignments, definitions):
        # Create reverse mapping of clusters to names
        clusters_to_names = {}
        for name, cluster in cluster_assignments.items():
            if cluster not in clusters_to_names:
                clusters_to_names[cluster] = []
            clusters_to_names[cluster] = clusters_to_names[cluster] + [name]

        # Analyze cluster sizes
        cluster_sizes = {cluster: len(names) for cluster, names in clusters_to_names.items()}

        # Calculate basic statistics
        total_items = sum(cluster_sizes.values())
        avg_cluster_size = total_items / len(cluster_sizes)

        # Print analysis
        print("\nCLUSTER STATISTICS")
        print("=================")
        print(f"\nTotal number of clusters: {len(cluster_sizes)}")
        print(f"Average cluster size: {avg_cluster_size:.1f} definitions")
        print("\nCluster sizes:")
        for cluster, size in sorted(cluster_sizes.items()):
            print(f"Cluster {cluster}: {size} definitions ({size/total_items*100:.1f}%)")

        return {
            'n_clusters': len(cluster_sizes),
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': avg_cluster_size,
            'total_items': total_items
        }
    
    async def get_cluster_analysis(self, cluster_assignments, definitions):
        # Create a reverse mapping of clusters to names
        clusters_to_names = {}
        for name, cluster in cluster_assignments.items():
            if cluster not in clusters_to_names:
                clusters_to_names[cluster] = []
            clusters_to_names[cluster] = clusters_to_names[cluster] + [name]

        # Create a name to definition mapping for quick lookup
        name_to_def = {d['Name']: d['Definition'] for d in definitions}

        ### GROUP DEFINITIONS TO CLUSTERS AND CALL CLAUDE
        cluster_analysis = {}
        for cluster_num in tqdm(sorted(clusters_to_names.keys())):
            cluster_definitions = []
            cluster_analysis[cluster_num] = {}

            cluster_names = []
            for name in sorted(clusters_to_names[cluster_num]):   
                definition = name_to_def[name]
                cluster_definitions.append(definition)
                cluster_names.append(name)
            cluster_analysis[cluster_num]['names'] = cluster_names

            ### CREATE A PROMPT STRING FROM THE GROUP'S DEFINITIONS
            definitions_str = '\n\n------\n'.join(cluster_definitions)
            # cluster_analysis[cluster_num]['definitions'] = definitions_str
            
            ### a) GET ANALYSIS
            analysis_template = self.Inference._read_prompt_template("cluster_ideas")
            group_analysis, _ = await self.Inference.acomplete(
                text=definitions_str, 
                system_prompt=analysis_template, 
                model=self.model
            )
            cluster_analysis[cluster_num]['group_analysis'] = group_analysis
            
            ### b) GET CONSENSUS DEFINITION FOR CLUSTER
            consensus_template = self.Inference._read_prompt_template("cluster_consensus")
            consensus_input_text = f"Here are the definitions:\n{definitions_str}\n---\nHere is the thematic analysis conducted on these definitions:\n{group_analysis}\n"
            consensus_definition, _ = await self.Inference.acomplete(
                text=consensus_input_text, 
                system_prompt=consensus_template, 
                model=self.model
            )
            cluster_analysis[cluster_num]['consensus_definition'] = consensus_definition
            
            ### c) GET CLUSTER TITLE
            title_template = self.Inference._read_prompt_template("cluster_title")
            title_input_text = f"Consensus Definition: {consensus_definition}\n---\nThematic Analysis: {group_analysis}\n"
            cluster_title, _ = await self.Inference.acomplete(
                text=title_input_text, 
                system_prompt=title_template, 
                model=self.model
            )
            cluster_analysis[cluster_num]['title'] = cluster_title
            
            # Store title in self.cluster_titles
            self.cluster_titles[cluster_num] = cluster_title

        return cluster_analysis
    
    def print_cluster_analysis(self, cluster_analysis, markdown_filename=None):
        # Create a string to store the markdown content
        markdown_content = ""

        for cluster_num in sorted(cluster_analysis.keys()):
            # Print to console
            print(f"\n{'='*80}")
            print(f"CLUSTER {cluster_num}")
            print(f"{'='*80}")
            # Print members
            print("\nMEMBERS:")
            print("-" * 40)
            for name in cluster_analysis[cluster_num]['names']:
                print(f"• {name}")
            # Print group title
            print("\nTitle:")
            print("-" * 40)
            print(cluster_analysis[cluster_num]['title'])
            # Print consensus
            print("\nCONSENSUS:")
            print("-" * 40)
            print(cluster_analysis[cluster_num]['consensus_definition'])
            # Print analysis
            print("\nANALYSIS:")
            print("-" * 40)
            print(cluster_analysis[cluster_num]['group_analysis'])
           

            # Add to markdown content
            markdown_content += f"# CLUSTER {cluster_num}\n\n"
            markdown_content += "## MEMBERS\n\n"
            for name in cluster_analysis[cluster_num]['names']:
                markdown_content += f"* {name}\n"
            markdown_content += "\n## TITLE\n\n"
            markdown_content += f"{cluster_analysis[cluster_num]['title']}\n\n"
            markdown_content += "\n## CONSENSUS\n\n"
            markdown_content += f"{cluster_analysis[cluster_num]['consensus_definition']}\n\n"
            markdown_content += "## ANALYSIS\n\n"
            markdown_content += f"{cluster_analysis[cluster_num]['group_analysis']}\n\n"
            markdown_content += "---\n\n"

        # Save to markdown file if filename is provided
        if markdown_filename:
            fout = os.path.join(self.output_dir, markdown_filename)
            try:
                with open(fout, 'w') as f:
                    f.write(markdown_content)
                print(f"\nCluster analysis saved to {fout}")
            except Exception as e:
                print(f"\nError saving markdown file: {e}")
                
    ##############################################
    ### EMBEDDING PROJECTION AND VISUALIZATION ###
    ##############################################
    def get_cluster_label(self, cluster):
        if hasattr(self, 'cluster_titles') and cluster in self.cluster_titles:
            return self.cluster_titles[cluster]
        return f"Cluster {cluster}"
    
    def get_definition_embeddings(self, embedding_type="openai", force_recompute=False):
        ### SET EMBEDDING TYPE + DIMENSION
        if embedding_type not in ["openai", "bedrock"]:
            raise ValueError("Embedding type must be 'openai' or 'bedrock'")
        if embedding_type == "openai":
            embedding_dim = self.Inference.openai_embedding_dimensions
        else:  # bedrock
            embedding_dim = self.Inference.bedrock_embedding_dimensions

        ### READ EMBEDDINGS FROM A FILE IF THEY EXIST
        embedding_path = os.path.join(self.embeddings_dir, f"embeddings_{embedding_type}.npy")
        if os.path.exists(embedding_path) and not force_recompute:
            print(f"Loading existing embeddings from {embedding_path}")
            return np.load(embedding_path)

        ### INITIALIZE EMPTY MATRIX AND GET EMBEDDINGS FOR EACH DEFINITION
        n_definitions = len(self.definitions)
        embeddings = np.zeros((n_definitions, embedding_dim))
        for i, definition in enumerate(tqdm(self.definitions, desc="Getting embeddings")):
            text = definition['Definition']
            if embedding_type == "openai":
                embedding = self.Inference.openai_embedding(text)
            else:  # bedrock
                embedding = self.Inference.bedrock_embedding(text)
            embeddings[i] = embedding
        np.save(embedding_path, embeddings)
        print(f"Embeddings saved to {embedding_path}")
        return embeddings

    ### PROJECT DEFINITIONS VIA UMAP
    def umap_projection(self, embeddings, n_components=2, n_neighbors=18, min_dist=0.09, random_state=33, force_recompute=False):
        umap_path = os.path.join(self.output_dir, f"umap_projection_{n_components}d.npy")
        if os.path.exists(umap_path) and not force_recompute:
            print(f"Loading existing UMAP projection from {umap_path}")
            return np.load(umap_path)

        ### INITIALIZE UMAP + FIT / TRANSFORM
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        ### PCA INTERMEDIATE TO REMOVE NOISE --> UMAP
        # embeddings = self.pca_projection(embeddings, n_components=33)
        umap_embeddings = umap_model.fit_transform(embeddings)
        np.save(umap_path, umap_embeddings)
        print(f"UMAP projection saved to {umap_path}")
        return umap_embeddings

    ### PCA PROJECTION
    def pca_projection(self, embeddings, n_components=2, random_state=33, force_recompute=False):
        pca_path = os.path.join(self.output_dir, f"pca_projection_{n_components}d.npy")
        if os.path.exists(pca_path) and not force_recompute:
            print(f"Loading existing PCA projection from {pca_path}")
            return np.load(pca_path)

        ### INITIALIZE PCA + FIT DATA
        pca_model = PCA(n_components=n_components, random_state=random_state)
        pca_embeddings = pca_model.fit_transform(embeddings)
        np.save(pca_path, pca_embeddings)
        print(f"PCA projection saved to {pca_path}")
        explained_variance = pca_model.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Cumulative explained variance: {np.sum(explained_variance):.4f}")
        return pca_embeddings

    def tsne_projection(self, embeddings, n_components=2, perplexity=24, random_state=33, force_recompute=False):
        tsne_path = os.path.join(self.output_dir, f"tsne_projection_{n_components}d.npy")
        if os.path.exists(tsne_path) and not force_recompute:
            print(f"Loading existing t-SNE projection from {tsne_path}")
            return np.load(tsne_path)

        ### INITIALIZE tSNE + TRANSFORM DATA
        tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state
        )
        ### PCA INTERMEDIATE TO REMOVE NOISE --> UMAP
        # embeddings = self.pca_projection(embeddings, n_components=33)
        tsne_embeddings = tsne_model.fit_transform(embeddings)
        np.save(tsne_path, tsne_embeddings)
        print(f"t-SNE projection saved to {tsne_path}")
        return tsne_embeddings

    ### ENSEMBLE PROJECTION METHOD
    def ensemble_projection(self, embeddings, methods=['umap', 'pca', 'tsne'], n_components=2, force_recompute=False):
        ensemble_path = os.path.join(self.output_dir, f"ensemble_projection_{n_components}d.npy")
        if os.path.exists(ensemble_path) and not force_recompute:
            print(f"Loading existing ensemble projection from {ensemble_path}")
            return np.load(ensemble_path)

        projections = []

        if 'umap' in methods:
            umap_proj = self.umap_projection(embeddings, n_components=n_components, force_recompute=force_recompute)
            # Normalize to [0,1] range
            umap_min = umap_proj.min(axis=0)
            umap_max = umap_proj.max(axis=0)
            umap_norm = (umap_proj - umap_min) / (umap_max - umap_min + 1e-10)  # Add small epsilon to avoid division by zero
            projections.append(umap_norm)

        if 'pca' in methods:
            pca_proj = self.pca_projection(embeddings, n_components=n_components, force_recompute=force_recompute)
            pca_min = pca_proj.min(axis=0)
            pca_max = pca_proj.max(axis=0)
            pca_norm = (pca_proj - pca_min) / (pca_max - pca_min + 1e-10)
            projections.append(pca_norm)

        if 'tsne' in methods:
            tsne_proj = self.tsne_projection(embeddings, n_components=n_components, force_recompute=force_recompute)
            tsne_min = tsne_proj.min(axis=0)
            tsne_max = tsne_proj.max(axis=0)
            tsne_norm = (tsne_proj - tsne_min) / (tsne_max - tsne_min + 1e-10)
            projections.append(tsne_norm)

        # Average the normalized projections
        ensemble_proj = np.mean(projections, axis=0)

        # Scale back to reasonable range for consistency with other plots
        ensemble_proj = (ensemble_proj * 2) - 1

        # Save the projection
        np.save(ensemble_path, ensemble_proj)
        print(f"Ensemble projection saved to {ensemble_path}")

        return ensemble_proj

    # plot_2d_projection to use last names and cluster titles
    def plot_2d_projection(self, projection, cluster_assignments, method_name, color_map=None, 
                           figsize=(14, 12), filename=None, include_names=True, dpi=300, include_density=True):
        # Use last names instead of full names
        names = [d['Last'] for d in self.definitions]
        full_names = [d['Name'] for d in self.definitions]

        # Get cluster IDs in the same order as projection
        clusters = [cluster_assignments[d['Name']] for d in self.definitions]

        # Create color map if not provided
        if color_map is None:
            unique_clusters = sorted(set(clusters))
            cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))
            color_map = dict(zip(unique_clusters, cluster_colors))

        # Create figure with improved styling
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Add subtle light gray background with white grid
        ax.set_facecolor('#f8f8f8')
        ax.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)

        # Extract x and y coordinates
        x = projection[:, 0]
        y = projection[:, 1]

        # Add density visualization if requested
        if include_density:
            # 1. Calculate density estimation
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)

            # 2. Add density-based contours
            # Create a fine grid for contour plotting
            x_min, x_max = x.min() - 0.1 * np.ptp(x), x.max() + 0.1 * np.ptp(x)
            y_min, y_max = y.min() - 0.1 * np.ptp(y), y.max() + 0.1 * np.ptp(y)
            xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xi.ravel(), yi.ravel()])

            # Calculate density on the grid
            zi = gaussian_kde(xy)(positions).reshape(xi.shape)

            # 3. Plot density as filled contours with a light alpha
            contour = ax.contourf(xi, yi, zi, 15, cmap='viridis', alpha=0.12, zorder=1)

            # 4. Add contour lines
            contour_lines = ax.contour(xi, yi, zi, 8, colors='black', linewidths=0.5, alpha=0.25, zorder=2)

            # # 5. Add colorbar for density
            # cbar = plt.colorbar(contour, ax=ax, pad=0.01, aspect=30, shrink=0.7)
            # cbar.ax.set_ylabel('Density', fontsize=10, fontweight='bold')
            # cbar.ax.tick_params(labelsize=8)

        # Calculate better offsets for text labels
        x_range = np.ptp(x)
        y_range = np.ptp(y)
        x_offset = -x_range * 0.009  # Slight offset to the left
        y_offset = y_range * 0.009   # Slight offset upward

        # Adjust marker properties for clarity
        marker_size = 140
        edge_width = 1.4
        alpha = 0.9

        # Plot each point with its cluster color
        for i, (x_i, y_i) in enumerate(projection):
            cluster = clusters[i]
            color = color_map[cluster]
            ax.scatter(x_i, y_i, c=[color], s=marker_size, alpha=alpha, 
                      edgecolors='black', linewidths=edge_width, zorder=20)

            # Add names with improved positioning if requested
            if include_names:
                # Use text with subtle background for better readability
                ax.text(x_i + x_offset, y_i + y_offset, names[i], 
                       fontsize=12, ha='right', va='bottom', 
                       bbox=dict(facecolor='white', alpha=0.85, 
                                 edgecolor='none', pad=1, boxstyle='round,pad=0.12'),
                       zorder=30)  # Ensure text is on top

        # Set title and labels with improved styling
        title_font = FontProperties(family='DejaVu Sans', weight='bold', size=18)
        title = r"$\mathit{Life}$ - " + method_name + " Definitional Landscape"
        ax.set_title(title, fontproperties=title_font, pad=20)
        ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12, fontweight='bold')

        # Add subtle border around the plot
        for spine in ax.spines.values():
            spine.set(color='#cccccc', linewidth=1)

        # Add legend with improved styling - use cluster titles instead of numbers
        # This assumes you have a get_cluster_label method; if not, revert to cluster numbers
        try:
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[cluster], 
                                     markeredgecolor='black',
                                     markeredgewidth=1,
                                     markersize=12, 
                                     label=f'{self.get_cluster_label(cluster)}') 
                              for cluster in sorted(color_map.keys())]
        except AttributeError:
            # Fall back to cluster numbers if get_cluster_label is not defined
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[cluster], 
                                     markeredgecolor='black',
                                     markeredgewidth=1,
                                     markersize=12, 
                                     label=f'Cluster {cluster}') 
                              for cluster in sorted(color_map.keys())]

        ax.legend(handles=legend_elements, title='Clusters', 
                 title_fontsize=14, fontsize=12,
                 loc='best', framealpha=0.9, 
                 edgecolor='#cccccc')

        # Adjust layout
        plt.tight_layout()

        # Save figure if filename provided
        if filename:
            fout = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(fout, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {fout}")

        return fig, ax


    # Modify plot_3d_projection to use last names and cluster titles
    def plot_3d_projection(self, projection, cluster_assignments, method_name, color_map=None, 
                          figsize=(10, 9), filename=None, include_names=True, dpi=300):
        # Check if projection is 3D
        if projection.shape[1] != 3:
            raise ValueError("Projection must have 3 dimensions for 3D plotting")

        # Extract last names and ensure order matches projection
        names = [d['Last'] for d in self.definitions]

        # Get cluster IDs in the same order as projection
        clusters = [cluster_assignments[d['Name']] for d in self.definitions]

        # Create color map if not provided
        if color_map is None:
            unique_clusters = sorted(set(clusters))
            cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))
            color_map = dict(zip(unique_clusters, cluster_colors))

        # Create figure and 3D axes with improved styling
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Improve 3D plot aesthetics
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.zaxis.pane.set_edgecolor('#cccccc')
        ax.grid(color='#eeeeee', linestyle='-', linewidth=0.5, alpha=0.8)

        # Calculate improved offsets for text labels
        x_range = np.ptp(projection[:, 0])
        y_range = np.ptp(projection[:, 1])
        z_range = np.ptp(projection[:, 2])
        x_offset = -x_range * 0.01
        y_offset = y_range * 0.01
        z_offset = z_range * 0.01

        # Adjust marker properties
        marker_size = 90
        edge_width = 0.7
        alpha = 0.9

        # Plot each point with its cluster color
        for i, (x, y, z) in enumerate(projection):
            cluster = clusters[i]
            color = color_map[cluster]
            ax.scatter(x, y, z, c=[color], s=marker_size, alpha=alpha, 
                      edgecolors='black', linewidths=edge_width, zorder=10)

            # Add names with improved positioning if requested
            if include_names:
                ax.text(x + x_offset, y + y_offset, z + z_offset, names[i], 
                       fontsize=8, ha='right', va='bottom', zorder=20)

        # Set title and labels with improved styling
        title = f"3D {method_name} Projection of Life Definitions"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12, fontweight='bold')
        ax.set_zlabel(f"{method_name} Dimension 3", fontsize=12, fontweight='bold')

        # Add legend with improved styling - use cluster titles
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color_map[cluster], 
                                markeredgecolor='black',
                                markeredgewidth=1,
                                markersize=10, 
                                label=f'{self.get_cluster_label(cluster)}') 
                          for cluster in sorted(color_map.keys())]

        ax.legend(handles=legend_elements, title='Clusters', 
                 title_fontsize=12, fontsize=10,
                 loc='best', framealpha=0.9, 
                 edgecolor='#cccccc')

        # Adjust layout
        plt.tight_layout()

        # Save figure if filename provided
        if filename:
            fout = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(fout, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {fout}")

        return fig, ax

    ### 2D PANNEL PROJECTION
    def plot_2d_panel(self, projections, cluster_assignments, color_map=None, 
                     figsize=(18, 16), filename=None, dpi=400, include_density=True):
        # Extract names and ensure order matches projections
        names = [d['Name'] for d in self.definitions]

        # Get cluster IDs in the same order as projections
        clusters = [cluster_assignments[name] for name in names]

        # Create color map if not provided
        if color_map is None:
            unique_clusters = sorted(set(clusters))
            cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))
            color_map = dict(zip(unique_clusters, cluster_colors))

        # Create figure and 2x2 grid
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2, 2, figure=fig, wspace=0.15, hspace=0.15)

        # Define methods to plot (2D projections)
        methods_2d = [
            ("umap_2d", "UMAP", 0, 0),
            ("pca_2d", "PCA", 0, 1),
            ("tsne_2d", "t-SNE", 1, 0),
            ("ensemble_2d", "Ensemble", 1, 1)
        ]

        # Marker styling
        marker_size = 90
        edge_width = 0.9
        alpha = 0.9

        # Plot each 2D projection in its panel
        for key, method_name, row, col in methods_2d:
            ax = fig.add_subplot(gs[row, col])

            # Get projection data (handling ensemble separately)
            if key == "ensemble_2d":
                projection = projections.get(key, None)
                if projection is None:
                    # Generate ensemble if not provided
                    projection = self.ensemble_projection(projections["embeddings"], n_components=2)
                    projections["ensemble_2d"] = projection
            else:
                projection = projections[key]

            # Set panel styling
            ax.set_facecolor('#f8f8f8')
            ax.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)

            for spine in ax.spines.values():
                spine.set(color='#cccccc', linewidth=1)

            # Extract coordinates
            x = projection[:, 0]
            y = projection[:, 1]

            # Add density visualization if requested
            if include_density:
                # Calculate density estimation
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                z = kde(xy)

                # Create a fine grid for contour plotting
                x_min, x_max = x.min() - 0.1 * np.ptp(x), x.max() + 0.1 * np.ptp(x)
                y_min, y_max = y.min() - 0.1 * np.ptp(y), y.max() + 0.1 * np.ptp(y)
                xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xi.ravel(), yi.ravel()])

                # Calculate density on the grid
                zi = kde(positions).reshape(xi.shape)

                # Plot density as filled contours with a light alpha
                contour = ax.contourf(xi, yi, zi, 15, cmap='viridis', alpha=0.15, zorder=1)

                # Add contour lines
                ax.contour(xi, yi, zi, 8, colors='black', linewidths=0.5, alpha=0.25, zorder=2)

                # # Add a small colorbar if this is the first subplot
                # if row == 0 and col == 0:
                #     cbar = plt.colorbar(contour, ax=ax, pad=0.01, aspect=30, shrink=0.7)
                #     cbar.ax.set_ylabel('Density', fontsize=10, fontweight='bold')
                #     cbar.ax.tick_params(labelsize=8)

            # Plot each point with its cluster color
            for j, (x_j, y_j) in enumerate(projection):
                cluster = clusters[j]
                color = color_map[cluster]
                ax.scatter(x_j, y_j, c=[color], s=marker_size, alpha=alpha, 
                         edgecolors='black', linewidths=edge_width, zorder=10)

            # Set title and labels
            ax.set_title(f"2D {method_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel(f"Dimension 1", fontsize=10, fontweight='bold')
            ax.set_ylabel(f"Dimension 2", fontsize=10, fontweight='bold')

        # Add a common legend at the bottom
        try:
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[cluster], 
                                     markeredgecolor='black',
                                     markeredgewidth=1,
                                     markersize=10, 
                                     label=f'{self.get_cluster_label(cluster)}') 
                              for cluster in sorted(color_map.keys())]
        except AttributeError:
            # Fall back to cluster numbers if get_cluster_label is not defined
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[cluster], 
                                     markeredgecolor='black',
                                     markeredgewidth=1,
                                     markersize=10, 
                                     label=f'Cluster {cluster}') 
                              for cluster in sorted(color_map.keys())]

        # Place legend below the subplots
        fig.legend(handles=legend_elements, title='Clusters', 
                  title_fontsize=12, fontsize=10,
                  loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=min(5, len(color_map.keys())), 
                  framealpha=0.9, 
                  edgecolor='#cccccc')

        # Add overall title
        plt.suptitle("2D Dimensionality Reduction Techniques", 
                    fontsize=18, fontweight='bold', y=0.98)

        # Save figure if filename provided
        if filename:
            fout = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(fout, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {fout}")

        return fig

    def plot_3d_panel(self, projections, cluster_assignments, color_map=None, 
                     figsize=(18, 16), filename=None, dpi=300):
        # Extract last names and ensure order matches projections
        names = [d['Last'] for d in self.definitions]

        # Get cluster IDs in the same order as projections
        clusters = [cluster_assignments[d['Name']] for d in self.definitions]

        # Create color map if not provided
        if color_map is None:
            unique_clusters = sorted(set(clusters))
            cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))
            color_map = dict(zip(unique_clusters, cluster_colors))

        # Create figure and 2x2 grid
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2, 2, figure=fig, wspace=0.15, hspace=0.15)

        # Define methods to plot (3D projections)
        methods_3d = [
            ("umap_3d", "UMAP", 0, 0),
            ("pca_3d", "PCA", 0, 1),
            ("tsne_3d", "t-SNE", 1, 0),
            ("ensemble_3d", "Ensemble", 1, 1)
        ]

        # Marker styling
        marker_size = 80
        edge_width = 0.7
        alpha = 0.9

        # Plot each 3D projection in its panel
        for key, method_name, row, col in methods_3d:
            ax = fig.add_subplot(gs[row, col], projection='3d')

            # Get projection data (handling ensemble separately)
            if key == "ensemble_3d":
                projection = projections.get(key, None)
                if projection is None:
                    # Generate ensemble if not provided
                    projection = self.ensemble_projection(projections["embeddings"], n_components=3)
                    projections["ensemble_3d"] = projection
            else:
                projection = projections[key]

            # Improve 3D plot aesthetics
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#cccccc')
            ax.yaxis.pane.set_edgecolor('#cccccc')
            ax.zaxis.pane.set_edgecolor('#cccccc')
            ax.grid(color='#eeeeee', linestyle='-', linewidth=0.5, alpha=0.8)

            # Plot each point with its cluster color
            for j, (x, y, z) in enumerate(projection):
                cluster = clusters[j]
                color = color_map[cluster]
                ax.scatter(x, y, z, c=[color], s=marker_size, alpha=alpha, 
                          edgecolors='black', linewidths=edge_width, zorder=10)

            # Set title and labels
            ax.set_title(f"3D {method_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel(f"Dim 1", fontsize=10, fontweight='bold')
            ax.set_ylabel(f"Dim 2", fontsize=10, fontweight='bold')
            ax.set_zlabel(f"Dim 3", fontsize=10, fontweight='bold')

        # Add a common legend at the bottom with cluster titles
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color_map[cluster], 
                                 markeredgecolor='black',
                                 markeredgewidth=1,
                                 markersize=10, 
                                 label=f'{self.get_cluster_label(cluster)}') 
                          for cluster in sorted(color_map.keys())]

        # Place legend below the subplots
        fig.legend(handles=legend_elements, title='Clusters', 
                  title_fontsize=12, fontsize=10,
                  loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=min(5, len(color_map.keys())), 
                  framealpha=0.9, 
                  edgecolor='#cccccc')

        # Add overall title
        plt.suptitle("3D Dimensionality Reduction Techniques", 
                    fontsize=18, fontweight='bold', y=0.98)

        # Save figure if filename provided
        if filename:
            fout = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(fout, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {fout}")

        return fig

    ### ENHANCED PROJECT AND VISUALIZE METHOD
    def project_and_visualize_embeddings(self, embedding_type="openai", cluster_assignments=None, color_map=None, force_recompute=False):
        # Get embeddings
        embeddings = self.get_definition_embeddings(embedding_type=embedding_type, force_recompute=force_recompute)

        # If no cluster assignments provided, use existing ones
        if cluster_assignments is None:
            # Load correlation matrix if needed
            corr_path = os.path.join(self.output_dir, f"m_correlation_{self.model}.npy")
            if not os.path.exists(corr_path):
                raise ValueError("No correlation matrix found. Please run correlation analysis first.")

            correlation_matrix = np.load(corr_path)

            # Get cluster assignments from correlation matrix
            _, _, _, cluster_assignments, _, color_map = self.plot_clustered_correlation_heatmap(
                correlation_matrix, self.definitions, filename=None
            )

        # Calculate all projections
        projections = {
            "embeddings": embeddings,
        }

        # 2D projections
        projections["umap_2d"] = self.umap_projection(embeddings, n_components=2, force_recompute=force_recompute)
        projections["pca_2d"] = self.pca_projection(embeddings, n_components=2, force_recompute=force_recompute)
        projections["tsne_2d"] = self.tsne_projection(embeddings, n_components=2, force_recompute=force_recompute)
        projections["ensemble_2d"] = self.ensemble_projection(embeddings, n_components=2, force_recompute=force_recompute)

        # 3D projections
        projections["umap_3d"] = self.umap_projection(embeddings, n_components=3, force_recompute=force_recompute)
        projections["pca_3d"] = self.pca_projection(embeddings, n_components=3, force_recompute=force_recompute)
        projections["tsne_3d"] = self.tsne_projection(embeddings, n_components=3, force_recompute=force_recompute)
        projections["ensemble_3d"] = self.ensemble_projection(embeddings, n_components=3, force_recompute=force_recompute)

        # Create individual plots with names
        self.plot_2d_projection(projections["umap_2d"], cluster_assignments, "UMAP - Embeddings", color_map, 
                              filename=f"umap_2d_{embedding_type}_with_names", include_names=True)

        self.plot_2d_projection(projections["pca_2d"], cluster_assignments, "PCA - Embeddings", color_map, 
                              filename=f"pca_2d_{embedding_type}_with_names", include_names=True)

        self.plot_2d_projection(projections["tsne_2d"], cluster_assignments, "t-SNE - Embeddings", color_map, 
                              filename=f"tsne_2d_{embedding_type}_with_names", include_names=True)

        self.plot_2d_projection(projections["ensemble_2d"], cluster_assignments, "Ensemble - Embeddings", color_map, 
                              filename=f"ensemble_2d_{embedding_type}_with_names", include_names=True)

        # Create 3D plots with names if needed
        self.plot_3d_projection(projections["umap_3d"], cluster_assignments, "UMAP - Embeddings", color_map, 
                              filename=f"umap_3d_{embedding_type}_with_names", include_names=True)

        self.plot_3d_projection(projections["pca_3d"], cluster_assignments, "PCA - Embeddings", color_map, 
                              filename=f"pca_3d_{embedding_type}_with_names", include_names=True)

        self.plot_3d_projection(projections["tsne_3d"], cluster_assignments, "t-SNE - Embeddings", color_map, 
                              filename=f"tsne_3d_{embedding_type}_with_names", include_names=True)

        self.plot_3d_projection(projections["ensemble_3d"], cluster_assignments, "Ensemble - Embeddings", color_map, 
                              filename=f"ensemble_3d_{embedding_type}_with_names", include_names=True)

        # Create 2D and 3D panel visualizations (without names)
        self.plot_2d_panel(projections, cluster_assignments, color_map, 
                          filename=f"2d_projections_panel_{embedding_type}")

        self.plot_3d_panel(projections, cluster_assignments, color_map, 
                          filename=f"3d_projections_panel_{embedding_type}")

        return projections

    #################################
    ### COMPLETE ANALYSIS WRAPPER ###
    #################################
    def analyze_definitions(self):
        ### COMPUTE CORRELATION MATRICES OF RESPONSES
        M, S = self.create_correlation_matrix()
        
        ### GENERATE CLUSTERED MATRIX
        fig, ax_heatmap, ax_dendrogram, cluster_assignments, reordered_idx, color_map = self.plot_clustered_correlation_heatmap(M, self.definitions, filename='clustered_correlations.png')
        plt.show()
        
        ### PERFORM SEMANTIC ANALYSIS
        stats = self.analyze_clusters(cluster_assignments, self.definitions)
        # self.print_cluster_definitions(cluster_assignments, self.definitions, color_map)
        analysis = asyncio.run(self.get_cluster_analysis(cluster_assignments, self.definitions))
        self.print_cluster_analysis(analysis, markdown_filename="cluster_semantic_analysis.md")
        
        ### PROJECT INTO EMBEDDING SPACE WITH APPLIED COLOR MAP
        projections = self.project_and_visualize_embeddings(
            embedding_type="bedrock",
            cluster_assignments=cluster_assignments,
            color_map=color_map,
            force_recompute=True
        )
        
        ### EMBED INTO 2/3D DIRECTLY FROM CORRELATION FEATURE VECTORS
        corr_projections = self.correlation_based_projections_with_testing(
            correlation_matrix=M,
            cluster_assignments=cluster_assignments,
            color_map=color_map,
            force_recompute=True
        )

        ### ! DONE !
        print("\nAnalysis complete! All visualizations and results have been saved to the output directory.")


        
  ### DEPR ###      
      
        
          
#     ### COMPARE CORRELATION --> DISTANCE MATRIX TRANSFORMATION OPTIONS FOR BEST OPTION THAT HIGHLIGHTS DISTANCES IN HIGHLY CORRELATED INDIVIDUALS
#     def compare_distance_transformations(self, correlation_matrix, cluster_assignments, color_map=None, 
#                                         projection_method="mds", n_components=2, save_plot=True):
#         # Define transformation methods to test
#         transformation_methods = {
#             "standard": {"method": "standard"},
#             "power_0.5": {"method": "power", "power": 0.5},
#             "power_0.3": {"method": "power", "power": 0.3},
#             "power_0.1": {"method": "power", "power": 0.1},
#             "log": {"method": "log"},
#             "sigmoid": {"method": "sigmoid", "center": 0.7, "steepness": 12},
#             "adaptive": {"method": "adaptive", "min_power": 0.1, "max_power": 0.5}
#         }

#         # Dictionary to store results
#         results = {}

#         # Define projection method to use
#         def apply_projection(distance_matrix, method="mds"):
#             if method == "mds":
#                 model = MDS(n_components=n_components, dissimilarity="precomputed", 
#                            random_state=33, n_jobs=-1)
#             elif method == "tsne":
#                 model = TSNE(n_components=n_components, metric="precomputed", 
#                             init="random", random_state=33,
#                             perplexity=min(30, distance_matrix.shape[0]-1))
#             elif method == "umap":
#                 model = umap.UMAP(n_components=n_components, metric="precomputed", 
#                                 n_neighbors=min(15, distance_matrix.shape[0]-1),
#                                 min_dist=0.1, random_state=33)
#             else:
#                 raise ValueError(f"Unknown projection method: {method}")

#             projection = model.fit_transform(distance_matrix)
#             return projection

#         # Compute stress metric (how well distances are preserved)
#         def compute_stress(orig_distances, projection):
#             # Compute pairwise distances in projection space
#             proj_distances = squareform(pdist(projection))

#             # Flatten the distance matrices for comparison (upper triangle)
#             n = orig_distances.shape[0]
#             flat_orig = np.array([orig_distances[i, j] for i in range(n) for j in range(i+1, n)])
#             flat_proj = np.array([proj_distances[i, j] for i in range(n) for j in range(i+1, n)])

#             # Normalize both to [0, 1] for fair comparison
#             flat_orig = flat_orig / flat_orig.max()
#             flat_proj = flat_proj / flat_proj.max()

#             # Compute stress as mean squared error
#             stress = np.mean((flat_orig - flat_proj) ** 2)

#             return stress, flat_orig, flat_proj

#         # Extract names and cluster assignments
#         names = [d['Name'] for d in self.definitions]
#         clusters = [cluster_assignments[name] for name in names]

#         # Create color map if not provided
#         if color_map is None:
#             unique_clusters = sorted(set(clusters))
#             cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))
#             color_map = dict(zip(unique_clusters, cluster_colors))

#         # Create figure to compare transformation methods
#         n_methods = len(transformation_methods)
#         fig_width = 22
#         fig_height = 4 * ((n_methods + 1) // 2)  # Adjust height based on number of methods

#         fig, axes = plt.subplots(nrows=(n_methods + 1) // 2, ncols=2, 
#                                figsize=(fig_width, fig_height), dpi=150)
#         axes = axes.flatten()

#         # Test each transformation method
#         for i, (method_name, params) in enumerate(transformation_methods.items()):
#             print(f"Testing {method_name}...")

#             # Apply transformation
#             if not hasattr(self, 'transform_correlation_to_distance'):
#                 # Fallback if the method is not available
#                 distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
#                 distance_matrix = (distance_matrix + distance_matrix.T) / 2
#             else:
#                 distance_matrix = self.transform_correlation_to_distance(correlation_matrix, **params)

#             # Apply projection
#             projection = apply_projection(distance_matrix, method=projection_method)

#             # Compute stress metric
#             stress, flat_orig, flat_proj = compute_stress(distance_matrix, projection)

#             # Store results
#             results[method_name] = {
#                 "distance_matrix": distance_matrix,
#                 "projection": projection,
#                 "stress": stress
#             }

#             # Plot projection
#             ax = axes[i]

#             # Set panel styling
#             ax.set_facecolor('#f8f8f8')
#             ax.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)

#             for spine in ax.spines.values():
#                 spine.set(color='#cccccc', linewidth=1)

#             # Extract coordinates
#             x = projection[:, 0]
#             y = projection[:, 1]

#             # Add density visualization
#             try:
#                 # Calculate density estimation
#                 xy = np.vstack([x, y])
#                 kde = gaussian_kde(xy)
#                 z = kde(xy)

#                 # Create a fine grid for contour plotting
#                 x_min, x_max = x.min() - 0.1 * np.ptp(x), x.max() + 0.1 * np.ptp(x)
#                 y_min, y_max = y.min() - 0.1 * np.ptp(y), y.max() + 0.1 * np.ptp(y)
#                 xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
#                 positions = np.vstack([xi.ravel(), yi.ravel()])

#                 # Calculate density on the grid
#                 zi = kde(positions).reshape(xi.shape)

#                 # Plot density as filled contours with a light alpha
#                 contour = ax.contourf(xi, yi, zi, 15, cmap='viridis', alpha=0.15, zorder=1)

#                 # Add contour lines
#                 ax.contour(xi, yi, zi, 8, colors='black', linewidths=0.5, alpha=0.25, zorder=2)
#             except:
#                 # Skip density plot if there's an error (e.g., for very small datasets)
#                 pass

#             # Plot each point with its cluster color
#             marker_size = 90
#             edge_width = 1.0
#             alpha = 0.85

#             for j, (x_j, y_j) in enumerate(projection):
#                 cluster = clusters[j]
#                 color = color_map[cluster]
#                 ax.scatter(x_j, y_j, c=[color], s=marker_size, alpha=alpha, 
#                          edgecolors='black', linewidths=edge_width, zorder=10)

#                 # Add last names for a few selected points (avoid overcrowding)
#                 if n_components == 2 and j % 3 == 0:  # Show every 3rd name
#                     last_name = self.definitions[j]['Last']
#                     ax.text(x_j, y_j, last_name, fontsize=7, ha='center', va='bottom',
#                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), 
#                            zorder=20)

#             # Set title and labels
#             title = f"{method_name} (Stress: {stress:.4f})"
#             ax.set_title(title, fontsize=14, fontweight='bold')
#             ax.set_xlabel(f"Dimension 1", fontsize=10, fontweight='bold')
#             ax.set_ylabel(f"Dimension 2", fontsize=10, fontweight='bold')

#         # Hide any extra subplots if there are empty spots
#         for i in range(len(transformation_methods), len(axes)):
#             axes[i].axis('off')

#         # Add overall title
#         plt.suptitle(f"Comparison of Distance Transformation Methods ({projection_method.upper()})", 
#                     fontsize=18, fontweight='bold', y=0.98)

#         plt.tight_layout()
#         plt.subplots_adjust(top=0.95)

#         # Save figure if requested
#         if save_plot:
#             filename = f"distance_transform_comparison_{projection_method}.png"
#             filepath = os.path.join(self.output_dir, filename)
#             plt.savefig(filepath, dpi=200, bbox_inches='tight')
#             print(f"Comparison plot saved to {filepath}")

#         # Sort methods by stress (lower is better)
#         sorted_methods = sorted(results.items(), key=lambda x: x[1]['stress'])

#         print("\nRanking of transformation methods by stress (lower is better):")
#         for rank, (method_name, result) in enumerate(sorted_methods, 1):
#             print(f"  {rank}. {method_name}: {result['stress']:.4f}")

#         # Return the best method and results
#         best_method = sorted_methods[0][0]
#         print(f"\nBest method: {best_method}")

#         return results, best_method

#     def apply_best_distance_transformation(self, correlation_matrix, force_test=True, **kwargs):
#         # If specific parameters are provided and testing is not forced, use them directly
#         if kwargs and not force_test:
#             if not hasattr(self, 'transform_correlation_to_distance'):
#                 print("Warning: transform_correlation_to_distance method not found. Using standard transformation.")
#                 distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
#                 distance_matrix = (distance_matrix + distance_matrix.T) / 2
#             else:
#                 distance_matrix = self.transform_correlation_to_distance(correlation_matrix, **kwargs)
#             return distance_matrix

#         # Load correlation matrix if needed
#         if correlation_matrix is None:
#             corr_path = os.path.join(self.output_dir, f"m_correlation_{self.model}.npy")
#             if not os.path.exists(corr_path):
#                 raise ValueError("No correlation matrix found. Please provide one or run correlation analysis first.")
#             correlation_matrix = np.load(corr_path)

#         # Get cluster assignments if needed for testing
#         cluster_assignments = kwargs.get('cluster_assignments', None)
#         if cluster_assignments is None:
#             print("Testing requires cluster assignments. Generating from correlation matrix...")
#             _, _, _, cluster_assignments, _, _ = self.plot_clustered_correlation_heatmap(
#                 correlation_matrix, self.definitions, filename=None
#             )

#         # Run comparison of methods
#         results, best_method = self.compare_distance_transformations(
#             correlation_matrix, 
#             cluster_assignments, 
#             color_map=kwargs.get('color_map', None),
#             projection_method=kwargs.get('projection_method', 'mds')
#         )
#         print(f"CHOSEN TRANSFORMATION: {best_method}")

#         # Get parameters for the best method
#         if best_method == "standard":
#             best_params = {"method": "standard"}
#         elif best_method.startswith("power_"):
#             power = float(best_method.split("_")[1])
#             best_params = {"method": "power", "power": power}
#         elif best_method == "log":
#             best_params = {"method": "log"}
#         elif best_method == "sigmoid":
#             best_params = {"method": "sigmoid", "center": 0.7, "steepness": 12}
#         elif best_method == "adaptive":
#             best_params = {"method": "adaptive", "min_power": 0.1, "max_power": 0.5}
#         else:
#             best_params = {"method": "standard"}

#         # Apply the best transformation
#         if not hasattr(self, 'transform_correlation_to_distance'):
#             print("Warning: transform_correlation_to_distance method not found. Using standard transformation.")
#             distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
#             distance_matrix = (distance_matrix + distance_matrix.T) / 2
#         else:
#             print(f"Applying best transformation method: {best_method}")
#             distance_matrix = self.transform_correlation_to_distance(correlation_matrix, **best_params)

#         return distance_matrix

"""
Knowledge Graph Visualization Module

Provides comprehensive visualization capabilities for knowledge graphs including:
- Full graph visualization with color-coded clusters
- Individual cluster visualization
- Entity-specific relationship visualization
"""

import numpy as np
import networkx as nx
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("❌ matplotlib not available. Install with: pip install matplotlib")


class KnowledgeGraphVisualizer:
    """Handles visualization of knowledge graphs."""
    
    def __init__(self, knowledge_graph):
        """Initialize visualizer with a knowledge graph instance."""
        self.kg = knowledge_graph
        
    def visualize_graph(self, figsize: Tuple[int, int] = (15, 10), 
                       node_size_multiplier: float = 300, 
                       show_edge_labels: bool = True,
                       layout: str = 'spring',
                       save_to_file: bool = False) -> None:
        """
        Visualize the complete knowledge graph with color-coded clusters.
        
        Args:
            figsize: Figure size tuple (width, height)
            node_size_multiplier: Multiplier for node sizes based on frequency
            show_edge_labels: Whether to show relation types on edges
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai')
            save_to_file: If True, saves to "Visualized_Graph.png" instead of displaying
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib not available for visualization")
            return
            
        if len(self.kg.graph.nodes) == 0:
            print("❌ No entities in graph to visualize")
            return
        
        plt.figure(figsize=figsize)
        
        # Get cluster assignments
        try:
            entity_clusters = self.kg.kmeans_clustering()
            num_clusters = len(set(entity_clusters.values()))
        except:
            entity_clusters = {}
            num_clusters = 1
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.kg.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.kg.graph)
        elif layout == 'shell':
            pos = nx.shell_layout(self.kg.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.kg.graph)
        else:
            pos = nx.spring_layout(self.kg.graph)
        
        # Create color map for clusters
        colors = plt.cm.Set3(np.linspace(0, 1, max(num_clusters, 1)))
        
        # Draw nodes by cluster
        for cluster_id in range(num_clusters):
            cluster_nodes = [node for node, cluster in entity_clusters.items() if cluster == cluster_id]
            if cluster_nodes:
                node_sizes = [self.kg.entities[node].frequency * node_size_multiplier for node in cluster_nodes]
                nx.draw_networkx_nodes(self.kg.graph, pos, 
                                     nodelist=cluster_nodes,
                                     node_color=colors[cluster_id],
                                     node_size=node_sizes,
                                     alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.kg.graph, pos, alpha=0.5, edge_color='gray', width=1)
        
        # Add node labels
        nx.draw_networkx_labels(self.kg.graph, pos, font_size=8, font_weight='bold')
        
        # Add edge labels (relation types) if requested
        if show_edge_labels and len(self.kg.graph.edges) < 50:  # Avoid cluttering
            edge_labels = {}
            for u, v, data in self.kg.graph.edges(data=True):
                edge_labels[(u, v)] = data.get('relation_type', '')
            nx.draw_networkx_edge_labels(self.kg.graph, pos, edge_labels, font_size=6)
        
        plt.title(f'🕸️ Knowledge Graph\n{len(self.kg.graph.nodes)} entities, {len(self.kg.graph.edges)} relations, {num_clusters} clusters', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Add legend for clusters
        if num_clusters > 1:
            legend_elements = []
            cluster_descriptions = self.kg.get_cluster_descriptions()
            for cluster_id in range(num_clusters):
                cluster_info = cluster_descriptions.get(cluster_id, {})
                cluster_name = self.kg.get_semantic_cluster_name(cluster_id, cluster_info)
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=colors[cluster_id], 
                                                markersize=10, label=f'Cluster {cluster_id}: {cluster_name}'))
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        # Save or show based on parameter
        if save_to_file:
            filename = "Visualized_Graph.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Graph visualization saved to '{filename}'")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

    def visualize_clusters(self, figsize: Tuple[int, int] = (16, 12), 
                         cluster_layout: str = 'grid',
                         save_to_file: bool = False) -> None:
        """
        Visualize each cluster separately in subplots.
        
        Args:
            figsize: Figure size tuple (width, height)
            cluster_layout: Layout for subplots ('grid' or 'horizontal')
            save_to_file: If True, saves to "Visualized_Clusters.png" instead of displaying
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib not available for visualization")
            return
            
        try:
            entity_clusters = self.kg.kmeans_clustering()
            cluster_descriptions = self.kg.get_cluster_descriptions()
            num_clusters = len(set(entity_clusters.values()))
        except:
            print("❌ Could not generate clusters for visualization")
            return
        
        if num_clusters == 0:
            print("❌ No clusters found")
            return
        
        # Determine subplot layout
        if cluster_layout == 'grid':
            cols = min(2, num_clusters)
            rows = (num_clusters + cols - 1) // cols
        else:  # horizontal
            cols = num_clusters
            rows = 1
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_clusters == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))
        
        for cluster_id in range(num_clusters):
            ax = axes[cluster_id]
            
            # Get entities in this cluster
            cluster_entities = [entity for entity, cluster in entity_clusters.items() if cluster == cluster_id]
            
            if not cluster_entities:
                ax.text(0.5, 0.5, f'Cluster {cluster_id}\n(Empty)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue
            
            # Create subgraph for this cluster
            cluster_subgraph = self.kg.graph.subgraph(cluster_entities)
            
            if len(cluster_subgraph.nodes) == 0:
                ax.text(0.5, 0.5, f'Cluster {cluster_id}\n(No connections)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                continue
            
            # Layout for this cluster
            if len(cluster_subgraph.nodes) == 1:
                pos = {list(cluster_subgraph.nodes)[0]: (0.5, 0.5)}
            else:
                pos = nx.spring_layout(cluster_subgraph, k=1, iterations=50)
            
            # Node sizes based on frequency
            node_sizes = [self.kg.entities[node].frequency * 200 for node in cluster_subgraph.nodes]
            
            # Draw the cluster subgraph
            nx.draw_networkx_nodes(cluster_subgraph, pos, ax=ax,
                                 node_color=colors[cluster_id],
                                 node_size=node_sizes,
                                 alpha=0.8)
            
            nx.draw_networkx_edges(cluster_subgraph, pos, ax=ax,
                                 alpha=0.6, edge_color='gray', width=1)
            
            nx.draw_networkx_labels(cluster_subgraph, pos, ax=ax,
                                  font_size=8, font_weight='bold')
            
            # Cluster title with description
            cluster_info = cluster_descriptions.get(cluster_id, {})
            cluster_name = self.kg.get_semantic_cluster_name(cluster_id, cluster_info)
            top_entities = [e['name'] for e in cluster_info.get('entities', [])[:3]]
            
            title = f'Cluster {cluster_id}: {cluster_name}\n'
            title += f'Entities: {len(cluster_entities)} | Top: {", ".join(top_entities)}'
            
            ax.set_title(title, fontsize=10, fontweight='bold', pad=20)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(num_clusters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('🎯 Knowledge Graph Clusters', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save or show based on parameter
        if save_to_file:
            filename = "Visualized_Clusters.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Cluster visualization saved to '{filename}'")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

    def visualize_entity_relationships(self, entity: str, max_depth: int = 2,
                                     figsize: Tuple[int, int] = (12, 8),
                                     save_to_file: bool = False) -> None:
        """
        Visualize relationships for a specific entity.
        
        Args:
            entity: The central entity to visualize
            max_depth: Maximum relationship depth to show
            figsize: Figure size tuple (width, height)
            save_to_file: If True, saves to "Visualized_Entity_Relationships.png" instead of displaying
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib not available for visualization")
            return
            
        if entity not in self.kg.graph:
            print(f"❌ Entity '{entity}' not found in graph")
            print(f"Available entities: {list(self.kg.entities.keys())[:10]}...")
            return
        
        # Get subgraph within max_depth of the entity
        nodes = set([entity])
        for depth in range(max_depth):
            new_nodes = set()
            for node in nodes:
                if node in self.kg.graph:
                    new_nodes.update(self.kg.graph.neighbors(node))
            nodes.update(new_nodes)
        
        subgraph = self.kg.graph.subgraph(nodes)
        
        plt.figure(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Color nodes by distance from central entity
        node_colors = []
        for node in subgraph.nodes:
            if node == entity:
                node_colors.append('red')  # Central entity
            elif entity in self.kg.graph and node in self.kg.graph.neighbors(entity):
                node_colors.append('orange')  # Direct neighbors
            else:
                node_colors.append('lightblue')  # Others
        
        # Node sizes based on frequency
        node_sizes = [self.kg.entities[node].frequency * 300 for node in subgraph.nodes]
        
        # Draw graph
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                             node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold')
        
        # Add edge labels (relation types)
        if len(subgraph.edges) < 20:  # Avoid cluttering
            edge_labels = {}
            for u, v, data in subgraph.edges(data=True):
                edge_labels[(u, v)] = data.get('relation_type', '')
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=8)
        
        plt.title(f'🎯 Relationships for "{entity}"\n'
                 f'{len(subgraph.nodes)} entities, {len(subgraph.edges)} relations (depth {max_depth})',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label=f'Central: {entity}'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label='Direct connections'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Extended network')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save or show based on parameter
        if save_to_file:
            filename = "Visualized_Entity_Relationships.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Entity relationships visualization saved to '{filename}'")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

    def print_visualization_summary(self):
        """Print a summary of available visualization methods."""
        print("🎨 Knowledge Graph Visualization Options:")
        print("=" * 50)
        print("1. visualize_graph() - Full graph with color-coded clusters")
        print("2. visualize_clusters() - Individual cluster subplots")
        print("3. visualize_entity_relationships() - Focus on specific entity")
        print()
        print("Usage Examples:")
        print(">>> from graph_visualizer import KnowledgeGraphVisualizer")
        print(">>> viz = KnowledgeGraphVisualizer(knowledge_graph)")
        print(">>> viz.visualize_graph()")
        print(">>> viz.visualize_clusters()")
        print(">>> viz.visualize_entity_relationships('CEO')") 
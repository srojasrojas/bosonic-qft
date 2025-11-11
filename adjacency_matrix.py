"""
Generate adjacency matrices for all connected unlabeled graphs with N nodes.

This script generates all possible connected unlabeled graphs for a given number of nodes N
and returns their adjacency matrices. Two graphs are considered the same if they are 
isomorphic (same structure, different labeling).

For waveguide coupling applications, each adjacency matrix represents a different
coupling configuration between N waveguides.
"""

import numpy as np
from itertools import combinations, product
import networkx as nx
from typing import List, Set, Tuple
import argparse
import os


def generate_all_graphs(n: int) -> List[np.ndarray]:
    """
    Generate all possible undirected graphs with n nodes.
    
    Args:
        n: Number of nodes
        
    Returns:
        List of adjacency matrices (numpy arrays)
    """
    # Total number of possible edges in an undirected graph
    total_edges = n * (n - 1) // 2
    
    # Get all possible edge combinations
    possible_edges = list(combinations(range(n), 2))
    
    graphs = []
    
    # Generate all possible subsets of edges (2^total_edges possibilities)
    for num_edges in range(total_edges + 1):
        for edge_subset in combinations(possible_edges, num_edges):
            # Create adjacency matrix
            adj_matrix = np.zeros((n, n), dtype=int)
            
            for i, j in edge_subset:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Symmetric for undirected graph
            
            graphs.append(adj_matrix)
    
    return graphs


def is_connected(adj_matrix: np.ndarray) -> bool:
    """
    Check if a graph represented by adjacency matrix is connected.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        True if the graph is connected, False otherwise
    """
    n = adj_matrix.shape[0]
    if n <= 1:
        return True
    
    # BFS to check connectivity
    visited = [False] * n
    queue = [0]  # Start from node 0
    visited[0] = True
    visited_count = 1
    
    while queue:
        current = queue.pop(0)
        
        # Check all neighbors
        for neighbor in range(n):
            if adj_matrix[current, neighbor] == 1 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                visited_count += 1
    
    return visited_count == n


def get_canonical_form(adj_matrix: np.ndarray) -> Tuple[int, ...]:
    """
    Get a canonical form of the adjacency matrix for isomorphism checking.
    
    This uses a simple approach based on degree sequence and adjacency patterns.
    For more robust isomorphism checking, consider using graph libraries like NetworkX.
    
    Args:
        adj_matrix: Adjacency matrix
        
    Returns:
        Tuple representing canonical form
    """
    n = adj_matrix.shape[0]
    
    # Calculate degree sequence
    degrees = [sum(adj_matrix[i]) for i in range(n)]
    degrees.sort(reverse=True)
    
    # Create a more detailed signature
    # This is a simplified approach - for production use, consider graph isomorphism libraries
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] == 1:
                edge_list.append((min(degrees[i], degrees[j]), max(degrees[i], degrees[j])))
    
    edge_list.sort()
    
    return tuple(degrees + [len(edge_list)] + list(sum(edge_list, ())))


def are_isomorphic_networkx(adj1: np.ndarray, adj2: np.ndarray) -> bool:
    """
    Check if two graphs are isomorphic using NetworkX (more reliable).
    
    Args:
        adj1, adj2: Adjacency matrices to compare
        
    Returns:
        True if graphs are isomorphic, False otherwise
    """
    try:
        G1 = nx.from_numpy_array(adj1)
        G2 = nx.from_numpy_array(adj2)
        return nx.is_isomorphic(G1, G2)
    except:
        # Fallback to simple canonical form if NetworkX fails
        return get_canonical_form(adj1) == get_canonical_form(adj2)


def remove_isomorphic_duplicates(graphs: List[np.ndarray], use_networkx: bool = True) -> List[np.ndarray]:
    """
    Remove isomorphic duplicates from a list of graphs.
    
    Args:
        graphs: List of adjacency matrices
        use_networkx: Whether to use NetworkX for isomorphism checking
        
    Returns:
        List of unique (non-isomorphic) adjacency matrices
    """
    unique_graphs = []
    
    for graph in graphs:
        is_duplicate = False
        
        for unique_graph in unique_graphs:
            if use_networkx:
                if are_isomorphic_networkx(graph, unique_graph):
                    is_duplicate = True
                    break
            else:
                if get_canonical_form(graph) == get_canonical_form(unique_graph):
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_graphs.append(graph)
    
    return unique_graphs


def generate_connected_unlabeled_graphs(n: int, use_networkx: bool = True) -> List[np.ndarray]:
    """
    Generate all connected unlabeled graphs with n nodes.
    
    Args:
        n: Number of nodes
        use_networkx: Whether to use NetworkX for better isomorphism checking
        
    Returns:
        List of adjacency matrices representing all connected unlabeled graphs
    """
    print(f"Generating all graphs with {n} nodes...")
    all_graphs = generate_all_graphs(n)
    print(f"Total graphs generated: {len(all_graphs)}")
    
    print("Filtering connected graphs...")
    connected_graphs = [graph for graph in all_graphs if is_connected(graph)]
    print(f"Connected graphs found: {len(connected_graphs)}")
    
    print("Removing isomorphic duplicates...")
    unique_graphs = remove_isomorphic_duplicates(connected_graphs, use_networkx)
    print(f"Unique connected unlabeled graphs: {len(unique_graphs)}")
    
    return unique_graphs


def print_graph_info(adj_matrix: np.ndarray, index: int) -> None:
    """
    Print information about a graph.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        index: Index of the graph in the list
    """
    n = adj_matrix.shape[0]
    num_edges = np.sum(adj_matrix) // 2  # Divide by 2 because matrix is symmetric
    degrees = [sum(adj_matrix[i]) for i in range(n)]
    
    print(f"\nGraph {index + 1}:")
    print(f"  Nodes: {n}")
    print(f"  Edges: {num_edges}")
    print(f"  Degree sequence: {sorted(degrees, reverse=True)}")
    print("  Adjacency matrix:")
    for row in adj_matrix:
        print(f"    {row}")


def save_matrices_to_file(matrices: List[np.ndarray], filename: str) -> None:
    """
    Save adjacency matrices to a file.
    
    Args:
        matrices: List of adjacency matrices
        filename: Name of the output file
    """
    with open(filename, 'w') as f:
        f.write(f"# Connected unlabeled graphs with {matrices[0].shape[0]} nodes\n")
        f.write(f"# Total number of unique graphs: {len(matrices)}\n\n")
        
        for i, matrix in enumerate(matrices):
            f.write(f"# Graph {i + 1}\n")
            for row in matrix:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("\n")


def main():
    """
    Main function with command line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Generate adjacency matrices for all connected unlabeled graphs with N nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python adjacency_matrix.py 4
  python adjacency_matrix.py 5 --verbose
  python adjacency_matrix.py 3 --output my_graphs.txt
  
Known counts:
  N=1: 1 graph    N=2: 1 graph    N=3: 2 graphs
  N=4: 6 graphs   N=5: 21 graphs  N=6: 112 graphs
        """
    )
    
    parser.add_argument(
        'N', 
        type=int, 
        help='Number of nodes in the graphs'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output filename (default: connected_graphs_nN.txt)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information for each graph'
    )
    
    parser.add_argument(
        '--no-networkx',
        action='store_true',
        help='Use simple isomorphism check instead of NetworkX (faster but less reliable)'
    )
    
    args = parser.parse_args()
    
    N = args.N
    
    # Validate input
    if N < 1:
        print("Error: N must be a positive integer")
        return
    
    if N > 8:
        response = input(f"Warning: N={N} will generate many graphs and may take a long time. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    print(f"Generating connected unlabeled graphs for N = {N}")
    print("=" * 50)
    
    # Show current working directory
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")
    
    # Generate the graphs
    use_networkx = not args.no_networkx
    if not use_networkx:
        print("Warning: Using simple isomorphism check (may be less accurate)")
    
    connected_graphs = generate_connected_unlabeled_graphs(N, use_networkx=use_networkx)
    
    # Display results
    print(f"\nFound {len(connected_graphs)} connected unlabeled graphs with {N} nodes:")
    print("=" * 50)
    
    if args.verbose:
        for i, graph in enumerate(connected_graphs):
            print_graph_info(graph, i)
    else:
        print("Use --verbose to see detailed information for each graph")
    
    # Determine output filename
    if args.output:
        filename = args.output
    else:
        filename = f"connected_graphs_n{N}.txt"
    
    # Get full path for output file
    full_path = os.path.abspath(filename)
    
    # Save to file
    save_matrices_to_file(connected_graphs, filename)
    print(f"\nAdjacency matrices saved to:")
    print(f"  File: {filename}")
    print(f"  Full path: {full_path}")
    
    # Known values for verification
    known_counts = {1: 1, 2: 1, 3: 2, 4: 6, 5: 21, 6: 112, 7: 853, 8: 11117}
    if N in known_counts:
        expected = known_counts[N]
        if len(connected_graphs) == expected:
            print(f"✓ Verification passed: Found {len(connected_graphs)} graphs (expected {expected})")
        else:
            print(f"✗ Verification failed: Found {len(connected_graphs)} graphs (expected {expected})")
    else:
        print(f"Note: No known count available for N={N} for verification")


if __name__ == "__main__":
    main()
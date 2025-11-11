#!/usr/bin/env python3
"""
QFT Matrix Generator

This script takes a JSON file containing lambda_k solutions from equations (27) and (28)
and generates all associated matrices according to the construction procedure:

1. Calculate U^(N) matrix using equation (25)
2. Calculate coefficients u_{m,n}^(N)
3. Construct phase-shift matrices Phi^in and Phi^out
4. Calculate final QFT matrix D^(N) = Phi^out @ U^(N) @ Phi^in

The script saves all intermediate and final matrices to a JSON file for analysis.

Usage:
    python qft_matrix_generator.py input_file.json --output matrices_output.json [options]
    
Examples:
    python qft_matrix_generator.py solutions/eq28_n4.json --output matrices_n4.json
    python qft_matrix_generator.py solutions/eq27_n7.json --output matrices_n7.json --equation 27
"""

import argparse
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys


class QFTMatrixGenerator:
    """Generator for QFT matrices from lambda_k solutions."""
    
    def __init__(self, N: int, lambda_k: np.ndarray):
        """
        Initialize QFT matrix generator.
        
        Args:
            N: Size of the QFT
            lambda_k: Array of lambda values (eigenvalues)
        """
        self.N = N
        self.lambda_k = np.array(lambda_k, dtype=float)
        
        # Validate input
        if len(self.lambda_k) != N:
            raise ValueError(f"Expected {N} lambda values, got {len(self.lambda_k)}")
    
    def calculate_U_matrix(self) -> np.ndarray:
        """
        Calculate U^(N) matrix using equation (25).
        
        U_{m,n}^{(N)} = (1/N) * sum_{k=0}^{N-1} exp(2πik(n-m)/N - iλ_k)
        
        Returns:
            U^(N) matrix (N x N complex matrix)
        """
        U = np.zeros((self.N, self.N), dtype=complex)
        
        for m in range(self.N):
            for n in range(self.N):
                total = 0.0 + 0.0j
                for k in range(self.N):
                    phase = 2 * np.pi * k * (n - m) / self.N - 1j * self.lambda_k[k]
                    total += np.exp(phase)
                U[m, n] = total / self.N
        
        return U
    
    def calculate_u_coefficients(self, U: np.ndarray) -> np.ndarray:
        """
        Calculate u_{m,n}^(N) coefficients.
        
        u_{m,n}^(N) = √N * U_{m,n}^(N)
        
        Args:
            U: U^(N) matrix
            
        Returns:
            u coefficients matrix
        """
        return np.sqrt(self.N) * U
    
    def calculate_phase_shift_matrices(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate phase-shift matrices Phi^in and Phi^out.
        
        Phi^in = diag(u_{0,0}^*, u_{0,1}^*, ..., u_{0,N-1}^*)
        Phi^out = diag(1, u_{1,0}^*, u_{2,0}^*, ..., u_{N-1,0}^*)
        
        Args:
            u: u coefficients matrix
            
        Returns:
            Tuple of (Phi_in, Phi_out) matrices
        """
        # Phi^in: diagonal matrix with conjugates of first row
        Phi_in = np.diag([np.conj(u[0, j]) for j in range(self.N)])
        
        # Phi^out: diagonal matrix with 1 and conjugates of first column (except first element)
        Phi_out_diag = [1.0 + 0.0j] + [np.conj(u[i, 0]) for i in range(1, self.N)]
        Phi_out = np.diag(Phi_out_diag)
        
        return Phi_in, Phi_out
    
    def calculate_qft_matrix(self, U: np.ndarray, Phi_in: np.ndarray, Phi_out: np.ndarray) -> np.ndarray:
        """
        Calculate final QFT matrix D^(N).
        
        D^(N) = Phi^out @ U^(N) @ Phi^in
        
        Args:
            U: U^(N) matrix
            Phi_in: Input phase-shift matrix
            Phi_out: Output phase-shift matrix
            
        Returns:
            QFT matrix D^(N)
        """
        return Phi_out @ U @ Phi_in
    
    def calculate_standard_qft(self) -> np.ndarray:
        """
        Calculate standard QFT matrix for comparison.
        
        U^(N) = (1/√N) * [ω^{mn}] where ω = e^{-2πi/N}
        
        Returns:
            Standard QFT matrix
        """
        omega = np.exp(-2j * np.pi / self.N)
        qft_standard = np.zeros((self.N, self.N), dtype=complex)
        
        for m in range(self.N):
            for n in range(self.N):
                qft_standard[m, n] = omega**(m * n) / np.sqrt(self.N)
        
        return qft_standard
    
    def verify_unitarity(self, matrix: np.ndarray, tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Verify if a matrix is unitary.
        
        Args:
            matrix: Matrix to verify
            tolerance: Numerical tolerance
            
        Returns:
            Dictionary with verification results
        """
        # Calculate U @ U†
        product = matrix @ np.conj(matrix.T)
        identity = np.eye(self.N)
        
        # Calculate maximum deviation from identity
        deviation = np.max(np.abs(product - identity))
        is_unitary = deviation < tolerance
        
        return {
            'is_unitary': bool(is_unitary),
            'max_deviation': float(deviation),
            'tolerance': float(tolerance)
        }
    
    def verify_qft_correctness(self, D: np.ndarray, standard_qft: np.ndarray, 
                              tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Verify if the generated QFT matrix matches the standard QFT.
        
        Args:
            D: Generated QFT matrix
            standard_qft: Standard QFT matrix
            tolerance: Numerical tolerance
            
        Returns:
            Dictionary with verification results
        """
        # Calculate maximum difference
        difference = np.max(np.abs(D - standard_qft))
        is_correct = difference < tolerance
        
        # Calculate relative error
        relative_error = difference / np.max(np.abs(standard_qft))
        
        return {
            'is_correct_qft': bool(is_correct),
            'max_difference': float(difference),
            'relative_error': float(relative_error),
            'tolerance': float(tolerance)
        }
    
    def generate_all_matrices(self, tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Generate all matrices and verification data.
        
        Args:
            tolerance: Numerical tolerance for verifications
            
        Returns:
            Dictionary containing all matrices and verification results
        """
        print(f"Generating QFT matrices for N = {self.N}")
        print(f"Lambda values: {self.lambda_k}")
        
        # Step 1: Calculate U^(N) matrix
        print("Step 1: Calculating U^(N) matrix...")
        U = self.calculate_U_matrix()
        
        # Step 2: Calculate u coefficients
        print("Step 2: Calculating u coefficients...")
        u = self.calculate_u_coefficients(U)
        
        # Step 3: Calculate phase-shift matrices
        print("Step 3: Calculating phase-shift matrices...")
        Phi_in, Phi_out = self.calculate_phase_shift_matrices(u)
        
        # Step 4: Calculate final QFT matrix
        print("Step 4: Calculating final QFT matrix...")
        D = self.calculate_qft_matrix(U, Phi_in, Phi_out)
        
        # Calculate standard QFT for comparison
        print("Calculating standard QFT for comparison...")
        standard_qft = self.calculate_standard_qft()
        
        # Perform verifications
        print("Performing verifications...")
        U_unitarity = self.verify_unitarity(U, tolerance)
        D_unitarity = self.verify_unitarity(D, tolerance)
        qft_correctness = self.verify_qft_correctness(D, standard_qft, tolerance)
        
        print(f"U matrix is unitary: {U_unitarity['is_unitary']}")
        print(f"D matrix is unitary: {D_unitarity['is_unitary']}")
        print(f"D matrix matches standard QFT: {qft_correctness['is_correct_qft']}")
        
        if qft_correctness['is_correct_qft']:
            print(f"✓ SUCCESS: Generated QFT matrix is correct!")
        else:
            print(f"✗ WARNING: Generated QFT matrix differs from standard QFT")
            print(f"  Max difference: {qft_correctness['max_difference']:.2e}")
        
        return {
            'N': int(self.N),
            'lambda_k': self.lambda_k.tolist(),
            'matrices': {
                'U_matrix': U.tolist(),
                'u_coefficients': u.tolist(),
                'Phi_in': Phi_in.tolist(),
                'Phi_out': Phi_out.tolist(),
                'D_qft_matrix': D.tolist(),
                'standard_qft': standard_qft.tolist()
            },
            'verification': {
                'U_unitarity': U_unitarity,
                'D_unitarity': D_unitarity,
                'qft_correctness': qft_correctness
            },
            'matrix_properties': {
                'U_magnitude_range': {
                    'min': float(np.min(np.abs(U))),
                    'max': float(np.max(np.abs(U))),
                    'expected': float(1.0 / np.sqrt(self.N))
                },
                'D_magnitude_range': {
                    'min': float(np.min(np.abs(D))),
                    'max': float(np.max(np.abs(D))),
                    'expected': float(1.0 / np.sqrt(self.N))
                }
            }
        }


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.complexfloating):
        return complex(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def load_lambda_values(filename: str, equation: Optional[int] = None) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """
    Load lambda values from JSON file.
    
    Args:
        filename: Path to JSON file
        equation: Which equation format to expect (27 or 28), auto-detect if None
        
    Returns:
        Tuple of (N, lambda_k array, metadata dict)
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Auto-detect equation type if not specified
    if equation is None:
        if 'equation' in data:
            equation = data['equation']
        elif 'results' in data:
            equation = 27  # Multi-method format typical of equation 27
        else:
            equation = 28  # Single result format typical of equation 28
    
    print(f"Loading lambda values for equation {equation}")
    
    # Extract lambda values based on format
    lambda_k = None
    N = None
    metadata = {}
    
    if equation == 27:
        # Handle equation 27 format (possibly multi-method or multi-run)
        if 'results' in data:
            # Multi-method format
            methods = data['results']
            if len(methods) == 1:
                method_name = list(methods.keys())[0]
                result = methods[method_name]
                print(f"Using method: {method_name}")
            else:
                print(f"Multiple methods available: {list(methods.keys())}")
                method_name = input("Select method: ")
                result = methods[method_name]
        else:
            # Single result format
            result = data
        
        # Check if it's a multi-run result
        if 'all_runs' in result:
            runs = result['all_runs']
            valid_runs = [r for r in runs if 'lambda_full' in r and r.get('verification', {}).get('is_valid', False)]
            
            if len(valid_runs) == 0:
                raise ValueError("No valid solutions found in multi-run results")
            elif len(valid_runs) == 1:
                print(f"Using single valid solution from run {valid_runs[0]['run_number']}")
                selected_run = valid_runs[0]
            else:
                print(f"Found {len(valid_runs)} valid solutions:")
                for i, run in enumerate(valid_runs):
                    obj_val = run.get('objective_value', 'N/A')
                    max_res = run.get('verification', {}).get('max_residual', 'N/A')
                    print(f"  {i}: Run {run['run_number']}, obj={obj_val:.2e}, max_res={max_res:.2e}")
                
                choice = input(f"Select solution (0-{len(valid_runs)-1}, or 'best' for lowest objective): ")
                if choice.lower() == 'best':
                    selected_run = min(valid_runs, key=lambda x: x.get('objective_value', float('inf')))
                    print(f"Selected best solution: Run {selected_run['run_number']}")
                else:
                    selected_run = valid_runs[int(choice)]
            
            lambda_k = np.array(selected_run['lambda_full'])
            metadata.update({
                'source_run': selected_run['run_number'],
                'objective_value': selected_run.get('objective_value'),
                'verification': selected_run.get('verification'),
                'total_valid_runs': len(valid_runs)
            })
        else:
            # Single run result
            lambda_k = np.array(result['lambda_full'])
            metadata.update({
                'objective_value': result.get('objective_value'),
                'verification': result.get('verification')
            })
        
        N = len(lambda_k)
        metadata.update({
            'source_equation': 27,
            'method': result.get('method', 'unknown')
        })
    
    elif equation == 28:
        # Handle equation 28 format
        if 'lambda_full' in data:
            # New flat format
            lambda_k = np.array(data['lambda_full'])
            N = data.get('N', len(lambda_k))
            metadata.update({
                'source_equation': 28,
                'method': data.get('method'),
                'objective_value': data.get('objective_value'),
                'verification': data.get('verification'),
                'eq27_file': data.get('eq27_file'),
                'eq27_solutions_used': data.get('eq27_solutions_used')
            })
        elif 'result' in data and 'lambda_full' in data['result']:
            # Old nested format
            result = data['result']
            lambda_k = np.array(result['lambda_full'])
            N = data.get('N', len(lambda_k))
            metadata.update({
                'source_equation': 28,
                'method': result.get('method'),
                'objective_value': result.get('objective_value'),
                'verification': result.get('verification'),
                'eq27_file': data.get('eq27_file'),
                'eq27_solutions_used': data.get('eq27_solutions_used')
            })
        else:
            raise ValueError("Cannot find lambda_full in equation 28 format")
    
    else:
        raise ValueError(f"Unsupported equation type: {equation}")
    
    if lambda_k is None:
        raise ValueError("Failed to extract lambda values from JSON file")
    
    print(f"Loaded {len(lambda_k)} lambda values for N = {N}")
    print(f"Lambda range: [{np.min(lambda_k):.6f}, {np.max(lambda_k):.6f}]")
    
    return N, lambda_k, metadata


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate QFT matrices from lambda_k solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate matrices from equation 28 solution
  python qft_matrix_generator.py solutions/eq28_n4.json --output matrices_n4.json
  
  # Generate matrices from equation 27 solution  
  python qft_matrix_generator.py solutions/eq27_n7.json --output matrices_n7.json --equation 27
  
  # Use custom tolerance for verification
  python qft_matrix_generator.py solutions/eq28_n4.json --output matrices_n4.json --tolerance 1e-12

The output JSON file contains:
  - All intermediate matrices (U, u_coefficients, Phi_in, Phi_out)
  - Final QFT matrix (D_qft_matrix)
  - Standard QFT matrix for comparison
  - Verification results (unitarity, correctness)
  - Matrix properties and metadata
        """
    )
    
    parser.add_argument('input_file', help='JSON file containing lambda_k solutions')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output JSON file for matrices')
    parser.add_argument('--equation', '-e', type=int, choices=[27, 28],
                       help='Equation type (27 or 28). Auto-detected if not specified.')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-10,
                       help='Tolerance for numerical verifications (default: 1e-10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Load lambda values
        N, lambda_k, metadata = load_lambda_values(args.input_file, args.equation)
        
        # Create generator
        generator = QFTMatrixGenerator(N, lambda_k)
        
        # Generate matrices
        results = generator.generate_all_matrices(tolerance=args.tolerance)
        
        # Add metadata
        results['metadata'] = metadata
        results['source_file'] = args.input_file
        results['generation_parameters'] = {
            'tolerance': args.tolerance,
            'equation_type': metadata.get('source_equation', 'unknown')
        }
        
        # Convert numpy types for JSON serialization
        results = convert_numpy_types(results)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nMatrices saved to: {args.output}")
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"  Input file: {args.input_file}")
        print(f"  Output file: {args.output}")
        print(f"  Matrix size: {N} × {N}")
        print(f"  Source equation: {metadata.get('source_equation', 'unknown')}")
        if 'objective_value' in metadata and metadata['objective_value'] is not None:
            print(f"  Source objective value: {metadata['objective_value']:.2e}")
        if 'verification' in metadata and metadata['verification']:
            is_valid = metadata['verification'].get('is_valid', False)
            max_res = metadata['verification'].get('max_residual', 'N/A')
            print(f"  Source solution valid: {is_valid}")
            if max_res != 'N/A':
                print(f"  Source max residual: {max_res:.2e}")
        
        print(f"  Generated QFT correctness: {results['verification']['qft_correctness']['is_correct_qft']}")
        if not results['verification']['qft_correctness']['is_correct_qft']:
            diff = results['verification']['qft_correctness']['max_difference']
            print(f"  Max difference from standard QFT: {diff:.2e}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
QFT Characterization - Equation 27 Solver
==========================================

Solves the complex Hadamard matrix condition (Equation 27) for QFT phase parameters λ_k.

Equation 27 (Corrected):
For each d = 0, 1, ..., N-1:
0 = Σ_{j=1}^{N-1} Σ_{k=j}^{N-1} cos(2π j d/N + λ_k - λ_{k-j})

This ensures orthogonality and complex Hadamard matrix structure.
Uses symmetry constraint: λ_k = λ_{N-k} to reduce degrees of freedom.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import argparse
import time


class QFTEquation27Solver:
    """Solver for QFT Equation 27 - Complex Hadamard matrix condition."""
    
    def __init__(self, N: int):
        """
        Initialize solver for dimension N.
        
        Args:
            N: QFT dimension
        """
        self.N = N
        # Correct number of independent parameters: M = floor(N/2) + 1
        self.num_params = N // 2 + 1
        self.solution_history = []
        
    def expand_lambda_vector(self, lambda_params: np.ndarray) -> np.ndarray:
        """
        Expand parameter vector using symmetry λ_k = λ_{N-k}.
        
        Args:
            lambda_params: Independent parameters [λ_0, λ_1, ..., λ_M] where M = floor(N/2)
            
        Returns:
            Full λ vector with symmetry constraint applied
        """
        lambda_full = np.zeros(self.N)
        
        # Set λ_0
        lambda_full[0] = lambda_params[0]
        
        # Set λ_k for k = 1, ..., floor(N/2) and apply symmetry λ_{N-k} = λ_k
        for k in range(1, len(lambda_params)):
            if k < self.N:
                lambda_full[k] = lambda_params[k]
                # Apply symmetry: λ_{N-k} = λ_k
                if self.N - k != k:  # Avoid overwriting middle element for odd N
                    lambda_full[self.N - k] = lambda_params[k]
        
        return lambda_full
    
    def equation_27_residual(self, lambda_params: np.ndarray, d: int) -> float:
        """
        Compute residual of Equation 27 for specific difference d = m-n.
        
        Equation 27 (corrected): For each d = 0, 1, ..., N-1:
        0 = Σ_{j=1}^{N-1} Σ_{k=j}^{N-1} cos(2π j d/N + λ_k - λ_{k-j})
        
        Args:
            lambda_params: Parameter vector (independent parameters)
            d: Difference d = m-n (mod N)
            
        Returns:
            Residual value for this specific d
        """
        lambda_full = self.expand_lambda_vector(lambda_params)
        
        total_sum = 0.0
        for j in range(1, self.N):
            for k in range(j, self.N):
                k_minus_j = k - j
                phase = (2 * np.pi * j * d / self.N + 
                        lambda_full[k] - lambda_full[k_minus_j])
                total_sum += np.cos(phase)
        
        return total_sum
    
    def equation_27_gradient(self, lambda_params: np.ndarray, d: int) -> np.ndarray:
        """
        Analytical gradient of Equation 27 residual w.r.t. lambda_params.
        
        Args:
            lambda_params: Parameter vector (independent parameters)
            d: Difference d = m-n (mod N)
            
        Returns:
            Gradient vector w.r.t. independent parameters
        """
        lambda_full = self.expand_lambda_vector(lambda_params)
        gradient = np.zeros_like(lambda_params)
        
        for i in range(len(lambda_params)):
            grad_i = 0.0
            
            # Direct contribution from λ_i
            for j in range(1, self.N):
                for k in range(j, self.N):
                    k_minus_j = k - j
                    phase = (2 * np.pi * j * d / self.N + 
                            lambda_full[k] - lambda_full[k_minus_j])
                    
                    # Contribution from λ_i as λ_k
                    if k == i:
                        grad_i -= np.sin(phase)
                    
                    # Contribution from λ_i as λ_{k-j} (negative term)
                    if k_minus_j == i:
                        grad_i += np.sin(phase)
            
            # Account for symmetry constraint λ_{N-i} = λ_i (if i > 0 and N-i != i)
            if i > 0 and self.N - i != i and self.N - i < self.N:
                symmetric_idx = self.N - i
                
                for j in range(1, self.N):
                    for k in range(j, self.N):
                        k_minus_j = k - j
                        phase = (2 * np.pi * j * d / self.N + 
                                lambda_full[k] - lambda_full[k_minus_j])
                        
                        # Symmetric contributions
                        if k == symmetric_idx:
                            grad_i -= np.sin(phase)
                        
                        if k_minus_j == symmetric_idx:
                            grad_i += np.sin(phase)
            
            gradient[i] = grad_i
        
        return gradient
    
    def objective_function(self, lambda_params: np.ndarray) -> float:
        """
        Total objective function: sum of squared residuals over all d values.
        
        Equation 27 gives N equations, one for each d = 0, 1, ..., N-1.
        
        Args:
            lambda_params: Parameter vector (independent parameters)
            
        Returns:
            Sum of squared residuals: S(λ) = Σ_{d=0}^{N-1} [F_d(λ)]²
        """
        total_error = 0.0
        
        for d in range(self.N):
            residual = self.equation_27_residual(lambda_params, d)
            total_error += residual**2
        
        return total_error
    
    def objective_gradient(self, lambda_params: np.ndarray) -> np.ndarray:
        """
        Gradient of total objective function.
        
        Args:
            lambda_params: Parameter vector (independent parameters)
            
        Returns:
            Gradient vector: ∇S(λ) = 2 Σ_{d=0}^{N-1} F_d(λ) ∇F_d(λ)
        """
        total_gradient = np.zeros_like(lambda_params)
        
        for d in range(self.N):
            residual = self.equation_27_residual(lambda_params, d)
            grad = self.equation_27_gradient(lambda_params, d)
            total_gradient += 2 * residual * grad
        
        return total_gradient
    
    def solve_bfgs(self, initial_guess: Optional[np.ndarray] = None, 
                   max_iterations: int = 1000) -> Dict:
        """
        Solve using BFGS with analytical gradient.
        
        Args:
            initial_guess: Initial parameter values (default: QFT-inspired)
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization result dictionary
        """
        if initial_guess is None:
            initial_guess = self.get_qft_inspired_initialization()
        
        start_time = time.time()
        
        result = minimize(
            fun=self.objective_function,
            x0=initial_guess,
            method='BFGS',
            jac=self.objective_gradient,
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        solve_time = time.time() - start_time
        
        # Expand solution to full vector
        lambda_full = self.expand_lambda_vector(result.x)
        
        return {
            'method': 'BFGS',
            'lambda_params': result.x,
            'lambda_full': lambda_full,
            'objective_value': result.fun,
            'success': result.success,
            'iterations': result.nit,
            'solve_time': solve_time,
            'message': result.message,
            'initial_guess_type': 'QFT-inspired'
        }
    
    def solve_lbfgs(self, initial_guess: Optional[np.ndarray] = None,
                    bounds: Optional[List[Tuple]] = None,
                    max_iterations: int = 1000) -> Dict:
        """
        Solve using L-BFGS-B (bounded version).
        
        Args:
            initial_guess: Initial parameter values
            bounds: Parameter bounds [(min, max), ...] (default: [0, 2π])
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization result dictionary
        """
        if initial_guess is None:
            initial_guess = self.get_linear_initialization()
        
        if bounds is None:
            bounds = [(0, 2*np.pi)] * self.num_params
        
        start_time = time.time()
        
        result = minimize(
            fun=self.objective_function,
            x0=initial_guess,
            method='L-BFGS-B',
            jac=self.objective_gradient,
            bounds=bounds,
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        solve_time = time.time() - start_time
        lambda_full = self.expand_lambda_vector(result.x)
        
        return {
            'method': 'L-BFGS-B',
            'lambda_params': result.x,
            'lambda_full': lambda_full,
            'objective_value': result.fun,
            'success': result.success,
            'iterations': result.nit,
            'solve_time': solve_time,
            'message': result.message,
            'initial_guess_type': 'Linear'
        }
    
    def solve_nelder_mead(self, initial_guess: Optional[np.ndarray] = None,
                         max_iterations: int = 2000) -> Dict:
        """
        Solve using Nelder-Mead (gradient-free).
        
        Args:
            initial_guess: Initial parameter values
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization result dictionary
        """
        if initial_guess is None:
            initial_guess = self.get_random_initialization(scale=np.pi)
        
        start_time = time.time()
        
        result = minimize(
            fun=self.objective_function,
            x0=initial_guess,
            method='Nelder-Mead',
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        solve_time = time.time() - start_time
        lambda_full = self.expand_lambda_vector(result.x)
        
        return {
            'method': 'Nelder-Mead',
            'lambda_params': result.x,
            'lambda_full': lambda_full,
            'objective_value': result.fun,
            'success': result.success,
            'iterations': result.nit,
            'solve_time': solve_time,
            'message': result.message,
            'initial_guess_type': 'Random'
        }
    
    def solve_differential_evolution(self, bounds_multiplier: float = 1.0,
                                   population_size: int = 15,
                                   max_iterations: int = 1000,
                                   seed: int = 42) -> Dict:
        """
        Solve using Differential Evolution (global optimization).
        
        Args:
            bounds_multiplier: Bounds as [0, bounds_multiplier*2π]
            population_size: DE population size
            max_iterations: Maximum generations
            
        Returns:
            Optimization result dictionary
        """
        bounds = [(0, bounds_multiplier*2*np.pi)] * self.num_params
        
        start_time = time.time()
        
        result = differential_evolution(
            func=self.objective_function,
            bounds=bounds,
            popsize=population_size,
            maxiter=max_iterations,
            disp=True,
            seed=seed
        )
        
        solve_time = time.time() - start_time
        lambda_full = self.expand_lambda_vector(result.x)
        
        return {
            'method': 'Differential Evolution',
            'lambda_params': result.x,
            'lambda_full': lambda_full,
            'objective_value': result.fun,
            'success': result.success,
            'iterations': result.nit,
            'solve_time': solve_time,
            'message': result.message,
            'initial_guess_type': 'Global search',
            'seed_used': seed
        }
    
    def get_qft_inspired_initialization(self) -> np.ndarray:
        """
        Get QFT-inspired initial parameters.
        
        Based on patterns like λ_k ∝ 4πk/5 for N=5, we use λ_k = 2πk²/N.
        
        Returns:
            Initial parameter vector (independent parameters only)
        """
        lambda_full = np.zeros(self.N)
        
        # QFT-inspired pattern: λ_k = 2πk²/N (quadratic phase pattern)
        for k in range(self.N):
            lambda_full[k] = 2 * np.pi * k * k / self.N
        
        # Extract independent parameters
        lambda_params = np.zeros(self.num_params)
        lambda_params[0] = lambda_full[0]  # λ_0
        
        for i in range(1, min(self.num_params, self.N)):
            lambda_params[i] = lambda_full[i]
        
        return lambda_params
    
    def get_linear_initialization(self) -> np.ndarray:
        """
        Get linear phase initialization.
        
        Pattern: λ_k = 4πk/N (similar to N=5 example)
        
        Returns:
            Initial parameter vector (independent parameters only)
        """
        lambda_full = np.zeros(self.N)
        
        # Linear pattern: λ_k = 4πk/N
        for k in range(self.N):
            lambda_full[k] = 4 * np.pi * k / self.N
        
        # Extract independent parameters
        lambda_params = np.zeros(self.num_params)
        lambda_params[0] = lambda_full[0]  # λ_0 = 0
        
        for i in range(1, min(self.num_params, self.N)):
            lambda_params[i] = lambda_full[i]
        
        return lambda_params
    
    def get_random_initialization(self, scale: float = 2*np.pi) -> np.ndarray:
        """
        Get random initialization in appropriate range.
        
        Args:
            scale: Random values in [0, scale]
            
        Returns:
            Random initial parameter vector
        """
        return np.random.uniform(0, scale, self.num_params)
    
    def solve_differential_evolution_multirun(self, n_runs: int = 10,
                                            bounds_multiplier: float = 1.0,
                                            population_size: int = 15,
                                            max_iterations: int = 1000,
                                            base_seed: int = 42) -> List[Dict]:
        """
        Run Differential Evolution multiple times with different seeds.
        
        Args:
            n_runs: Number of runs with different seeds
            bounds_multiplier: Parameter bounds multiplier
            population_size: DE population size
            max_iterations: Maximum iterations per run
            base_seed: Base seed (will generate n_runs different seeds from this)
            
        Returns:
            List of results from each run
        """
        results = []
        unique_solutions = []
        
        print(f"Running Differential Evolution {n_runs} times with different seeds...")
        
        for i in range(n_runs):
            # Generate different seed for each run
            run_seed = base_seed + i * 1000  # Spread seeds apart
            
            print(f"\nRun {i+1}/{n_runs} (seed = {run_seed})...")
            
            try:
                result = self.solve_differential_evolution(
                    bounds_multiplier=bounds_multiplier,
                    population_size=population_size,
                    max_iterations=max_iterations,
                    seed=run_seed
                )
                
                result['run_number'] = i + 1
                result['seed_used'] = run_seed
                
                # Check if this is a truly new solution (not just numerical noise)
                is_new_solution = True
                for prev_solution in unique_solutions:
                    # Check if solutions are essentially the same (within tolerance)
                    diff = np.max(np.abs(np.array(result['lambda_full']) - 
                                        np.array(prev_solution['lambda_full'])))
                    if diff < 1e-6:  # Same solution within numerical precision
                        is_new_solution = False
                        result['duplicate_of_run'] = prev_solution['run_number']
                        break
                
                if is_new_solution:
                    unique_solutions.append(result)
                    print(f"  → New unique solution found!")
                else:
                    print(f"  → Duplicate of run {result['duplicate_of_run']}")
                
                results.append(result)
                
            except Exception as e:
                print(f"  → Error in run {i+1}: {e}")
                error_result = {
                    'run_number': i + 1,
                    'seed_used': run_seed,
                    'error': str(e)
                }
                results.append(error_result)
        
        print(f"\nMulti-run completed:")
        print(f"  Total runs: {n_runs}")
        print(f"  Successful runs: {len([r for r in results if 'error' not in r])}")
        print(f"  Unique solutions: {len(unique_solutions)}")
        
        return results
    
    def get_standard_qft_phases(self) -> np.ndarray:
        """
        Get phase parameters for standard QFT (reference solution).
        
        Returns:
            Standard QFT lambda parameters (full vector)
        """
        # For standard QFT, all λ_k = 0
        return np.zeros(self.N)
    
    def verify_solution(self, lambda_full: np.ndarray, tolerance: float = 1e-10) -> Dict:
        """
        Verify if solution satisfies Equation 27 for all d values.
        
        Args:
            lambda_full: Full parameter vector
            tolerance: Numerical tolerance for verification
            
        Returns:
            Verification results
        """
        max_residual = 0.0
        residuals = []
        
        # Extract independent parameters from full vector
        lambda_params = np.zeros(self.num_params)
        lambda_params[0] = lambda_full[0]  # λ_0
        for i in range(1, min(self.num_params, len(lambda_full))):
            lambda_params[i] = lambda_full[i]
        
        # Check residual for each d = 0, 1, ..., N-1
        for d in range(self.N):
            residual = abs(self.equation_27_residual(lambda_params, d))
            residuals.append(residual)
            max_residual = max(max_residual, residual)
        
        is_valid = max_residual < tolerance
        
        return {
            'is_valid': is_valid,
            'max_residual': max_residual,
            'mean_residual': np.mean(residuals),
            'tolerance': tolerance,
            'equations_checked': len(residuals),
            'residuals_by_d': residuals
        }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Solve QFT Equation 27 - Complex Hadamard Matrix Condition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qft_equation_27.py 4
  python qft_equation_27.py 8 --method bfgs --output results_n8.txt
  python qft_equation_27.py 5 --method all --tolerance 1e-12
  
  # Multi-run to find multiple solutions:
  python qft_equation_27.py 4 --method differential-evolution --multi-run 10 --output multiple_n4.json
  python qft_equation_27.py 5 --method differential-evolution --multi-run 20 --seed 100
  
Methods available: bfgs, lbfgs, nelder-mead, differential-evolution, all
Note: --multi-run only works with differential-evolution method
        """
    )
    
    parser.add_argument('N', type=int, help='QFT dimension')
    parser.add_argument('--method', '-m', default='bfgs', 
                       choices=['bfgs', 'lbfgs', 'nelder-mead', 'differential-evolution', 'all'],
                       help='Optimization method to use')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-10,
                       help='Numerical tolerance for solution verification')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results')
    parser.add_argument('--max-iter', type=int, default=1000,
                       help='Maximum optimization iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for differential evolution (default: 42)')
    parser.add_argument('--multi-run', type=int, default=None,
                       help='Run multiple times with different seeds to find multiple solutions')
    
    args = parser.parse_args()
    
    print(f"QFT Equation 27 Solver - Dimension N = {args.N}")
    print("=" * 50)
    
    # Initialize solver
    solver = QFTEquation27Solver(args.N)
    print(f"Independent parameters: {solver.num_params} (M = floor(N/2) + 1)")
    print(f"Total equations: {args.N} (one for each d = 0, 1, ..., N-1)")
    print(f"Using symmetry constraint: λ_k = λ_{{N-k}}")
    
    # Choose methods to run
    if args.method == 'all':
        methods = ['bfgs', 'lbfgs', 'nelder-mead', 'differential-evolution']
    else:
        methods = [args.method]
    
    results = {}
    
    # Solve with each method
    for method in methods:
        print(f"\nSolving with {method.upper()}...")
        print("-" * 30)
        
        try:
            if method == 'bfgs':
                result = solver.solve_bfgs(max_iterations=args.max_iter)
            elif method == 'lbfgs':
                result = solver.solve_lbfgs(max_iterations=args.max_iter)
            elif method == 'nelder-mead':
                result = solver.solve_nelder_mead(max_iterations=args.max_iter)
            elif method == 'differential-evolution':
                if args.multi_run:
                    # Multi-run mode: generate multiple solutions
                    multi_results = solver.solve_differential_evolution_multirun(
                        n_runs=args.multi_run,
                        max_iterations=args.max_iter,
                        base_seed=args.seed
                    )
                    # VERIFY ALL SOLUTIONS in multi-run
                    successful_results = []
                    verified_results = []
                    
                    for mr in multi_results:
                        if 'error' not in mr:
                            # Verify each individual solution
                            mr_verification = solver.verify_solution(mr['lambda_full'], args.tolerance)
                            mr['verification'] = mr_verification
                            successful_results.append(mr)
                            
                            # Track verified (mathematically valid) solutions
                            if mr_verification['is_valid']:
                                verified_results.append(mr)
                    
                    if successful_results:
                        # Choose the best VERIFIED solution as main result
                        if verified_results:
                            # Find the verified solution with lowest objective value
                            result = min(verified_results, key=lambda r: r['objective_value'])
                            print(f"✅ Selected best verified solution from run {result['run_number']} "
                                  f"(objective: {result['objective_value']:.2e})")
                            
                            # Only include VERIFIED runs in the stored multi_run_results
                            verified_multi_results = [mr for mr in multi_results 
                                                    if ('error' not in mr and 
                                                        mr.get('verification', {}).get('is_valid', False)) or 
                                                       'error' in mr]  # Keep error entries for transparency
                            
                            result['multi_run_results'] = verified_multi_results
                            result['total_runs'] = args.multi_run
                            result['successful_runs'] = len(successful_results)
                            result['verified_runs'] = len(verified_results)
                            result['stored_runs'] = len(verified_results)  # Only verified runs stored
                            result['unique_solutions'] = len(set(
                                tuple(r['lambda_full']) for r in verified_results 
                                if 'duplicate_of_run' not in r
                            ))
                        else:
                            # No verified solutions found - don't store anything
                            print(f"❌ No mathematically verified solutions found in {args.multi_run} runs.")
                            print(f"   No results will be stored. Try different parameters or increase runs.")
                            raise ValueError(f"No mathematically valid solutions found in multi-run mode. "
                                           f"All {len(successful_results)} successful runs failed verification.")
                    else:
                        raise ValueError("No successful runs in multi-run mode")
                else:
                    # Single run mode
                    result = solver.solve_differential_evolution(
                        max_iterations=args.max_iter, 
                        seed=args.seed
                    )
            
            # Verify solution
            verification = solver.verify_solution(result['lambda_full'], args.tolerance)
            result['verification'] = verification
            
            # Only store mathematically valid solutions
            if verification['is_valid']:
                results[method] = result
                print(f"✅ Valid solution found and stored for {method}")
            else:
                print(f"❌ Solution for {method} is not mathematically valid (max_residual: {verification['max_residual']:.2e})")
                print(f"   Solution not stored. Try different parameters or methods.")
            
            # Print results
            print(f"Success: {result['success']}")
            print(f"Objective value: {result['objective_value']:.2e}")
            print(f"Solve time: {result['solve_time']:.3f} seconds")
            print(f"Iterations: {result['iterations']}")
            print(f"Initial guess: {result.get('initial_guess_type', 'Default')}")
            print(f"Solution valid: {verification['is_valid']}")
            print(f"Max residual: {verification['max_residual']:.2e}")
            
            # Show some parameter values if non-trivial
            if result['objective_value'] < 1e-6:
                print("First few λ parameters:")
                for i in range(min(5, len(result['lambda_full']))):
                    print(f"  λ_{i} = {result['lambda_full'][i]:.6f}")
                if len(result['lambda_full']) > 5:
                    print(f"  ... (total {len(result['lambda_full'])} parameters)")
            
        except Exception as e:
            print(f"Error with {method}: {e}")
            results[method] = {'error': str(e)}
    
    # Compare with standard QFT
    print(f"\nStandard QFT Reference:")
    print("-" * 25)
    qft_phases = solver.get_standard_qft_phases()
    qft_verification = solver.verify_solution(qft_phases, args.tolerance)
    print(f"Standard QFT satisfies equation: {qft_verification['is_valid']}")
    print(f"Standard QFT max residual: {qft_verification['max_residual']:.2e}")
    
    # Save results if requested AND if we have valid solutions
    if args.output and results:  # Only save if we have valid results
        import json
        
        print(f"\nSaving {len(results)} valid solution(s) to file...")
        
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(np, 'bool_') and isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert numpy arrays and types to JSON-serializable format
        json_results = {}
        for method, result in results.items():
            if 'error' not in result:
                json_result = {
                    'method': result['method'],
                    'lambda_full': result['lambda_full'].tolist(),
                    'objective_value': float(result['objective_value']),
                    'success': bool(result['success']),
                    'iterations': int(result['iterations']),
                    'solve_time': float(result['solve_time']),
                    'verification': convert_numpy_types(result['verification'])
                }
                
                # Add multi-run specific fields if present
                if 'multi_run_results' in result:
                    json_result['is_multi_run'] = True
                    json_result['total_runs'] = int(result['total_runs'])
                    json_result['successful_runs'] = int(result.get('successful_runs', 0))
                    json_result['verified_runs'] = int(result.get('verified_runs', 0))
                    json_result['stored_runs'] = int(result.get('stored_runs', 0))
                    json_result['unique_solutions'] = int(result['unique_solutions'])
                    
                    # Convert multi-run results - ONLY verified solutions are included
                    multi_results = []
                    for mr in result['multi_run_results']:
                        if 'error' not in mr:
                            # Only include if verified (this should always be true now)
                            if mr.get('verification', {}).get('is_valid', False):
                                multi_result = {
                                    'run_number': int(mr['run_number']),
                                    'seed_used': int(mr['seed_used']),
                                    'lambda_full': mr['lambda_full'].tolist(),
                                    'objective_value': float(mr['objective_value']),
                                    'success': bool(mr['success']),
                                    'iterations': int(mr['iterations']),
                                    'solve_time': float(mr['solve_time']),
                                    'verification': convert_numpy_types(mr['verification'])
                                }
                                
                                if 'duplicate_of_run' in mr:
                                    multi_result['duplicate_of_run'] = int(mr['duplicate_of_run'])
                                multi_results.append(multi_result)
                        else:
                            # Keep error entries for transparency
                            multi_results.append({
                                'run_number': int(mr['run_number']),
                                'seed_used': int(mr['seed_used']),
                                'error': mr['error']
                            })
                    
                    json_result['all_runs'] = multi_results
                
                # Add seed information if present
                if 'seed_used' in result:
                    json_result['seed_used'] = int(result['seed_used'])
                
                json_results[method] = json_result
            else:
                json_results[method] = result
        
        # Convert QFT verification as well
        qft_verification_json = convert_numpy_types(qft_verification)
        
        with open(args.output, 'w') as f:
            json.dump({
                'N': int(args.N),
                'tolerance': float(args.tolerance),
                'results': json_results,
                'standard_qft': {
                    'lambda_full': qft_phases.tolist(),
                    'verification': qft_verification_json
                }
            }, f, indent=2)
        
        print(f"\n✅ Results saved to: {args.output}")
    
    elif args.output and not results:
        print(f"\n❌ No valid solutions found. No file will be created.")
        print(f"   Try different methods, parameters, or increase tolerance.")


if __name__ == "__main__":
    main()
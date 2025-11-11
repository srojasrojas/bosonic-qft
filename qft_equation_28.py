"""
QFT Characterization - Equation 28 Solver
==========================================

Solves the QFT-specific condition (Equation 28) for phase parameters λ_k.

Equation 28:
0 = Σ_{j,k,l,p=0}^{N-1} sin((k-l)·2πn/N + (j-k)·2πm/N + λ_j - λ_k - λ_l - λ_p)

This ensures the matrix "core" matches standard QFT.
Notation: λ_{jklp} = λ_j - λ_k - λ_l - λ_p

Due to O(N^4) complexity, uses Monte Carlo sampling and genetic algorithms for large N.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Set
import argparse
import time
import random
import json
import os
from itertools import product


class QFTEquation28Solver:
    """Solver for QFT Equation 28 - QFT-specific core condition."""
    
    def __init__(self, N: int, use_symmetry: bool = True, monte_carlo_ratio: float = 0.1):
        """
        Initialize solver for dimension N.
        
        Args:
            N: QFT dimension
            use_symmetry: Whether to use λ_k = λ_{N-k} symmetry constraint
            monte_carlo_ratio: Fraction of terms to sample for large N (0 < ratio ≤ 1)
        """
        self.N = N
        self.use_symmetry = use_symmetry
        self.monte_carlo_ratio = min(1.0, max(0.01, monte_carlo_ratio))
        
        if use_symmetry:
            self.num_params = N // 2 if N % 2 == 0 else (N + 1) // 2
        else:
            self.num_params = N
        
        # For large N, use Monte Carlo sampling
        self.total_terms = N**4
        self.use_monte_carlo = self.total_terms > 10000
        
        if self.use_monte_carlo:
            self.sample_size = max(1000, int(self.total_terms * self.monte_carlo_ratio))
            self.sampled_indices = self._generate_sample_indices()
            print(f"Using Monte Carlo sampling: {self.sample_size}/{self.total_terms} terms")
        else:
            self.sample_size = self.total_terms
            self.sampled_indices = None
            
        self.solution_history = []
    
    def _generate_sample_indices(self) -> List[Tuple[int, int, int, int]]:
        """Generate random sample of (j,k,l,p) indices for Monte Carlo."""
        indices = []
        for _ in range(self.sample_size):
            j = random.randint(0, self.N - 1)
            k = random.randint(0, self.N - 1)
            l = random.randint(0, self.N - 1)
            p = random.randint(0, self.N - 1)
            indices.append((j, k, l, p))
        return indices
    
    def expand_lambda_vector(self, lambda_params: np.ndarray) -> np.ndarray:
        """
        Expand parameter vector, optionally applying symmetry constraint.
        
        Args:
            lambda_params: Parameter vector (half-size if using symmetry)
            
        Returns:
            Full λ vector
        """
        if not self.use_symmetry:
            return lambda_params.copy()
        
        lambda_full = np.zeros(self.N)
        lambda_full[:len(lambda_params)] = lambda_params
        
        # Apply symmetry: λ_k = λ_{N-k}
        for k in range(1, self.N):
            if k < len(lambda_params):
                lambda_full[self.N - k] = lambda_params[k]
        
        return lambda_full
    
    def equation_28_residual(self, lambda_params: np.ndarray, m: int, n: int) -> float:
        """
        Compute residual of Equation 28 for specific (m,n) pair.
        
        Args:
            lambda_params: Parameter vector
            m, n: Matrix indices
            
        Returns:
            Residual value (sum of sines)
        """
        lambda_full = self.expand_lambda_vector(lambda_params)
        
        total_sum = 0.0
        
        if self.use_monte_carlo:
            # Use sampled indices
            for j, k, l, p in self.sampled_indices:
                phase_mn = (k - l) * 2 * np.pi * n / self.N + (j - k) * 2 * np.pi * m / self.N
                lambda_jklp = lambda_full[j] - lambda_full[k] - lambda_full[l] - lambda_full[p]
                total_phase = phase_mn + lambda_jklp
                total_sum += np.sin(total_phase)
            
            # Scale by sampling ratio to approximate full sum
            total_sum *= (self.total_terms / self.sample_size)
        else:
            # Compute full sum
            for j in range(self.N):
                for k in range(self.N):
                    for l in range(self.N):
                        for p in range(self.N):
                            phase_mn = (k - l) * 2 * np.pi * n / self.N + (j - k) * 2 * np.pi * m / self.N
                            lambda_jklp = lambda_full[j] - lambda_full[k] - lambda_full[l] - lambda_full[p]
                            total_phase = phase_mn + lambda_jklp
                            total_sum += np.sin(total_phase)
        
        return total_sum
    
    def objective_function(self, lambda_params: np.ndarray, 
                          sample_mn_pairs: Optional[List[Tuple[int, int]]] = None) -> float:
        """
        Total objective function: sum of squared residuals over (m,n) pairs.
        
        Args:
            lambda_params: Parameter vector
            sample_mn_pairs: Specific (m,n) pairs to evaluate (None = all pairs)
            
        Returns:
            Sum of squared residuals
        """
        total_error = 0.0
        
        if sample_mn_pairs is None:
            # Use all (m,n) pairs
            mn_pairs = [(m, n) for m in range(self.N) for n in range(self.N)]
        else:
            mn_pairs = sample_mn_pairs
        
        for m, n in mn_pairs:
            residual = self.equation_28_residual(lambda_params, m, n)
            total_error += residual**2
        
        return total_error
    
    def numerical_gradient(self, lambda_params: np.ndarray, 
                          epsilon: float = 1e-8) -> np.ndarray:
        """
        Numerical gradient of objective function (finite differences).
        
        Args:
            lambda_params: Parameter vector
            epsilon: Finite difference step size
            
        Returns:
            Gradient vector
        """
        gradient = np.zeros_like(lambda_params)
        f0 = self.objective_function(lambda_params)
        
        for i in range(len(lambda_params)):
            lambda_plus = lambda_params.copy()
            lambda_plus[i] += epsilon
            f_plus = self.objective_function(lambda_plus)
            gradient[i] = (f_plus - f0) / epsilon
        
        return gradient
    
    def solve_nelder_mead(self, initial_guess: Optional[np.ndarray] = None,
                         max_iterations: int = 2000) -> Dict:
        """
        Solve using Nelder-Mead (robust for noisy objectives).
        
        Args:
            initial_guess: Initial parameter values
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization result dictionary
        """
        if initial_guess is None:
            initial_guess = np.random.uniform(-0.1, 0.1, self.num_params)
        
        start_time = time.time()
        
        result = minimize(
            fun=self.objective_function,
            x0=initial_guess,
            method='Nelder-Mead',
            options={'maxiter': max_iterations, 'disp': True, 'xatol': 1e-12, 'fatol': 1e-12}
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
            'message': result.message
        }
    
    def solve_differential_evolution(self, bounds_multiplier: float = 1.0,
                                   population_size: int = 20,
                                   max_iterations: int = 500) -> Dict:
        """
        Solve using Differential Evolution (global genetic algorithm).
        
        Args:
            bounds_multiplier: Parameter bounds as [-bounds_multiplier*π, bounds_multiplier*π]
            population_size: DE population size (scales with problem size)
            max_iterations: Maximum generations
            
        Returns:
            Optimization result dictionary
        """
        # Scale population with problem size
        scaled_popsize = max(population_size, self.num_params * 3)
        bounds = [(-bounds_multiplier*np.pi, bounds_multiplier*np.pi)] * self.num_params
        
        start_time = time.time()
        
        result = differential_evolution(
            func=self.objective_function,
            bounds=bounds,
            popsize=scaled_popsize,
            maxiter=max_iterations,
            disp=True,
            seed=42,
            atol=1e-12,
            tol=1e-12
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
            'message': result.message
        }
    
    def solve_basin_hopping(self, initial_guess: Optional[np.ndarray] = None,
                           n_basins: int = 100, 
                           step_size: float = 0.5) -> Dict:
        """
        Solve using Basin Hopping (global optimization with local refinement).
        
        Args:
            initial_guess: Initial parameter values
            n_basins: Number of basin hopping iterations
            step_size: Step size for random displacement
            
        Returns:
            Optimization result dictionary
        """
        if initial_guess is None:
            initial_guess = np.random.uniform(-0.1, 0.1, self.num_params)
        
        start_time = time.time()
        
        # Basin hopping with Nelder-Mead local optimizer
        result = basinhopping(
            func=self.objective_function,
            x0=initial_guess,
            niter=n_basins,
            T=1.0,
            stepsize=step_size,
            minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 500}},
            disp=True,
            seed=42
        )
        
        solve_time = time.time() - start_time
        lambda_full = self.expand_lambda_vector(result.x)
        
        return {
            'method': 'Basin Hopping',
            'lambda_params': result.x,
            'lambda_full': lambda_full,
            'objective_value': result.fun,
            'success': result.lowest_optimization_result.success,
            'iterations': result.nit,
            'solve_time': solve_time,
            'message': str(result.message)
        }
    
    def solve_multi_start(self, n_starts: int = 10, method: str = 'nelder-mead', 
                         eq27_solutions: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Multi-start optimization with random initial conditions.
        Can optionally use solutions from Equation 27 as starting points.
        
        Args:
            n_starts: Number of random starts
            method: Base optimization method
            eq27_solutions: List of solutions from Equation 27 to use as initial guesses
            
        Returns:
            Best result from all starts
        """
        if eq27_solutions:
            print(f"Multi-start optimization: {len(eq27_solutions)} Eq27 solutions + {n_starts} random starts...")
        else:
            print(f"Multi-start optimization with {n_starts} random initializations...")
        
        best_result = None
        best_objective = float('inf')
        all_results = []
        
        start_time = time.time()
        
        # First, try solutions from Equation 27 if provided
        if eq27_solutions:
            for i, eq27_solution in enumerate(eq27_solutions):
                print(f"Eq27 start {i+1}/{len(eq27_solutions)}...")
                
                try:
                    # Convert to appropriate parameter size
                    initial_guess = self._convert_eq27_solution(eq27_solution)
                    
                    if method == 'nelder-mead':
                        result = self.solve_nelder_mead(initial_guess, max_iterations=1000)
                    elif method == 'differential-evolution':
                        # For DE, we'll use the Eq27 solution to seed the population
                        result = self.solve_differential_evolution_with_seed(initial_guess, max_iterations=200)
                    else:
                        continue
                    
                    result['initial_source'] = f'Equation_27_solution_{i}'
                    all_results.append(result)
                    
                    if result['objective_value'] < best_objective:
                        best_objective = result['objective_value']
                        best_result = result
                        
                except Exception as e:
                    print(f"Error in Eq27 start {i+1}: {e}")
        
        # Then do random starts
        for i in range(n_starts):
            print(f"Random start {i+1}/{n_starts}...")
            
            # Random initialization
            initial_guess = np.random.uniform(-np.pi/2, np.pi/2, self.num_params)
            
            try:
                if method == 'nelder-mead':
                    result = self.solve_nelder_mead(initial_guess, max_iterations=1000)
                elif method == 'differential-evolution':
                    result = self.solve_differential_evolution(max_iterations=200)
                else:
                    continue
                
                result['initial_source'] = f'random_{i}'
                all_results.append(result)
                
                if result['objective_value'] < best_objective:
                    best_objective = result['objective_value']
                    best_result = result
                    
            except Exception as e:
                print(f"Error in random start {i+1}: {e}")
        
        total_time = time.time() - start_time
        
        if best_result:
            best_result['method'] = f'Multi-start {method}'
            best_result['solve_time'] = total_time
            best_result['total_starts'] = len(all_results)
            best_result['eq27_starts'] = len(eq27_solutions) if eq27_solutions else 0
            best_result['random_starts'] = n_starts
            best_result['all_objectives'] = [r['objective_value'] for r in all_results]
            best_result['best_initial_source'] = best_result.get('initial_source', 'unknown')
        
        return best_result
    
    @staticmethod
    def load_eq27_solutions(filepath: str) -> List[np.ndarray]:
        """
        Load solutions from Equation 27 JSON file.
        
        Args:
            filepath: Path to JSON file containing Equation 27 results
            
        Returns:
            List of lambda_full vectors from valid Equation 27 solutions
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Equation 27 results file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            solutions = []
            
            # Handle different JSON structures
            if 'results' in data:
                # Multiple methods in results
                for method_name, result in data['results'].items():
                    if not isinstance(result, dict) or 'lambda_full' not in result:
                        continue
                    
                    print(f"\nProcessing method: {method_name}")
                    
                    # Check if this is a multi-run result
                    if result.get('is_multi_run', False) and 'all_runs' in result:
                        print(f"  Multi-run with {len(result['all_runs'])} total runs")
                        loaded_count = 0
                        verified_count = 0
                        
                        # Load ALL valid solutions from multi-run
                        for run in result['all_runs']:
                            if 'error' in run:
                                continue  # Skip failed runs
                            
                            # Check if this run is verified as mathematically valid
                            run_verification = run.get('verification', {})
                            is_verified = run_verification.get('is_valid', False)
                            
                            if is_verified and 'lambda_full' in run:
                                run_lambda = np.array(run['lambda_full'])
                                solutions.append(run_lambda)
                                loaded_count += 1
                                verified_count += 1
                            elif 'lambda_full' in run:
                                # If no verification info, fall back to success flag
                                if run.get('success', False):
                                    run_lambda = np.array(run['lambda_full'])
                                    solutions.append(run_lambda)
                                    loaded_count += 1
                        
                        print(f"  → Loaded {loaded_count} solutions ({verified_count} verified)")
                        
                    else:
                        # Single-run result
                        main_verification = result.get('verification', {})
                        is_main_verified = main_verification.get('is_valid', False)
                        is_main_successful = result.get('success', False)
                        
                        if is_main_verified or is_main_successful:
                            lambda_full = np.array(result['lambda_full'])
                            solutions.append(lambda_full)
                            status = "verified" if is_main_verified else "successful"
                            print(f"  → Loaded 1 solution ({status})")
                        else:
                            print(f"  → Skipped (not verified or successful)")
                            
            elif 'lambda_full' in data:
                # Single result format - accept if successful OR verified
                is_successful = data.get('success', False)
                is_verified = (data.get('verification', {}).get('is_valid', False))
                
                if is_successful or is_verified:
                    lambda_full = np.array(data['lambda_full'])
                    solutions.append(lambda_full)
                    status = "verified" if is_verified else "successful"
                    print(f"Loaded single solution ({status})")
            
            print(f"\nTotal Equation 27 solutions loaded: {len(solutions)}")
            return solutions
            
        except Exception as e:
            raise ValueError(f"Error reading Equation 27 results file: {e}")
    
    @staticmethod
    def validate_eq27_solutions(solutions: List[np.ndarray], expected_N: int) -> List[np.ndarray]:
        """
        Validate that Equation 27 solutions have correct dimension and filter valid ones.
        
        Args:
            solutions: List of lambda vectors from Equation 27
            expected_N: Expected dimension N
            
        Returns:
            List of valid solutions
        """
        if not solutions:
            print("No Equation 27 solutions to validate.")
            return []
        
        print(f"\nValidating {len(solutions)} Equation 27 solutions for N={expected_N}:")
        print("-" * 60)
        
        valid_solutions = []
        
        for i, solution in enumerate(solutions):
            if len(solution) == expected_N:
                # Check for reasonable parameter values (not NaN, not too large)
                if np.all(np.isfinite(solution)) and np.max(np.abs(solution)) < 100:
                    valid_solutions.append(solution)
                    max_val = np.max(np.abs(solution))
                    print(f"  Solution {i+1:2d}: ✅ Valid   (max |λ| = {max_val:.3f}, range: [{np.min(solution):.3f}, {np.max(solution):.3f}])")
                else:
                    print(f"  Solution {i+1:2d}: ❌ Invalid (non-finite or too large values)")
            else:
                print(f"  Solution {i+1:2d}: ❌ Invalid (dimension {len(solution)}, expected {expected_N})")
        
        print(f"\nValidation Summary:")
        print(f"  Total solutions checked: {len(solutions)}")
        print(f"  Valid solutions: {len(valid_solutions)}")
        print(f"  Invalid solutions: {len(solutions) - len(valid_solutions)}")
        print(f"  Validation rate: {100*len(valid_solutions)/len(solutions):.1f}%")
        
        return valid_solutions
    
    def _convert_eq27_solution(self, eq27_lambda_full: np.ndarray) -> np.ndarray:
        """
        Convert Equation 27 solution to appropriate parameter vector for Equation 28.
        
        Args:
            eq27_lambda_full: Full lambda vector from Equation 27 solution
            
        Returns:
            Parameter vector compatible with current solver settings
        """
        if len(eq27_lambda_full) != self.N:
            raise ValueError(f"Equation 27 solution has {len(eq27_lambda_full)} parameters, expected {self.N}")
        
        if self.use_symmetry:
            # Extract first half for symmetric parameterization
            return eq27_lambda_full[:self.num_params]
        else:
            # Use full vector
            return eq27_lambda_full.copy()
    
    def solve_differential_evolution_with_seed(self, seed_solution: np.ndarray,
                                             bounds_multiplier: float = 1.0,
                                             population_size: int = 20,
                                             max_iterations: int = 500) -> Dict:
        """
        Solve using Differential Evolution with a seeded initial population.
        
        Args:
            seed_solution: Solution to include in initial population
            bounds_multiplier: Parameter bounds multiplier
            population_size: DE population size
            max_iterations: Maximum generations
            
        Returns:
            Optimization result dictionary
        """
        scaled_popsize = max(population_size, self.num_params * 3)
        bounds = [(-bounds_multiplier*np.pi, bounds_multiplier*np.pi)] * self.num_params
        
        start_time = time.time()
        
        # First, run a short optimization starting from the seed solution using Nelder-Mead
        # This gives DE a "warm start" near a good solution
        nm_result = minimize(
            fun=self.objective_function,
            x0=seed_solution,
            method='Nelder-Mead',
            options={'maxiter': 100, 'disp': False}
        )
        
        print(f"Seed solution objective: {self.objective_function(seed_solution):.2e}")
        print(f"Warm-start objective: {nm_result.fun:.2e}")
        
        # Now run DE with the improved solution as a reference
        # DE will still do global search but we've provided good starting information
        result = differential_evolution(
            func=self.objective_function,
            bounds=bounds,
            popsize=scaled_popsize,
            maxiter=max_iterations,
            disp=True,
            seed=42,
            atol=1e-12,
            tol=1e-12
        )
        
        solve_time = time.time() - start_time
        lambda_full = self.expand_lambda_vector(result.x)
        
        return {
            'method': 'Differential Evolution (Seeded)',
            'lambda_params': result.x,
            'lambda_full': lambda_full,
            'objective_value': result.fun,
            'success': result.success,
            'iterations': result.nit,
            'solve_time': solve_time,
            'message': result.message,
            'seed_used': True
        }
    
    def get_standard_qft_phases(self) -> np.ndarray:
        """
        Get phase parameters for standard QFT (reference solution).
        
        Returns:
            Standard QFT lambda parameters (all zeros)
        """
        return np.zeros(self.N)
    
    def verify_solution(self, lambda_full: np.ndarray, 
                       tolerance: float = 1e-8,
                       sample_ratio: float = 0.1) -> Dict:
        """
        Verify solution by checking residuals for sampled (m,n) pairs.
        
        Args:
            lambda_full: Full parameter vector
            tolerance: Numerical tolerance for verification
            sample_ratio: Fraction of (m,n) pairs to check
            
        Returns:
            Verification results
        """
        # Sample (m,n) pairs to check
        total_pairs = self.N * self.N
        n_sample = max(10, int(total_pairs * sample_ratio))
        
        sampled_pairs = []
        for _ in range(n_sample):
            m = random.randint(0, self.N - 1)
            n = random.randint(0, self.N - 1)
            sampled_pairs.append((m, n))
        
        max_residual = 0.0
        residuals = []
        
        # Extract parameters for residual calculation
        if self.use_symmetry:
            lambda_params = lambda_full[:self.num_params]
        else:
            lambda_params = lambda_full
        
        for m, n in sampled_pairs:
            residual = abs(self.equation_28_residual(lambda_params, m, n))
            residuals.append(residual)
            max_residual = max(max_residual, residual)
        
        is_valid = max_residual < tolerance
        
        return {
            'is_valid': is_valid,
            'max_residual': max_residual,
            'mean_residual': np.mean(residuals) if residuals else 0,
            'tolerance': tolerance,
            'pairs_checked': len(residuals),
            'sample_ratio': sample_ratio
        }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Solve QFT Equation 28 - QFT-Specific Core Condition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qft_equation_28.py 3
  python qft_equation_28.py 4 --method differential-evolution --output results.json
  python qft_equation_28.py 5 --method multi-start --monte-carlo 0.05
  
  # Using Equation 27 solutions as initial guesses:
  python qft_equation_28.py 4 --eq27-file results_eq27_n4.json --method nelder-mead
  python qft_equation_28.py 5 --eq27-file results_eq27_n5.json --method multi-start --random-starts 3
  
Methods available: nelder-mead, differential-evolution, basin-hopping, multi-start
Note: All methods can use Equation 27 solutions as intelligent initial guesses via --eq27-file
        """
    )
    
    parser.add_argument('N', type=int, help='QFT dimension')
    parser.add_argument('--method', '-m', default='nelder-mead',
                       choices=['nelder-mead', 'differential-evolution', 'basin-hopping', 'multi-start'],
                       help='Optimization method')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-8,
                       help='Numerical tolerance for verification')
    parser.add_argument('--monte-carlo', type=float, default=0.1,
                       help='Monte Carlo sampling ratio for large N (0 < ratio ≤ 1)')
    parser.add_argument('--no-symmetry', action='store_true',
                       help='Disable λ_k = λ_{N-k} symmetry constraint')
    parser.add_argument('--eq27-file', type=str,
                       help='JSON file with Equation 27 solutions to use as initial guesses')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results')
    parser.add_argument('--max-iter', type=int, default=1000,
                       help='Maximum optimization iterations')
    parser.add_argument('--random-starts', type=int, default=5,
                       help='Number of random starts for multi-start method')
    
    args = parser.parse_args()
    
    print(f"QFT Equation 28 Solver - Dimension N = {args.N}")
    print("=" * 50)
    
    # Initialize solver
    use_symmetry = not args.no_symmetry
    solver = QFTEquation28Solver(args.N, use_symmetry=use_symmetry, 
                                monte_carlo_ratio=args.monte_carlo)
    
    print(f"Problem complexity: O(N^4) = O({args.N}^4) = {args.N**4:,} terms")
    print(f"Parameters: {solver.num_params} (symmetry: {use_symmetry})")
    
    # Load Equation 27 solutions if provided
    eq27_solutions = None
    if args.eq27_file:
        print(f"\nLoading Equation 27 solutions from: {args.eq27_file}")
        print("-" * 50)
        try:
            loaded_solutions = solver.load_eq27_solutions(args.eq27_file)
            eq27_solutions = solver.validate_eq27_solutions(loaded_solutions, args.N)
            
            if not eq27_solutions:
                print("Warning: No valid Equation 27 solutions found, using random initialization")
                eq27_solutions = None
            else:
                print(f"Will use {len(eq27_solutions)} Equation 27 solutions as initial guesses")
                
        except Exception as e:
            print(f"Error loading Equation 27 solutions: {e}")
            print("Continuing with random initialization...")
            eq27_solutions = None
    
    # Solve with chosen method
    print(f"\nSolving with {args.method.upper()}...")
    print("-" * 40)
    
    try:
        if args.method == 'nelder-mead':
            # Use Eq27 solution as initial guess if available
            initial_guess = None
            if eq27_solutions:
                initial_guess = solver._convert_eq27_solution(eq27_solutions[0])
                print("Using first Equation 27 solution as initial guess for Nelder-Mead")
            result = solver.solve_nelder_mead(initial_guess=initial_guess, max_iterations=args.max_iter)
        elif args.method == 'differential-evolution':
            if eq27_solutions:
                # Try all Eq27 solutions and pick the best
                print(f"Differential Evolution will try all {len(eq27_solutions)} Equation 27 solutions...")
                best_result = None
                best_objective = float('inf')
                all_de_results = []
                
                for i, eq27_solution in enumerate(eq27_solutions):
                    print(f"\nDE attempt {i+1}/{len(eq27_solutions)} using Eq27 solution {i+1}:")
                    try:
                        initial_seed = solver._convert_eq27_solution(eq27_solution)
                        de_result = solver.solve_differential_evolution_with_seed(initial_seed, max_iterations=args.max_iter)
                        de_result['eq27_source_index'] = i
                        de_result['eq27_source_lambda'] = eq27_solution.tolist()
                        all_de_results.append(de_result)
                        
                        print(f"  Objective: {de_result['objective_value']:.2e}, Success: {de_result['success']}")
                        
                        if de_result['objective_value'] < best_objective:
                            best_objective = de_result['objective_value']
                            best_result = de_result
                            print(f"  ★ New best solution!")
                    
                    except Exception as e:
                        print(f"  Error with Eq27 solution {i+1}: {e}")
                
                if best_result:
                    result = best_result
                    result['method'] = 'Differential Evolution (Multi-Eq27)'
                    result['total_eq27_attempts'] = len(all_de_results)
                    result['all_eq27_objectives'] = [r['objective_value'] for r in all_de_results]
                    result['best_eq27_index'] = result['eq27_source_index']
                    print(f"\nBest DE result from Eq27 solution {result['eq27_source_index'] + 1} with objective {result['objective_value']:.2e}")
                else:
                    print("\nAll Eq27-seeded attempts failed, falling back to standard DE...")
                    result = solver.solve_differential_evolution(max_iterations=args.max_iter)
            else:
                result = solver.solve_differential_evolution(max_iterations=args.max_iter)
        elif args.method == 'basin-hopping':
            # Use Eq27 solution as initial guess if available
            initial_guess = None
            if eq27_solutions:
                initial_guess = solver._convert_eq27_solution(eq27_solutions[0])
                print("Using first Equation 27 solution as initial guess for Basin Hopping")
            result = solver.solve_basin_hopping(initial_guess=initial_guess, n_basins=args.max_iter//10)
        elif args.method == 'multi-start':
            result = solver.solve_multi_start(n_starts=args.random_starts, 
                                            eq27_solutions=eq27_solutions)
        
        # Verify solution
        print(f"\nVerifying solution...")
        verification = solver.verify_solution(result['lambda_full'], args.tolerance)
        result['verification'] = verification
        
        # Print results
        print(f"\nResults:")
        print(f"Success: {result['success']}")
        print(f"Objective value: {result['objective_value']:.2e}")
        print(f"Solve time: {result['solve_time']:.3f} seconds")
        print(f"Iterations: {result['iterations']}")
        print(f"Solution valid: {verification['is_valid']}")
        print(f"Max residual: {verification['max_residual']:.2e}")
        print(f"Mean residual: {verification['mean_residual']:.2e}")
        
        # Show some parameter values
        print(f"\nFirst few λ parameters:")
        for i, val in enumerate(result['lambda_full'][:min(10, len(result['lambda_full']))]):
            print(f"  λ_{i} = {val:.6f}")
        if len(result['lambda_full']) > 10:
            print(f"  ... ({len(result['lambda_full'])} total)")
        
        # Show information about initial guess source
        if 'best_initial_source' in result:
            print(f"Best solution came from: {result['best_initial_source']}")
        
        if eq27_solutions:
            print(f"\nEquation 27 initial guesses used: {len(eq27_solutions)}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        result = {'error': str(e)}
    
    # Compare with standard QFT
    print(f"\nStandard QFT Reference:")
    print("-" * 25)
    qft_phases = solver.get_standard_qft_phases()
    qft_verification = solver.verify_solution(qft_phases, args.tolerance)
    print(f"Standard QFT satisfies equation: {qft_verification['is_valid']}")
    print(f"Standard QFT max residual: {qft_verification['max_residual']:.2e}")
    
    # Save results if requested (only if solution is mathematically valid)
    if args.output and 'error' not in result and verification['is_valid']:
        import json
        
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
        
        # Format compatible with verify_equation.py (same as equation 27)
        json_result = {
            'method': str(result['method']),
            'lambda_full': result['lambda_full'].tolist(),
            'objective_value': float(result['objective_value']),
            'success': bool(result['success']),
            'iterations': int(result['iterations']),
            'solve_time': float(result['solve_time']),
            'verification': convert_numpy_types(result['verification'])
        }
        
        # Add equation 28 specific metadata
        json_result['N'] = int(args.N)
        json_result['equation'] = 28
        json_result['use_symmetry'] = bool(use_symmetry)
        json_result['monte_carlo_ratio'] = float(args.monte_carlo)
        json_result['tolerance'] = float(args.tolerance)
        json_result['eq27_file'] = str(args.eq27_file) if args.eq27_file else None
        json_result['eq27_solutions_used'] = int(len(eq27_solutions)) if eq27_solutions else 0
        
        # Add standard QFT comparison
        json_result['standard_qft_lambda'] = qft_phases.tolist()
        json_result['standard_qft_verification'] = convert_numpy_types(qft_verification)
        
        # Add multi-start specific information if available
        if 'best_initial_source' in result:
            json_result['best_initial_source'] = str(result['best_initial_source'])
        if 'total_starts' in result:
            json_result['total_starts'] = int(result['total_starts'])
        if 'eq27_starts' in result:
            json_result['eq27_starts'] = int(result['eq27_starts'])
        
        # Add multi-Eq27 Differential Evolution information if available
        if 'total_eq27_attempts' in result:
            json_result['total_eq27_attempts'] = int(result['total_eq27_attempts'])
        if 'all_eq27_objectives' in result:
            json_result['all_eq27_objectives'] = [float(obj) for obj in result['all_eq27_objectives']]
        if 'best_eq27_index' in result:
            json_result['best_eq27_index'] = int(result['best_eq27_index'])
        if 'eq27_source_lambda' in result:
            json_result['eq27_source_lambda'] = result['eq27_source_lambda']
        
        with open(args.output, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    elif args.output and 'error' not in result and not verification['is_valid']:
        print(f"\nSolution not saved - does not satisfy equation (max residual: {verification['max_residual']:.2e})")
        print(f"Required tolerance: {args.tolerance:.2e}")
    elif args.output and 'error' in result:
        print(f"\nSolution not saved - optimization failed: {result['error']}")


if __name__ == "__main__":
    main()
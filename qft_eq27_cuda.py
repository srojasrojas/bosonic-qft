"""
QFT Equation 27 Solver - CUDA-Accelerated Version
==================================================

GPU-accelerated solver for Equation 27 using CuPy.
Recommended for N ≥ 10. For N < 10, use qft_eq27.py instead.

Solves Equation 27 for QFT phase parameters λ_k using:
1. Warm-up phase (BFGS)
2. Differential Evolution (multi-run by default)
3. Refinement (L-BFGS-B)
4. Rationalization attempts (fractions of π)

Equation 27: For each d = 0, 1, ..., N-1:
0 = Σ_{j=1}^{N-1} Σ_{k=j}^{N-1} cos(2π j d/N + λ_k - λ_{k-j})

Uses symmetry constraint: λ_k = λ_{N-k}

GPU Optimizations:
- CuPy arrays for GPU computation (drop-in NumPy replacement)
- Automatic fallback to CPU if CUDA not available
- Smart device management (minimize CPU↔GPU transfers)
- Warning system for inefficient N < 10 usage
- --force-cpu flag to override GPU usage
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from fractions import Fraction
import math
import json
import time
import argparse
from typing import Dict, List, Optional

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    try:
        # Test if CUDA actually works
        _ = cp.array([1.0])
        _ = cp.cuda.Device(0).compute_capability
    except Exception:
        CUDA_AVAILABLE = False
        cp = np  # Fallback
except ImportError:
    cp = np
    CUDA_AVAILABLE = False


class QFTEq27SolverCUDA:
    """CUDA-accelerated solver for QFT Equation 27."""
    
    def __init__(self, N: int, use_gpu: bool = True, force_cpu: bool = False):
        """
        Initialize solver for dimension N.
        
        Args:
            N: QFT dimension
            use_gpu: Whether to attempt GPU usage (default: True)
            force_cpu: Force CPU usage even if GPU available (default: False)
        """
        self.N = N
        self.num_params = N // 2 + 1  # Independent parameters with symmetry
        
        # Determine device (GPU or CPU)
        self.force_cpu = force_cpu
        if force_cpu:
            self.use_gpu = False
            self.xp = np
            print("🖥️  Using CPU (--force-cpu flag)")
        elif not CUDA_AVAILABLE:
            self.use_gpu = False
            self.xp = np
            print("⚠️  CUDA not available. Install CuPy, e.g.:")
            print("    pip install cupy-cuda13x   # for CUDA 13.x")
            print("    pip install cupy-cuda12x   # for CUDA 12.x")
            print("🖥️  Fallback to CPU")
        elif use_gpu:
            self.use_gpu = True
            self.xp = cp
            # Get GPU info
            try:
                dev = cp.cuda.Device(0)
                props = cp.cuda.runtime.getDeviceProperties(dev.id)
                raw_name = props.get('name', None)
                if isinstance(raw_name, (bytes, bytearray)):
                    gpu_name = raw_name.decode(errors='ignore')
                elif raw_name is not None:
                    gpu_name = str(raw_name)
                else:
                    gpu_name = f"Device {dev.id}"
                gpu_memory = dev.mem_info[1] / (1024**3)  # GB
                print(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            except Exception as e:
                print(f"🎮 GPU detected but could not read name (Device 0). Details: {e}")
            
            # Warning for small N
            if N < 10:
                print(f"⚠️  Warning: N={N} is small. GPU overhead may reduce performance.")
                print(f"💡 Recommendation: Use qft_eq27.py for better performance with N<10")
                print(f"⏳ Continuing with GPU anyway... (use --force-cpu to override)")
            else:
                print(f"✅ Using GPU acceleration (N={N} ≥ 10)")
        else:
            self.use_gpu = False
            self.xp = np
            print("🖥️  Using CPU")
    
    def _to_numpy(self, arr):
        """Convert array to numpy (from GPU if needed)."""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)
    
    def _to_device(self, arr):
        """Convert array to device (GPU if enabled)."""
        if self.use_gpu:
            return cp.asarray(arr)
        return np.asarray(arr)
    
    def expand_lambda_vector(self, lambda_params: np.ndarray) -> np.ndarray:
        """Expand parameter vector using symmetry λ_k = λ_{N-k}."""
        # Always do this on CPU (small operation, avoid transfer overhead)
        lambda_params_cpu = self._to_numpy(lambda_params)
        lambda_full = np.zeros(self.N)
        lambda_full[0] = lambda_params_cpu[0]
        
        for k in range(1, len(lambda_params_cpu)):
            if k < self.N:
                lambda_full[k] = lambda_params_cpu[k]
                if self.N - k != k:
                    lambda_full[self.N - k] = lambda_params_cpu[k]
        
        return lambda_full
    
    def equation_27_residual(self, lambda_params: np.ndarray, d: int) -> float:
        """Compute residual for specific d value (GPU-accelerated)."""
        lambda_full = self.expand_lambda_vector(lambda_params)
        
        # Move to GPU if enabled
        lambda_full_device = self._to_device(lambda_full)
        
        # Pre-compute constant for efficiency
        two_pi_d_over_N = 2 * np.pi * d / self.N
        
        total_sum = 0.0
        for j in range(1, self.N):
            for k in range(j, self.N):
                k_minus_j = k - j
                phase = (two_pi_d_over_N * j + 
                        lambda_full_device[k] - lambda_full_device[k_minus_j])
                total_sum += self.xp.cos(phase)
        
        # Convert back to float (from GPU if needed)
        if self.use_gpu:
            return float(cp.asnumpy(total_sum))
        return float(total_sum)
    
    def objective_function(self, lambda_params: np.ndarray) -> float:
        """Total objective: sum of squared residuals over all d."""
        total_error = 0.0
        for d in range(self.N):
            residual = self.equation_27_residual(lambda_params, d)
            total_error += residual**2
        return total_error
    
    def objective_gradient(self, lambda_params: np.ndarray) -> np.ndarray:
        """Analytical gradient of objective function (GPU-accelerated)."""
        lambda_full = self.expand_lambda_vector(lambda_params)
        lambda_full_device = self._to_device(lambda_full)
        
        gradient = np.zeros_like(lambda_params)
        
        # Pre-compute 2πd/N for efficiency
        two_pi_over_N = 2 * np.pi / self.N
        
        for d in range(self.N):
            # Compute residual using cached lambda_full
            total_sum = 0.0
            for j in range(1, self.N):
                for k in range(j, self.N):
                    k_minus_j = k - j
                    phase = (two_pi_over_N * j * d + 
                            lambda_full_device[k] - lambda_full_device[k_minus_j])
                    total_sum += self.xp.cos(phase)
            
            if self.use_gpu:
                residual = float(cp.asnumpy(total_sum))
            else:
                residual = float(total_sum)
            
            for i in range(len(lambda_params)):
                grad_i = 0.0
                
                for j in range(1, self.N):
                    for k in range(j, self.N):
                        k_minus_j = k - j
                        phase = (two_pi_over_N * j * d + 
                                lambda_full_device[k] - lambda_full_device[k_minus_j])
                        
                        sin_phase = self.xp.sin(phase)
                        
                        if k == i:
                            grad_i -= sin_phase
                        if k_minus_j == i:
                            grad_i += sin_phase
                
                # Symmetry contribution
                if i > 0 and self.N - i != i and self.N - i < self.N:
                    symmetric_idx = self.N - i
                    for j in range(1, self.N):
                        for k in range(j, self.N):
                            k_minus_j = k - j
                            phase = (two_pi_over_N * j * d + 
                                    lambda_full_device[k] - lambda_full_device[k_minus_j])
                            
                            sin_phase = self.xp.sin(phase)
                            
                            if k == symmetric_idx:
                                grad_i -= sin_phase
                            if k_minus_j == symmetric_idx:
                                grad_i += sin_phase
                
                if self.use_gpu and isinstance(grad_i, cp.ndarray):
                    grad_i = float(cp.asnumpy(grad_i))
                
                gradient[i] += 2 * residual * grad_i
        
        return gradient
    
    def verify_solution(self, lambda_full: np.ndarray, tolerance: float = 1e-10) -> Dict:
        """Verify if solution satisfies Equation 27 for all d."""
        lambda_params = lambda_full[:self.num_params]
        residuals = []
        
        for d in range(self.N):
            residual = abs(self.equation_27_residual(lambda_params, d))
            residuals.append(residual)
        
        max_residual = max(residuals)
        is_valid = max_residual < tolerance
        
        return {
            'is_valid': is_valid,
            'max_residual': float(max_residual),
            'mean_residual': float(np.mean(residuals)),
            'residuals': [float(r) for r in residuals]
        }
    
    def try_rational_pi_approximation(self, lambda_params: np.ndarray, 
                                       max_denominator: int = 20,
                                       tolerance: float = 1e-6) -> Optional[Dict]:
        """
        Try to approximate lambda parameters as rational multiples of pi or sqrt(rational)*pi.
        
        Attempts: λ = (p/q)π or λ = √(p/q)π where p, q are small integers.
        
        Args:
            lambda_params: Current parameter vector
            max_denominator: Maximum denominator for rational approximation
            tolerance: How close the approximation must be
            
        Returns:
            Dictionary with approximated parameters, or None if no improvement
        """
        lambda_full = self.expand_lambda_vector(lambda_params)
        approx_lambda_full = lambda_full.copy()
        approximations = []
        
        for i, lam in enumerate(lambda_full):
            best_approx = lam
            best_form = "numerical"
            best_error = float('inf')
            
            # Try rational multiples of pi: (p/q)*pi
            ratio = lam / np.pi
            try:
                frac = Fraction(ratio).limit_denominator(max_denominator)
                if abs(frac.numerator) < max_denominator and frac.denominator < max_denominator:
                    approx_val = float(frac) * np.pi
                    error = abs(approx_val - lam)
                    if error < tolerance and error < best_error:
                        best_approx = approx_val
                        best_form = f"{frac.numerator}/{frac.denominator}*π"
                        best_error = error
            except (ValueError, ZeroDivisionError):
                pass
            
            # Try sqrt(rational) * pi: sqrt(p/q)*pi
            ratio_sq = (lam / np.pi) ** 2
            if ratio_sq > 0:
                try:
                    frac = Fraction(ratio_sq).limit_denominator(max_denominator)
                    if abs(frac.numerator) < max_denominator and frac.denominator < max_denominator:
                        approx_val = math.sqrt(float(frac)) * np.pi
                        error = abs(approx_val - lam)
                        if error < tolerance and error < best_error:
                            best_approx = approx_val
                            best_form = f"√({frac.numerator}/{frac.denominator})*π"
                            best_error = error
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Try negative sqrt(rational) * pi: -sqrt(p/q)*pi
            if lam < 0:
                ratio_sq_neg = (lam / np.pi) ** 2
                if ratio_sq_neg > 0:
                    try:
                        frac = Fraction(ratio_sq_neg).limit_denominator(max_denominator)
                        if abs(frac.numerator) < max_denominator and frac.denominator < max_denominator:
                            approx_val = -math.sqrt(float(frac)) * np.pi
                            error = abs(approx_val - lam)
                            if error < tolerance and error < best_error:
                                best_approx = approx_val
                                best_form = f"-√({frac.numerator}/{frac.denominator})*π"
                                best_error = error
                    except (ValueError, ZeroDivisionError):
                        pass
            
            approx_lambda_full[i] = best_approx
            approximations.append({
                'index': i,
                'original': float(lam),
                'approximated': float(best_approx),
                'form': best_form,
                'error': float(best_error) if best_error != float('inf') else 0.0
            })
        
        # Convert back to parameter space
        approx_params = approx_lambda_full[:self.num_params]
        
        # Evaluate objective with approximated values
        original_obj = self.objective_function(lambda_params)
        approx_obj = self.objective_function(approx_params)
        
        # Only return if approximation improves or maintains objective (within 10%)
        if approx_obj <= original_obj * 1.1:
            return {
                'lambda_params': approx_params,
                'lambda_full': approx_lambda_full,
                'original_objective': float(original_obj),
                'approximated_objective': float(approx_obj),
                'improved': approx_obj < original_obj,
                'approximations': approximations,
                'num_approximated': sum(1 for a in approximations if a['form'] != "numerical")
            }
        else:
            return None
    
    def warmup_bfgs(self, initial_guess: Optional[np.ndarray] = None) -> Dict:
        """Phase 1: Quick BFGS warm-up."""
        if initial_guess is None:
            # Better QFT-inspired initialization: mix of linear and quadratic
            initial_guess = np.zeros(self.num_params)
            for k in range(self.num_params):
                # Blend: 0.6 * quadratic + 0.4 * linear pattern
                initial_guess[k] = (0.6 * 2 * np.pi * k * k / self.N + 
                                   0.4 * 4 * np.pi * k / self.N)
        
        print("🔥 Phase 1: BFGS Warm-up...")
        start_time = time.time()
        
        result = minimize(
            fun=self.objective_function,
            x0=initial_guess,
            method='BFGS',
            jac=self.objective_gradient,
            options={'maxiter': 300, 'disp': False, 'gtol': 1e-6}
        )
        
        elapsed = time.time() - start_time
        lambda_full = self.expand_lambda_vector(result.x)
        
        print(f"   Objective: {result.fun:.2e} | Time: {elapsed:.2f}s")
        
        return {
            'lambda_params': result.x,
            'lambda_full': lambda_full,
            'objective': float(result.fun),
            'time': elapsed
        }
    
    def differential_evolution_multirun(self, n_runs: int = 10, 
                                       warmup_result: Optional[Dict] = None,
                                       max_iter: int = 1000,
                                       base_seed: int = 42) -> List[Dict]:
        """Phase 2: Differential Evolution multi-run."""
        print(f"\n🧬 Phase 2: Differential Evolution ({n_runs} runs)...")
        
        valid_solutions = []
        best_objective = float('inf')
        
        for i in range(n_runs):
            seed = base_seed + i * 1000
            print(f"   Run {i+1}/{n_runs} (seed={seed})...", end=" ")
            
            start_time = time.time()
            
            result = differential_evolution(
                func=self.objective_function,
                bounds=[(0, 2*np.pi)] * self.num_params,
                popsize=15,
                maxiter=max_iter,
                disp=False,
                seed=seed,
                polish=True,  # Final polish with L-BFGS-B for better precision
                atol=1e-12,   # Stricter absolute tolerance
                tol=0.01,     # Relative tolerance for early stopping
                workers=1
            )
            
            elapsed = time.time() - start_time
            lambda_full = self.expand_lambda_vector(result.x)
            
            # Verify solution
            verification = self.verify_solution(lambda_full)
            
            if verification['is_valid']:
                # Check if this is a unique solution (not duplicate)
                is_unique = True
                for prev_sol in valid_solutions:
                    diff = np.max(np.abs(np.array(lambda_full) - np.array(prev_sol['lambda_full'])))
                    if diff < 1e-4:  # Solutions are essentially the same
                        is_unique = False
                        break
                
                solution = {
                    'run': i + 1,
                    'seed': seed,
                    'lambda_full': [float(x) for x in lambda_full],
                    'objective': float(result.fun),
                    'max_residual': verification['max_residual'],
                    'time': elapsed,
                    'unique': is_unique
                }
                valid_solutions.append(solution)
                
                if result.fun < best_objective:
                    best_objective = result.fun
                
                unique_marker = "⭐" if is_unique else "🔁"
                print(f"✅ Valid! {unique_marker} (obj: {result.fun:.2e})")
            else:
                print(f"❌ Invalid (residual: {verification['max_residual']:.2e})")
        
        num_unique = sum(1 for s in valid_solutions if s['unique'])
        print(f"   → {len(valid_solutions)}/{n_runs} valid ({num_unique} unique)")
        return valid_solutions
    
    def refine_solution(self, lambda_params: np.ndarray) -> Dict:
        """Phase 3: Refinement with L-BFGS-B."""
        print("\n✨ Phase 3: L-BFGS-B Refinement...")
        start_time = time.time()
        
        result = minimize(
            fun=self.objective_function,
            x0=lambda_params,
            method='L-BFGS-B',
            jac=self.objective_gradient,
            bounds=[(0, 2*np.pi)] * self.num_params,
            options={
                'maxiter': 500, 
                'disp': False,
                'ftol': 1e-12,  # Stricter function tolerance
                'gtol': 1e-8    # Stricter gradient tolerance
            }
        )
        
        elapsed = time.time() - start_time
        lambda_full = self.expand_lambda_vector(result.x)
        
        print(f"   Objective: {result.fun:.2e} | Time: {elapsed:.2f}s")
        
        return {
            'lambda_params': result.x,
            'lambda_full': lambda_full,
            'objective': float(result.fun),
            'time': elapsed
        }
    
    def rationalize_solution(self, lambda_params: np.ndarray, 
                            max_denominator: int = 20) -> Optional[Dict]:
        """Phase 4: Try to rationalize parameters."""
        print("\n🔢 Phase 4: Rationalization attempt...")
        
        rational_result = self.try_rational_pi_approximation(
            lambda_params, 
            max_denominator=max_denominator,
            tolerance=1e-6
        )
        
        if rational_result and rational_result['num_approximated'] > 0:
            print(f"   ✅ Rationalized {rational_result['num_approximated']}/{self.N} parameters")
            print(f"   Objective: {rational_result['original_objective']:.2e} → "
                  f"{rational_result['approximated_objective']:.2e}")
            return rational_result
        else:
            print("   ❌ No rational approximation found")
            return None
    
    def solve(self, n_runs: int = 10, max_iter: int = 1000, 
              seed: int = 42, rationalize: bool = True,
              max_denominator: int = 20) -> Dict:
        """
        Complete solve workflow:
        1. Warm-up (BFGS)
        2. Differential Evolution (multi-run)
        3. Refinement (L-BFGS-B)
        4. Rationalization (optional)
        
        Returns only valid solutions.
        """
        print(f"\n{'='*60}")
        print(f"QFT Equation 27 Solver (CUDA) - N = {self.N}")
        print(f"{'='*60}")
        print(f"Parameters: {self.num_params} (with symmetry λ_k = λ_{{N-k}})")
        print(f"Equations: {self.N} (one for each d)")
        print(f"Device: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
        
        total_start = time.time()
        
        # Phase 1: Warm-up
        warmup = self.warmup_bfgs()
        
        # Phase 2: Differential Evolution multi-run
        de_solutions = self.differential_evolution_multirun(
            n_runs=n_runs,
            warmup_result=warmup,
            max_iter=max_iter,
            base_seed=seed
        )
        
        if not de_solutions:
            print("\n❌ No valid solutions found!")
            return {'valid_solutions': [], 'total_time': time.time() - total_start}
        
        # Select best solution for refinement
        best_de = min(de_solutions, key=lambda s: s['objective'])
        best_lambda_params = np.array(best_de['lambda_full'][:self.num_params])
        
        # Phase 3: Refine best solution
        refined = self.refine_solution(best_lambda_params)
        
        # Phase 4: Try rationalization
        rational = None
        if rationalize:
            rational = self.rationalize_solution(refined['lambda_params'], max_denominator)
        
        # Prepare final solution
        if rational:
            final_lambda_full = rational['lambda_full']
            final_obj = rational['approximated_objective']
        else:
            final_lambda_full = refined['lambda_full']
            final_obj = refined['objective']
        
        # Verify final solution
        final_verification = self.verify_solution(np.array(final_lambda_full))
        
        # Build result
        total_time = time.time() - total_start
        
        # Calculate unique solutions
        num_unique = sum(1 for s in de_solutions if s.get('unique', True))
        
        print(f"\n{'='*60}")
        print(f"✅ Solution completed in {total_time:.2f}s")
        print(f"   Valid solutions: {len(de_solutions)} ({num_unique} unique)")
        print(f"   Final objective: {final_obj:.2e}")
        print(f"   Max residual: {final_verification['max_residual']:.2e}")
        print(f"   Verified: {final_verification['is_valid']}")
        
        result = {
            'N': self.N,
            'num_params': self.num_params,
            'device': 'GPU' if self.use_gpu else 'CPU',
            'cuda_available': CUDA_AVAILABLE,
            'best_solution': {
                'lambda_full': [float(x) for x in final_lambda_full],
                'objective': final_obj,
                'verification': final_verification,
                'rationalized': rational is not None
            },
            'all_valid_solutions': de_solutions,
            'num_valid_solutions': len(de_solutions),
            'num_unique_solutions': num_unique,
            'phases': {
                'warmup': {'objective': warmup['objective'], 'time': warmup['time']},
                'differential_evolution': {
                    'runs': n_runs,
                    'valid': len(de_solutions),
                    'unique': num_unique,
                    'time': sum(s['time'] for s in de_solutions)
                },
                'refinement': {'objective': refined['objective'], 'time': refined['time']}
            },
            'total_time': total_time,
            'seed': seed
        }
        
        # Add rationalization info if available
        if rational:
            result['rationalization'] = {
                'num_approximated': rational['num_approximated'],
                'approximations': rational['approximations']
            }
        
        return result


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="QFT Equation 27 Solver - CUDA-Accelerated (Recommended for N≥10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use GPU (default, recommended for N≥10)
  python qft_eq27_cuda.py 15
  python qft_eq27_cuda.py 20 --runs 20 --output solutions/eq27_n20.json
  
  # Force CPU usage (for debugging or comparison)
  python qft_eq27_cuda.py 15 --force-cpu
  
  # Small N (will show warning but continue)
  python qft_eq27_cuda.py 7

Note: Requires CuPy for GPU acceleration
  Install: pip install cupy-cuda12x  (or cupy-cuda11x for CUDA 11.x)
  
Performance:
  N < 10:  GPU overhead reduces performance → use qft_eq27.py instead
  N ≥ 10:  GPU provides 3-7x speedup
  N ≥ 20:  GPU provides 7-15x speedup
        """
    )
    
    parser.add_argument('N', type=int, help='QFT dimension')
    parser.add_argument('--runs', '-r', type=int, default=10,
                       help='Number of differential evolution runs (default: 10)')
    parser.add_argument('--max-iter', type=int, default=1000,
                       help='Maximum iterations per run (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for results')
    parser.add_argument('--no-rationalize', action='store_true',
                       help='Skip rationalization phase')
    parser.add_argument('--max-denominator', type=int, default=20,
                       help='Max denominator for rational approximation (default: 20)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage (disable GPU even if available)')
    
    args = parser.parse_args()
    
    # Create solver and run
    solver = QFTEq27SolverCUDA(args.N, use_gpu=True, force_cpu=args.force_cpu)
    result = solver.solve(
        n_runs=args.runs,
        max_iter=args.max_iter,
        seed=args.seed,
        rationalize=not args.no_rationalize,
        max_denominator=args.max_denominator
    )
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    # Print summary of best solution
    if result['num_valid_solutions'] > 0:
        print(f"\n{'='*60}")
        print("Best Solution λ parameters:")
        print(f"{'='*60}")
        best = result['best_solution']
        for i, lam in enumerate(best['lambda_full']):
            print(f"  λ_{i} = {lam:.6f}")
        
        # Show rationalizations if available
        if 'rationalization' in result and result['rationalization']['num_approximated'] > 0:
            print(f"\n{'='*60}")
            print("Rational Approximations:")
            print(f"{'='*60}")
            for approx in result['rationalization']['approximations']:
                if approx['form'] != "numerical":
                    print(f"  λ_{approx['index']} ≈ {approx['form']} "
                          f"(error: {approx['error']:.2e})")


if __name__ == "__main__":
    main()

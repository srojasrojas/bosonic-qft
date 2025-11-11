"""
QFT Equation Solution Verifier
=============================

Verifies that lambda_full solutions from JSON files satisfy either:
- Equation 27 (Complex Hadamard condition)
- Equation 28 (QFT-specific condition)

Equation 27 (Complex Hadamard):
For each d = 0, 1, ..., N-1:
0 = Σ_{j=1}^{N-1} Σ_{k=j}^{N-1} cos(2π j d/N + λ_k - λ_{k-j})

Equation 28 (QFT-specific):
For each m,n = 0, 1, ..., N-1:
0 = Σ_{j,k,l,p=0}^{N-1} sin((k-l)2πn/N + (j-k)2πm/N + λ_j - λ_k - λ_l - λ_p)

This script loads solutions from JSON and verifies them independently.
"""

import numpy as np
import json
import argparse
import sys
import os


def equation_27_residual(lambda_full, d, N):
    """
    Compute residual of Equation 27 for specific difference d.
    
    Args:
        lambda_full: Full lambda vector [λ_0, λ_1, ..., λ_{N-1}]
        d: Difference d = m-n (mod N)
        N: QFT dimension
        
    Returns:
        Residual value for this specific d
    """
    total_sum = 0.0
    
    for j in range(1, N):
        for k in range(j, N):
            k_minus_j = k - j
            phase = (2 * np.pi * j * d / N + 
                    lambda_full[k] - lambda_full[k_minus_j])
            total_sum += np.cos(phase)
    
    return total_sum


def equation_28_residual(lambda_full, m, n, N):
    """
    Compute residual of Equation 28 for specific (m,n) pair.
    
    Args:
        lambda_full: Full lambda vector [λ_0, λ_1, ..., λ_{N-1}]
        m, n: QFT indices (0 ≤ m,n ≤ N-1)
        N: QFT dimension
        
    Returns:
        Residual value for this specific (m,n) pair
    """
    total_sum = 0.0
    
    for j in range(N):
        for k in range(N):
            for l in range(N):
                for p in range(N):
                    phase_mn = (k - l) * 2 * np.pi * n / N + (j - k) * 2 * np.pi * m / N
                    lambda_jklp = lambda_full[j] - lambda_full[k] - lambda_full[l] - lambda_full[p]
                    total_phase = phase_mn + lambda_jklp
                    total_sum += np.sin(total_phase)
    
    return total_sum


def verify_solution(lambda_full, N, equation=27, tolerance=1e-10, verbose=True):
    """
    Verify if lambda_full satisfies the specified equation.
    
    Args:
        lambda_full: Full lambda vector
        N: QFT dimension
        equation: Which equation to verify (27 or 28)
        tolerance: Numerical tolerance
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with verification results
    """
    if verbose:
        print(f"Verifying solution for N = {N}, Equation {equation}")
        print("Lambda values:")
        for i, val in enumerate(lambda_full):
            print(f"  λ_{i} = {val:.6f}")
        print()
    
    residuals = []
    max_residual = 0.0
    
    if equation == 27:
        if verbose:
            print("Equation 27 residuals for each d:")
        
        for d in range(N):
            residual = equation_27_residual(lambda_full, d, N)
            residuals.append(residual)
            max_residual = max(max_residual, abs(residual))
            
            if verbose:
                print(f"  d = {d}: residual = {residual:12.6e} (|residual| = {abs(residual):12.6e})")
    
    elif equation == 28:
        if verbose:
            print("Equation 28 residuals for each (m,n) pair:")
        
        for m in range(N):
            for n in range(N):
                residual = equation_28_residual(lambda_full, m, n, N)
                residuals.append(residual)
                max_residual = max(max_residual, abs(residual))
                
                if verbose:
                    print(f"  (m={m}, n={n}): residual = {residual:12.6e} (|residual| = {abs(residual):12.6e})")
    
    else:
        raise ValueError(f"Unsupported equation number: {equation}")
    
    is_valid = max_residual < tolerance
    mean_residual = np.mean([abs(r) for r in residuals])
    
    if verbose:
        print(f"\nSummary:")
        print(f"  Equation:        {equation}")
        print(f"  Max |residual|:  {max_residual:.2e}")
        print(f"  Mean |residual|: {mean_residual:.2e}")
        print(f"  Tolerance:       {tolerance:.2e}")
        print(f"  Valid solution:  {is_valid}")
        print("=" * 50)
    
    return {
        'is_valid': is_valid,
        'max_residual': max_residual,
        'mean_residual': mean_residual,
        'residuals': residuals,
        'tolerance': tolerance
    }


def print_equation(equation_num):
    """Print the equation being verified."""
    if equation_num == 27:
        print("EQUATION 27 (Complex Hadamard Matrix Condition)")
        print("=" * 60)
        print("For each d = 0, 1, 2, ..., N-1:")
        print()
        print("    0 = Σ_{j=1}^{N-1} Σ_{k=j}^{N-1} cos(2π j d/N + λ_k - λ_{k-j})")
        print()
        print("Where:")
        print("  • N = QFT dimension")
        print("  • d = difference parameter (0 ≤ d ≤ N-1)")
        print("  • j, k = summation indices")
        print("  • λ_k = phase parameters with symmetry λ_k = λ_{N-k}")
        print()
        print("This gives N equations that must be simultaneously satisfied")
        print("for the matrix to be a complex Hadamard matrix.")
        
    elif equation_num == 28:
        print("EQUATION 28 (QFT-Specific Condition)")
        print("=" * 60)
        print("For each m,n = 0, 1, 2, ..., N-1:")
        print()
        print("    0 = Σ_{j,k,l,p=0}^{N-1} sin((k-l)2πn/N + (j-k)2πm/N + λ_j - λ_k - λ_l - λ_p)")
        print()
        print("Where:")
        print("  • N = QFT dimension")
        print("  • m, n = QFT indices (0 ≤ m,n ≤ N-1)")
        print("  • j, k, l, p = summation indices")
        print("  • λ_j = phase parameters")
        print()
        print("This gives N² equations that ensure the core submatrix")
        print("matches the standard QFT.")
    
    print("=" * 60)
    print()


def load_and_verify_json(filepath, equation=27, tolerance=1e-10, 
                        verify_all_runs=False, brief=False):
    """
    Load JSON file and verify all solutions.
    
    Args:
        filepath: Path to JSON file with results
        equation: Which equation to verify (27 or 28)
        tolerance: Numerical tolerance for verification
        verify_all_runs: Whether to verify each individual run in multi-run results
        brief: Whether to show only summary without detailed output
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return
    
    # Print equation
    print_equation(equation)
    
    # Get N from data
    N = data.get('N')
    if N is None:
        print("Error: N not found in JSON data")
        return
    
    print(f"VERIFICATION RESULTS FOR N = {N}, EQUATION {equation}")
    print("=" * 60)
    
    # Check if we have results from multiple methods
    if 'results' in data:
        # Format from equation 27: multiple methods
        methods_data = data['results']
    elif 'equation' in data and data['equation'] == 28:
        # Updated format from equation 28: single result in flat structure
        methods_data = {data.get('method', 'unknown_method'): data}
    elif 'result' in data and 'lambda_full' in data['result']:
        # Old format from equation 28: single result in 'result' key
        methods_data = {data.get('method', 'unknown_method'): data['result']}
    else:
        # Legacy single result format (equation 27 single run)
        methods_data = {'single_result': data}
    
    all_valid = True
    method_count = 0
    total_solutions = 0
    
    for method_name, result in methods_data.items():
        if isinstance(result, dict) and 'lambda_full' in result:
            method_count += 1
            
            print(f"\nMETHOD: {method_name.upper()}")
            print("-" * 40)
            
            # Check if this is a multi-run result
            is_multi_run = result.get('is_multi_run', False)
            
            if is_multi_run:
                print(f"Multi-run result: {result['total_runs']} runs, {result['unique_solutions']} unique solutions")
                print(f"Main solution (from run 1):")
                print()
                
            # Basic info about the main optimization result
            if 'objective_value' in result:
                print(f"Optimization objective value: {result['objective_value']:.2e}")
            if 'success' in result:
                print(f"Optimization success: {result['success']}")
            if 'seed_used' in result:
                print(f"Seed used: {result['seed_used']}")
            print()
            
            # Verify the main solution
            lambda_full = np.array(result['lambda_full'])
            verification = verify_solution(lambda_full, N, equation, tolerance, verbose=not brief)
            
            if not verification['is_valid']:
                all_valid = False
            
            total_solutions += 1
            
            # If multi-run, show summary of all runs
            if is_multi_run and 'all_runs' in result:
                print(f"\nALL RUNS SUMMARY:")
                print("-" * 25)
                
                unique_count = 0
                duplicate_count = 0
                error_count = 0
                verified_count = 0
                
                for run in result['all_runs']:
                    if 'error' in run:
                        error_count += 1
                        continue
                    
                    # Check if this run has verification info
                    run_verification = run.get('verification', {})
                    is_run_valid = run_verification.get('is_valid', False)
                    run_max_residual = run_verification.get('max_residual', float('inf'))
                    
                    if is_run_valid:
                        verified_count += 1
                        verification_status = f"✅ valid (residual: {run_max_residual:.1e})"
                    else:
                        verification_status = f"❌ invalid (residual: {run_max_residual:.1e})" if run_max_residual != float('inf') else "❌ not verified"
                    
                    if 'duplicate_of_run' in run:
                        duplicate_count += 1
                        status = f"(duplicate of run {run['duplicate_of_run']}) - {verification_status}"
                    else:
                        unique_count += 1
                        status = f"(unique) - {verification_status}"
                        total_solutions += 1
                    
                    print(f"  Run {run['run_number']:2d} (seed {run['seed_used']:5d}): "
                          f"obj = {run['objective_value']:.2e} {status}")
                
                print(f"\nRun statistics:")
                print(f"  Total runs: {result.get('total_runs', len(result['all_runs']))}")
                print(f"  Successful runs: {result.get('successful_runs', len(result['all_runs']) - error_count)}")
                print(f"  Verified runs: {verified_count}/{result.get('verified_runs', '?')}")
                print(f"  Unique solutions: {unique_count}")
                print(f"  Duplicates: {duplicate_count}")
                print(f"  Errors: {error_count}")
                print(f"  Verification rate: {100*verified_count/(len(result['all_runs']) - error_count):.1f}% of successful runs")
                
                # Optionally verify all individual runs
                if verify_all_runs:
                    print(f"\nDETAILED VERIFICATION OF ALL RUNS:")
                    print("-" * 50)
                    
                    for i, run in enumerate(result['all_runs']):
                        if 'error' in run:
                            print(f"\nRun {run['run_number']} (seed {run['seed_used']}): ERROR - {run['error']}")
                            continue
                        
                        print(f"\nRun {run['run_number']} (seed {run['seed_used']}):")
                        print("-" * 25)
                        
                        lambda_full = np.array(run['lambda_full'])
                        run_verification = verify_solution(lambda_full, N, equation, tolerance, verbose=not brief)
                        
                        # Compare with stored verification (if available)
                        stored_verification = run.get('verification', {})
                        if stored_verification:
                            stored_valid = stored_verification.get('is_valid', False)
                            stored_residual = stored_verification.get('max_residual', float('inf'))
                            
                            if abs(run_verification['max_residual'] - stored_residual) < 1e-14 and run_verification['is_valid'] == stored_valid:
                                print(f"✅ Verification matches stored result")
                            else:
                                print(f"⚠️  Verification differs from stored:")
                                print(f"   Stored: valid={stored_valid}, residual={stored_residual:.2e}")
                                print(f"   Computed: valid={run_verification['is_valid']}, residual={run_verification['max_residual']:.2e}")
                        
                        if not run_verification['is_valid']:
                            all_valid = False
    
    # Check standard QFT if available
    if 'standard_qft' in data and 'lambda_full' in data['standard_qft']:
        print(f"\nSTANDARD QFT REFERENCE")
        print("-" * 30)
        qft_lambda = np.array(data['standard_qft']['lambda_full'])
        qft_verification = verify_solution(qft_lambda, N, equation, tolerance, verbose=not brief)
    
    # Final summary
    print(f"\nFINAL SUMMARY")
    print("=" * 40)
    print(f"Methods verified: {method_count}")
    print(f"Total solutions verified: {total_solutions}")
    print(f"All solutions valid: {all_valid}")
    print(f"Tolerance used: {tolerance:.2e}")
    
    if all_valid and method_count > 0:
        print("✅ All solutions successfully verified!")
    else:
        print("❌ Some solutions failed verification")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Verify QFT equation solutions from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_equation.py eq27_results_n4.json --equation 27
  python verify_equation.py eq28_results_n4.json --equation 28 --tolerance 1e-12
  python verify_equation.py eq27_results_n7.json --equation 27 --verify-all-runs
  python verify_equation.py eq27_results_n7.json --equation 27 --brief
  
Equation 27 (Complex Hadamard): N equations for d = 0, 1, ..., N-1
Equation 28 (QFT-specific): N² equations for m,n = 0, 1, ..., N-1

Options:
  --verify-all-runs: Individually verify each solution in multi-run results
  --brief: Show only summary statistics without detailed verification output
        """
    )
    
    parser.add_argument('json_file', type=str, 
                       help='JSON file containing equation results')
    parser.add_argument('--equation', '-e', type=int, choices=[27, 28], required=True,
                       help='Which equation to verify: 27 (Complex Hadamard) or 28 (QFT-specific)')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-10,
                       help='Numerical tolerance for verification (default: 1e-10)')
    parser.add_argument('--verify-all-runs', action='store_true',
                       help='Verify each individual solution in multi-run results (detailed output)')
    parser.add_argument('--brief', action='store_true',
                       help='Show only summary statistics without detailed verification output')
    
    args = parser.parse_args()
    
    # Verify the solutions
    load_and_verify_json(args.json_file, args.equation, args.tolerance, 
                        verify_all_runs=args.verify_all_runs, brief=args.brief)


if __name__ == "__main__":
    main()
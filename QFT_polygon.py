"""
Quantum Fourier Transform (QFT) Polygon Analysis Script
=======================================================

This script performs comprehensive numerical analysis of quantum Fourier transforms
using polygon-like graph structures with different dimensions (3D, 5D, 6D).

OVERVIEW:
---------
The main goal is to find graph Hamiltonians (coupling matrices) and evolution times
that produce unitary matrices with both:
1. Uniform amplitude distribution (|U_ij|² = 1/N for NxN matrices)
2. Phase patterns matching the discrete Fourier transform (DFT)

METHODOLOGY:
------------
For each polygon dimension (triangle, pentagon, hexagon):
1. Define a coupling matrix representing connections between vertices
2. Use matrix exponential U = exp(-iCt) for time evolution
3. Optimize parameters to minimize deviations from uniform amplitudes
4. Verify phase patterns match those of the corresponding DFT matrix
5. Apply phase corrections to align with standard DFT structure

GRAPH STRUCTURES:
-----------------
- 3x3 (Triangle): Modified triangle with additional self-coupling
- 5x5 (Pentagon): Pentagon with nearest and next-nearest neighbor couplings
- 6x6 (Hexagon): Hexagon with nearest, next-nearest, and opposite vertex couplings

OPTIMIZATION APPROACH:
----------------------
- Uses scipy.optimize.least_squares with trust-region-reflective method
- Dual criteria: amplitude uniformity AND phase pattern matching
- Extensive random initialization for robust parameter exploration
- High precision tolerances (1e-14) for accurate solutions

APPLICATIONS:
-------------
This analysis is relevant for:
- Digital quantum simulation of Fourier transforms
- Understanding quantum walk dynamics on polygon graphs  
- Designing quantum circuits for efficient QFT implementation
- Studying the relationship between graph topology and quantum transforms
"""

import numpy as np
import math
import cmath
import time
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex, Image

# Scientific computing libraries
from scipy.linalg import eigh          # Eigenvalue decomposition
from scipy.linalg import expm, dft, logm  # Matrix exponential, discrete Fourier transform, matrix logarithm
from scipy.optimize import least_squares  # Non-linear least squares optimization
from scipy.optimize import minimize      # General optimization


# Configuration and Setup
# =======================

# Time evolution parameter range for analysis
length = np.linspace(0.0, 5.0, 10001)  # 10001 points from 0 to 5

# NumPy output formatting for better precision display
np.set_printoptions(linewidth=200)     # Wider output lines
np.set_printoptions(precision=17)      # High precision for numerical accuracy
np.set_printoptions(legacy='1.25')    # Legacy formatting
# np.set_printoptions(suppress=True, precision=8)  # Alternative formatting option

# Initial matrix dimension test
# =============================
M = 5  # Matrix dimension for initial analysis

# Print difference matrix (m-n) for understanding index relationships
# This helps visualize the cyclic nature of polygon structures
for m in range(M):
    print("m-n=", end='')
    for n in range(M):
        print(f'{m-n}, ', end='')
    print("\n")


# Test Case: 3x3 Triangle Graph Evolution
# =======================================
x = 2.0 * np.pi / 9.0  # Coupling parameter

# Create unitary evolution operator using matrix exponential
# Triangle graph structure: each vertex connected to the other two
U = expm(-1j * np.array([[0, x, x], [x, 0, x], [x, x, 0]]))

# Output reference value (1/sqrt(3)) and matrix properties
print(1/np.sqrt(3.0), "\n", np.abs(U), "\n")  # Magnitude comparison
print(np.angle(U), "\n")  # Phase angles

# 3x3 System Optimization Function
# =================================

def F3(vec):
    """
    Objective function for 3x3 quantum system optimization.
    
    Parameters:
    - vec: [t, C12, C23, C11] where:
      - t: evolution time parameter
      - C12, C23, C11: coupling strengths between different vertices
    
    Returns:
    - Array of squared amplitude deviations from uniform distribution (1/3)
    """
    t, C12, C23, C11 = vec
    
    # Construct coupling matrix with specific graph structure
    # This represents a modified triangle with additional self-coupling
    C = np.array([
        [ 0 , C12,  0 ],   # Vertex 0 connects to vertex 1
        [C12, C11, C23],   # Vertex 1 connects to vertices 0, 2 and has self-coupling
        [ 0 , C23,  0 ]    # Vertex 2 connects to vertex 1
    ], dtype=complex)
    
    # Diagonalize the coupling matrix
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Compute time evolution operator and extract squared amplitudes
    exp_C = np.abs(eigenvectors @ np.diag(np.exp(-1j*eigenvalues * t)) @ eigenvectors.conj().T)**2 - 1/3
    
    # Return flattened upper triangular elements (objective function components)
    return np.array([exp_C[0,0], exp_C[0,1], exp_C[0,2], exp_C[1,1], exp_C[1,2], exp_C[2,2]])

# Optimization Loop for 3x3 System
# ==================================
# Search for optimal parameters that minimize the objective function

for n in range(100):  # Maximum 100 optimization attempts
    # Generate random initial guess for parameters
    guess = np.random.uniform(low=0.01, high=3, size=4)
    guess[0] = 1  # Fix time parameter initial guess
    
    # Run trust-region optimization with tight tolerances
    result = least_squares(
        fun=F3, 
        x0=guess, 
        method='trf',           # Trust Region Reflective algorithm
        bounds=(1e-2, 4.15),    # Parameter bounds
        ftol=1e-14,             # Function tolerance
        xtol=1e-14,             # Parameter tolerance  
        gtol=1e-14,             # Gradient tolerance
        max_nfev=15000          # Maximum function evaluations
    )
    
    # Display optimization progress
    print(f'step {n},   Opt->1e{int(math.floor(math.log10(abs(result.optimality))))}')
    
    # Check for successful convergence with high precision
    if result.success and result.optimality < 1e-14:
        print("Solution =", list(result.x))
        print("Optimality=", result.optimality)
        print("guess=", list(guess))


# Previous optimization results (for reference)
# =============================================
Solution_old = [1.3002771389148073, 1.3750843651865545, 0.8383328524485424, 1.6705325892254657]
print(np.dot(F3(Solution_old), F3(Solution_old)))  # Verify solution quality

# Current optimal solution for 3x3 system
# ========================================
Solution = [1.1883107640491744, 1.0900477253633614, 1.0900477253633618, 1.4347679636892618]
# Previous initial guess: [1.0, 1.5522069706541315, 0.7043412741762702, 1.7681882115922838]

# Alternative parameterization (commented out):
# print(fr"\theta={1.1883107640491744*1.0900477253633614}", rf"\beta={1.1883107640491744*1.4347679636892618}")

# Extract optimized parameters
t0, C12, C23, C22 = Solution
# Construct optimized coupling matrix
C = np.array([
    [ 0 , C12,  0 ],    # Coupling structure from optimization
    [C12, C22, C23],
    [ 0 , C23,  0 ]
])

# Compute eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Time evolution at optimal time t0
exp_C = eigenvectors @ np.diag(np.exp(-1j*eigenvalues * t0)) @ eigenvectors.conj().T

# Phase correction to match DFT structure
# ======================================
# Input phase correction: align first row phases
fase_in = np.diag(np.exp(-1j * np.angle(exp_C[0, :])))

# Output phase correction: align first column phases to first element
fase_out = np.diag(np.exp(1j * (np.angle(exp_C[0, 0]) - np.angle(exp_C[:, 0]))))

# Apply phase corrections
exp_C = fase_out @ exp_C @ fase_in

# Compare with discrete Fourier transform matrix
print(np.abs(exp_C), "\n")              # Amplitudes of our matrix
print(np.angle(exp_C), "\n \n", np.angle(dft(3)), "\n")  # Phase comparison with DFT

# Display phase correction factors
print("fase in : ", np.angle(fase_in[0,0]), np.angle(fase_in[1,1]), np.angle(fase_in[2,2]))
print("fase out: ", np.angle(fase_out[0,0]), np.angle(fase_out[1,1]), np.angle(fase_out[2,2]))

# Time Evolution Analysis
# ========================
# Compute matrix elements as functions of time parameter

# Initialize amplitude arrays for all matrix elements
A_11, A_12, A_13 = [], [], []  # First row elements
A_22, A_23 = [], []            # Second row elements (diagonal and off-diagonal)
A_33 = []                      # Third row diagonal element

A11, A12, A22, A23 = [], [], [], []  # Alternative naming (unused)

# Compute evolution for each time point
for t in length:
    # Calculate time evolution operator at time t
    exp_C = eigenvectors @ np.diag(np.exp(-1j*eigenvalues*t)) @ eigenvectors.conj().T
    
    # Store absolute values of matrix elements
    A_11.append(np.abs(exp_C[0,0]))  # |A_00(t)|
    A_12.append(np.abs(exp_C[0,1]))  # |A_01(t)|
    A_13.append(np.abs(exp_C[0,2]))  # |A_02(t)|

    A_22.append(np.abs(exp_C[1,1]))  # |A_11(t)|
    A_23.append(np.abs(exp_C[1,2]))  # |A_12(t)|

    A_33.append(np.abs(exp_C[2,2]))  # |A_22(t)|


# Plotting Configuration and Visualization
# =========================================

# Configure matplotlib for publication-quality plots
plt.rcParams['figure.figsize'] = [4, 4]                          # Square figure
plt.rcParams['text.usetex'] = True                               # Enable LaTeX rendering
plt.rcParams['axes.unicode_minus'] = False                       # Fix minus sign rendering
plt.rcParams['text.latex.preamble'] = r'\usepackage{braket}'     # Braket notation package
plt.rcParams['font.family'] = 'sans-serif'                       # Font family

# Alternative plots (commented out for comparison):
# plt.plot(length,A11,color='yellow',lw=4.5,label='$|A_{11}|$')
# plt.plot(length,A12,color='yellow',lw=4.5,label='$|A_{12}|$')
# plt.plot(length,A22,color='yellow',ls='dashed',lw=4.5,label='$|A_{22}|$')
# plt.plot(length,A23,color='yellow',ls='dashed',lw=4.5,label='$|A_{23}|$')

# Plot matrix element amplitudes vs time
plt.plot(length, A_33, color='lime', lw=1.5, label='$|A_{33}|$')          # Diagonal element (3,3)
plt.plot(length, A_23, color='blueviolet', lw=1.5, label='$|A_{23}|$')    # Off-diagonal (2,3)
plt.plot(length, A_22, color='cyan', ls='dashed', lw=1.0, label='$|A_{22}|$')  # Diagonal (2,2)

plt.plot(length, A_13, color='crimson', ls='dashed', lw=1.5, label='$|A_{13}|$')  # Off-diagonal (1,3)
plt.plot(length, A_12, color='crimson', lw=1.5, label='$|A_{12}|$')              # Off-diagonal (1,2)
plt.plot(length, A_11, color='black', lw=1.5, label='$|A_{11}|$')                # Diagonal (1,1)

# Reference lines
plt.axhline(y=1.0/np.sqrt(3.0), color='black', ls='dashed', lw=0.6)  # Uniform distribution level
plt.axvline(x=t0, color='black', lw=0.6, ls='dashed')                # Optimal time

# Axis labels and formatting
plt.xlabel(r"$ z$", fontsize=25, labelpad=15)        # Time parameter
plt.ylabel('$|A_{j,k}|$ ', fontsize=25, labelpad=15)  # Matrix element amplitudes
plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0], fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0, 2])  # Focus on relevant time range

# Tick formatting
plt.tick_params(axis='x', direction='out', length=10, width=1.4)
plt.tick_params(axis='y', direction='out', length=10, width=1.4)

# Optional text annotations (commented out):
# plt.text(0.64,0.85,s=r"$ t_0 = \pi/4$",ha="center", va="center",fontsize=17)
# plt.text(0.25,0.63,s=r"$ 1/\sqrt{3} $",ha="center", va="center",fontsize=17)

# Legend configuration
plt.legend(loc='lower center', fontsize=17, handlelength=1.25, handletextpad=0.9, 
          labelspacing=0.25, ncol=5, markerscale=1.0, bbox_to_anchor=(0.5, 1), frameon=False)

# Save option (commented out):
# plt.savefig('CHT4_evolution2.png', bbox_inches='tight', pad_inches=0)

plt.show()

# 5x5 Pentagon System Analysis Functions
# ======================================

def Angles_F5(vec):
    """
    Phase angle comparison function for 5x5 pentagon system.
    
    Compares the phase angles of the evolved matrix with those of the 5-point DFT.
    This function checks how well our pentagon graph evolution matches the QFT structure.
    
    Parameters:
    - vec: [t, C12, C13] where:
      - t: evolution time parameter
      - C12: nearest-neighbor coupling strength
      - C13: next-nearest-neighbor coupling strength
    
    Returns:
    - Norm of difference between sorted phase angles and DFT phase angles
    """
    t, C12, C13 = vec
    
    # Pentagon coupling matrix (5-vertex cycle with next-nearest neighbor connections)
    C = np.array([
        [ 0 , C12, C13, C13, C12],  # Vertex 0: connects to vertices 1,4 (nearest) and 2,3 (next-nearest)
        [C12,  0 , C12, C13, C13],  # Vertex 1: connects to vertices 0,2 (nearest) and 3,4 (next-nearest)
        [C13, C12,  0 , C12, C13],  # Vertex 2: connects to vertices 1,3 (nearest) and 0,4 (next-nearest)
        [C13, C13, C12,  0 , C12],  # Vertex 3: connects to vertices 2,4 (nearest) and 0,1 (next-nearest)
        [C12, C13, C13, C12,  0 ]   # Vertex 4: connects to vertices 0,3 (nearest) and 1,2 (next-nearest)
    ], dtype=complex)
    
    M = C.shape[0]  # Matrix dimension (5)
    
    # Eigendecomposition and time evolution
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    exp_C = eigenvectors @ np.diag(np.exp(-1j*eigenvalues * t)) @ eigenvectors.conj().T
    
    # Apply phase corrections to match DFT structure
    fase_in = np.diag(np.exp(-1j * np.angle(exp_C[0, :])))
    fase_out = np.diag(np.exp(1j * (np.angle(exp_C[0, 0]) - np.angle(exp_C[:, 0]))))
    exp_C = fase_out @ exp_C @ fase_in
    
    # Compare phase angles with 5-point DFT (focus on second column)
    return np.linalg.norm(np.sort(np.abs(np.angle(exp_C)[:,1])) - np.sort(np.abs(np.angle(dft(M))[:,1])))


def F5(vec):
    """
    Objective function for 5x5 pentagon system optimization.
    
    Optimizes for uniform amplitude distribution (1/5) across matrix elements.
    This is the main objective function for finding pentagon parameters that 
    produce quantum Fourier transform-like behavior.
    
    Parameters:
    - vec: [t, C12, C13] parameter vector
    
    Returns:
    - Array of squared amplitude deviations from uniform distribution (1/5)
    """
    t, C12, C13 = vec
    
    # Same pentagon coupling matrix as in Angles_F5
    C = np.array([
        [ 0 , C12, C13, C13, C12],
        [C12,  0 , C12, C13, C13],
        [C13, C12,  0 , C12, C13],
        [C13, C13, C12,  0 , C12],
        [C12, C13, C13, C12,  0 ]
    ], dtype=complex)
    
    # Compute time evolution and extract amplitude deviations
    eigenvalues, eigenvectors = np.linalg.eigh(C) 
    exp_C = np.abs(eigenvectors @ np.diag(np.exp(-1j*eigenvalues * t)) @ eigenvectors.conj().T)**2 - 1.0/5.0
    
    # Return key matrix elements for optimization (first row elements)
    return np.array([exp_C[0,0], exp_C[0,1], exp_C[0,2]])


# Optimization Loop for 5x5 Pentagon System
# ==========================================
# Search for parameters that produce QFT-like behavior in pentagon graph

for n in range(100):  # Maximum 100 optimization attempts
    # Generate random initial guess for 3 parameters
    guess = np.random.uniform(low=0.01, high=3, size=3)
    guess[0] = 1  # Fix time parameter initial guess
    
    # Run optimization with very tight bounds and tolerances
    result = least_squares(
        fun=F5, 
        x0=guess, 
        method='trf',
        bounds=(1e-10, 4.15),   # Very tight lower bound, reasonable upper bound
        ftol=1e-14, 
        xtol=1e-14, 
        gtol=1e-14, 
        max_nfev=15000
    )
    
    # Display progress with both amplitude and phase accuracy
    print(f'step {n},   Opt->1e{int(math.floor(math.log10(abs(result.optimality))))}', Angles_F5(result.x))
    
    # Check for successful convergence in both amplitude and phase
    if result.success and result.optimality < 1e-14 and Angles_F5(result.x) < 1e-6:
        print("Solution =", list(result.x))
        print("Optimality=", result.optimality)
        print("guess=", list(guess))
        print("Angle=", Angles_F5(result.x), "\n")
        
        # Analyze coupling strength ratios
        Bla = [result.x[1], result.x[2]]  # Extract coupling parameters
        Bla = np.sort(Bla)                # Sort for consistent ratio calculation
        print(Bla[1]/Bla[0])              # Print ratio of larger to smaller coupling


# Analysis with Optimal 5x5 Pentagon Parameters
# ==============================================

# Previous optimization results (commented out for reference):
# t0, C12, C13 = [0.9825098219042625,  2.776494747494319,1.0605266239572728]
# t0,C12,C13 =[1.0,np.pi*(5+np.sqrt(5))/25,np.pi*(5-np.sqrt(5))/25]  # Golden ratio related

# Current optimal solution for 5x5 pentagon system
t0, C12, C13 = [0.9156063846520861, 2.6000336589035706, 0.1448949097485951]

# Construct optimal pentagon coupling matrix
C = np.array([
    [ 0 , C12, C13, C13, C12],
    [C12,  0 , C12, C13, C13],
    [C13, C12,  0 , C12, C13],
    [C13, C13, C12,  0 , C12],
    [C12, C13, C13, C12,  0 ]
])

# Compute eigendecomposition and time evolution
eigenvalues, eigenvectors = np.linalg.eigh(C) 
exp_C = eigenvectors @ np.diag(np.exp(-1j*eigenvalues * t0)) @ eigenvectors.conj().T

# Apply phase corrections
fase_in = np.diag(np.exp(-1j * np.angle(exp_C[0, :])))
fase_out = np.diag(np.exp(1j * (np.angle(exp_C[0, 0]) - np.angle(exp_C[:, 0]))))
exp_C = fase_out @ exp_C @ fase_in

# Verification: Compare with theoretical uniform distribution and 5-point DFT
print(f"1/sqrt(5) = {1/np.sqrt(5)}", "\t", np.abs(exp_C[3,3]), "\n")  # Check uniform amplitude
print(np.angle(exp_C), "\n")        # Our matrix phases
print(np.angle(dft(5)), "\n")       # 5-point DFT phases for comparison

# Display phase correction factors for all 5 vertices
print("fase in : ", np.angle(fase_in[0,0]), np.angle(fase_in[1,1]), np.angle(fase_in[2,2]), 
      np.angle(fase_in[3,3]), np.angle(fase_in[4,4]), "\n")
print("fase out: ", np.angle(fase_out[0,0]), np.angle(fase_out[1,1]), np.angle(fase_out[2,2]), 
      np.angle(fase_out[3,3]), np.angle(fase_out[4,4]), "\n")




# 6x6 Hexagon System Analysis Functions
# ======================================

def Angles_F6(vec):
    """
    Phase angle comparison function for 6x6 hexagon system.
    
    Compares phase angle patterns across all columns of the evolved matrix
    with those of the 6-point DFT. This provides a comprehensive check of
    how well the hexagon graph evolution matches QFT structure.
    
    Parameters:
    - vec: [t, C12, C13, C14] where:
      - t: evolution time parameter
      - C12: nearest-neighbor coupling strength
      - C13: next-nearest-neighbor coupling strength  
      - C14: opposite vertex coupling strength (across hexagon)
    
    Returns:
    - Squared norm of difference between DFT and evolved matrix phase patterns
    """
    t, C12, C13, C14 = vec
    
    # Hexagon coupling matrix (6-vertex cycle with multiple neighbor interactions)
    C = np.array([
        [ 0 , C12, C13, C14, C13, C12],  # Vertex 0: nearest (1,5), next-nearest (2,4), opposite (3)
        [C12,  0 , C12, C13, C14, C13],  # Vertex 1: nearest (0,2), next-nearest (3,5), opposite (4)
        [C13, C12,  0 , C12, C13, C14],  # Vertex 2: nearest (1,3), next-nearest (0,4), opposite (5)
        [C14, C13, C12,  0 , C12, C13],  # Vertex 3: nearest (2,4), next-nearest (1,5), opposite (0)
        [C13, C14, C13, C12,  0 , C12],  # Vertex 4: nearest (3,5), next-nearest (0,2), opposite (1)
        [C12, C13, C14, C13, C12,  0 ]   # Vertex 5: nearest (0,4), next-nearest (1,3), opposite (2)
    ], dtype=complex)

    # Eigendecomposition and time evolution
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    exp_C = eigenvectors @ np.diag(np.exp(-1j*eigenvalues * t)) @ eigenvectors.conj().T
    
    # Apply phase corrections
    fase_in = np.diag(np.exp(-1j * np.angle(exp_C[0, :])))
    fase_out = np.diag(np.exp(1j * (np.angle(exp_C[0, 0]) - np.angle(exp_C[:, 0]))))
    exp_C = fase_out @ exp_C @ fase_in
    
    # Compare phase patterns across all columns (excluding first column)
    # Compute norms of sorted absolute phase angles for each column
    angles_dft = np.array([
        np.linalg.norm(np.sort(np.abs(np.angle(dft(6))[:,1]))),  # DFT column 1
        np.linalg.norm(np.sort(np.abs(np.angle(dft(6))[:,2]))),  # DFT column 2
        np.linalg.norm(np.sort(np.abs(np.angle(dft(6))[:,3]))),  # DFT column 3
        np.linalg.norm(np.sort(np.abs(np.angle(dft(6))[:,4]))),  # DFT column 4
        np.linalg.norm(np.sort(np.abs(np.angle(dft(6))[:,5])))   # DFT column 5
    ])
    
    angles_exp = np.array([
        np.linalg.norm(np.sort(np.abs(np.angle(exp_C)[:,1]))),   # Our matrix column 1
        np.linalg.norm(np.sort(np.abs(np.angle(exp_C)[:,2]))),   # Our matrix column 2
        np.linalg.norm(np.sort(np.abs(np.angle(exp_C)[:,3]))),   # Our matrix column 3
        np.linalg.norm(np.sort(np.abs(np.angle(exp_C)[:,4]))),   # Our matrix column 4
        np.linalg.norm(np.sort(np.abs(np.angle(exp_C)[:,5])))    # Our matrix column 5
    ])

    # Return squared error between phase patterns
    return np.dot(angles_dft - angles_exp, angles_dft - angles_exp)


def F6(vec):
    """
    Objective function for 6x6 hexagon system optimization.
    
    Optimizes for uniform amplitude distribution (1/6) across matrix elements.
    This is the main objective function for finding hexagon parameters that 
    produce quantum Fourier transform-like behavior.
    
    Parameters:
    - vec: [t, C12, C13, C14] parameter vector
    
    Returns:
    - Array of squared amplitude deviations from uniform distribution (1/6)
    """
    t, C12, C13, C14 = vec
    
    # Same hexagon coupling matrix as in Angles_F6
    C = np.array([
        [ 0 , C12, C13, C14, C13, C12],
        [C12,  0 , C12, C13, C14, C13],
        [C13, C12,  0 , C12, C13, C14],
        [C14, C13, C12,  0 , C12, C13],
        [C13, C14, C13, C12,  0 , C12],
        [C12, C13, C14, C13, C12,  0 ]
    ], dtype=complex)
    
    # Compute time evolution and extract amplitude deviations
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    exp_C = np.abs(eigenvectors @ np.diag(np.exp(-1j*eigenvalues * t)) @ eigenvectors.conj().T)**2 - 1/6
    
    # Return key matrix elements for optimization (first row elements)
    return np.array([exp_C[0,0], exp_C[0,1], exp_C[0,2]])

# Extensive Optimization Search for 6x6 Hexagon System
# =====================================================
# Large-scale search for optimal hexagon parameters

# Arrays for collecting statistical data on solutions (currently unused)
z_length, ratio12, ratio23, ratio13 = [], [], [], []

for n in range(5000):  # Extensive search with 5000 attempts
    # Generate random initial guess for 4 parameters
    guess = np.random.uniform(low=0.01, high=4, size=4)
    # Note: time parameter is not fixed (unlike 3x3 and 5x5 cases)
    
    # Run optimization with very wide bounds for thorough exploration
    result = least_squares(
        fun=F6, 
        x0=guess, 
        method='trf',
        bounds=(1e-12, 150.15),  # Very wide bounds for exploration
        ftol=1e-14, 
        xtol=1e-14, 
        gtol=1e-14, 
        max_nfev=15000
    )
    
    # Progress monitoring (commented out for cleaner output during extensive search)
    # print(f'step {n},   Opt->1e{int(math.floor(math.log10(abs(result.optimality))))}', Angles_F6(result.x))
    
    # Check for successful convergence in both amplitude and phase
    if result.success and result.optimality < 1e-14 and Angles_F6(result.x) < 1e-6:
        print("Solution =", list(result.x))
        # print("Optimality=", result.optimality)  # Commented out for cleaner output
        
        # Analyze coupling strength relationships
        couplings = np.sort(result.x[1:])  # Sort coupling parameters (smallest to largest)
        print(f'C3/C1={couplings[0]/couplings[2]},   C2/C1={couplings[0]/couplings[1]}')
        print("guess=", list(guess))
        print("Angle=", Angles_F6(result.x), "\n")
        
        # Statistical data collection (currently commented out):
        # couplings = np.sort(result.x[1:])
        # z_length.append(result.x[0])           # Evolution time
        # ratio12.append(couplings[0]/couplings[1])  # Ratio of smallest to middle coupling
        # ratio23.append(couplings[1]/couplings[2])  # Ratio of middle to largest coupling
        # ratio13.append(couplings[0]/couplings[2])  # Ratio of smallest to largest coupling
import numpy as np
import matplotlib.pyplot as plt
import time


def generate_linear_system(n):
  """
  Generates a linear system with a diagonally dominant matrix A and vector b.

  Args:
    n: Dimension of the system (size of the matrix A).

  Returns:
    A: Coefficient matrix (numpy array).
    b: Right-hand side vector (numpy array).
  """
  
  #définition de A : 
  A = ((-1)*np.ones((n, n)))
  for i in range(n):
    A[i,i]  = 5*(i+1)
    
  b = np.random.rand(n) #b est le membre de droite, généré aléatoirement
  print(A)

  return A, b

# Example usage:
n = 500  # Dimension of the system
A, b = generate_linear_system(n)


#METHODE AVEC LES DEUX SOMMES
def jacobi_method(A, b, x0, tol=1e-5, max_iter=1000):
  """
  Implements the Jacobi method for solving the linear system Ax = b.

  Args:
    A: Coefficient matrix (numpy array).
    b: Right-hand side vector (numpy array).
    x0: Initial guess for the solution vector (numpy array).
    tol: Tolerance for convergence.
    max_iter: Maximum number of iterations.

  Returns:
    x: Approximate solution vector.
    iterations: Number of iterations performed.
    errors: List of errors between exact and approximate solution at each iteration.
  """

  n = A.shape[0]

  x = x0.copy()
  errors = []
  start_time = time.time()

  #itérations 
  for i in range(max_iter):
    x_new = np.zeros_like(x)
    for j in range(n):
    
      L = np.sum(A[j, :j] * x[:j]) # Calcul de la somme des produits des éléments de la partie inférieure (L)

      U = np.sum(A[j, j+1:] * x[j+1:]) # Calcul de la somme des produits des éléments de la partie supérieure (U) 
      
      x_new[j] = (b[j] - L - U) / A[j, j]

    #calcul erreur
    error = np.linalg.norm(x_new - x)
    print(error)
    errors.append(error)

    x = x_new

    if error < tol:
        break
    
  end_time = time.time()
  time_taken = end_time - start_time
  print(f"Temps écoulé: {time_taken}") #On trouve Temps écoulé: 0.4586033821105957 Iterations: 450


  return x, i + 1, errors


def spectral_radius(A):
# Vérification du rayon spectral

  D = np.zeros_like(A)

  for i in range(n):
    D[i,i] = A[i,i]

  T = np.dot(np.linalg.inv(D),(A-D)) # Calcul de la matrice d'itération T 
  rho = max(abs(np.linalg.eigvals(T))) # Calcul du rayon spectral (valeur propre de plus grande norme de T)
  print("Le rayon spectral est égal à", rho)
  
  return rho


def diagonale_dominante(A):
# Vérifie si la matrice est de diagonale dominante
  n = A.shape[0]
  for i in range(n):
      # Somme des éléments hors-diagonale
      somme_hors_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
      
      if (abs(A[i, i]) < somme_hors_diag):
          print("La matrice n'est pas diagonale dominante.")
          return 

  print("La matrice est diagonale dominante.")
  return


def plot_error(errors, iterations):
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations (Jacobi Method)")
    plt.grid(True)
    plt.show()


# Example usage:
n = 100
A, b = generate_linear_system(n)  # Generate a linear system
x0 = np.zeros(np.size(b))

# Solve using Jacobi method
x_jacobi, iterations, errors = jacobi_method(A, b, x0)

#calculate spectral radius
rho=spectral_radius(A)

#diagonale dominante 
diagonale_dominante(A)

# Calculate exact solution
x_exact = np.linalg.solve(A, b)

# Print results
print(f"Iterations: {iterations}")
print(f"Solution Jacobi: {x_jacobi}")
print(f"Exact solution: {x_exact}")

# Plot the error
plot_error(errors, iterations)
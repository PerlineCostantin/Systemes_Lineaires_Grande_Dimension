import numpy as np
import matplotlib.pyplot as plt
import time


def generate_linear_system(n):
    """
    Génère un système linéaire avec une matrice A dominante par la diagonale et un vecteur b.

    Args:
        n: Dimension du système (taille de la matrice A).

    Retourne:
        A: Matrice des coefficients (numpy array).
        b: Vecteur du côté droit (numpy array).
    """
    
    #définition de A : 
    A = (-1) * np.ones((n, n))  
    for i in range(n):
        A[i, i] = 5 * (i + 1)  

    b = np.random.rand(n)  #b est le membre de droite, généré aléatoirement
    return A, b


#METHODE AVEC LA PREMIERE FORMULE (calcul de l'inverse de D)
def jacobi_method(A, b, x0, tol=1e-5, max_iter=1000):
    """
    Implémente la méthode de Jacobi pour résoudre le système linéaire Ax = b.

    Args:
        A: Matrice des coefficients (numpy array).
        b: Vecteur du côté droit (numpy array).
        x0: Estimation initiale du vecteur solution (numpy array).
        tol: Tolérance pour la convergence.
        max_iter: Nombre maximum d'itérations.

    Retourne:
        x: Vecteur solution approximatif.
        iterations: Nombre d'itérations effectuées.
        errors: Liste des erreurs entre la solution exacte et la solution approximative à chaque itération.
    """
    n = A.shape[0]
    x = x0.copy()
    errors = []
    start_time = time.time()

    # extraction de la diagonale de A
    D = np.diag(np.diagonal(A)) 

    # Calcul de L et U 
    LU = A - D  # car L + U = A - D

    # D_inv : calcul de l'inverse de D
    D_inv = np.linalg.inv(D) #on inverse D : compléxité élevée d'ordre O(n³)

    # D_inv * b (calcul intermédiaire)
    G = D_inv @ b  # multiplication matricielle
    print(f"G: {G}")

    # Itérations
    for i in range(max_iter):
        x_new = np.zeros_like(x)

        
        for j in range(n):
            sum_LU = 0
            for k in range(n):
                if k != j: #pour ne pas prendre en compte l'élément de la diagonale
                    sum_LU += LU[j, k] * x[k]  

            x_new[j] = G[j] - sum_LU / A[j, j] 

        error = np.linalg.norm(x_new - x)  # Calcul erreur
        errors.append(error)
        x = x_new

        if error < tol:
            break

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Temps écoulé: {time_taken}") #pour n=100, on trouve Temps écoulé: 1.025627613067627 Nombre d'itérations: 442

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
    """
    Trace l'erreur en fonction du nombre d'itérations.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # échelle logarithmique sur l'axe y
    plt.xlabel("Itérations")
    plt.ylabel("Erreur estimée")
    plt.title("Erreur vs Itérations (Méthode de Jacobi)")
    plt.grid(True)
    plt.show()


# Exemple d'utilisation

n = 100  # Dimension du système
A, b = generate_linear_system(n)  # Générer un système linéaire
x0 = np.zeros(np.size(b))  # Estimation initiale de x

# Résolution avec la méthode de Jacobi
x_jacobi, iterations, errors = jacobi_method(A, b, x0)

#calculate spectral radius
rho=spectral_radius(A)

#diagonale dominante 
diagonale_dominante(A)

# Calcul de la solution exacte (pour comparaison)
x_exact = np.linalg.solve(A, b)

# Affichage des résultats
print(f"Nombre d'itérations: {iterations}")
print(f"Solution Jacobi: {x_jacobi}")
print(f"Solution exacte: {x_exact}")

# Tracer l'erreur
plot_error(errors, iterations)


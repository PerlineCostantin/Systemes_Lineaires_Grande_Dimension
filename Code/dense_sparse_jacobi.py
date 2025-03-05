import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time


def generate_corrected_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=-1):
    """
    Generates a sparse tridiagonal matrix, ensuring no overlaps.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    
    main_diag = np.full(n, diagonal_value) #diagonale principale
    upp_diag= np.full(n-1,off_diagonal_value) #diagonale supérieure
    low_diag=np.full(n-1,off_diagonal_value)#diagonale inférieure


    # Construct sparse matrix
    data=np.concatenate((main_diag, upp_diag,low_diag ))
    rows = np.concatenate(( np.arange(n),np.arange(n-1),np.arange(1,n))) # might need to use np.concatenate
    cols = np.concatenate((np.arange(n), np.arange(1,n),np.arange(n-1)))
   
    As = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (np.abs(j - i) ==1) :
                A_dense[i,j] = off_diagonal_value
            elif i==j:
                A_dense[i, i] = diagonal_value
            
    b = np.random.rand(n)
    return As, A_dense, b



def jacobi_dense(A, b, x0, tol=1e-6, max_iter=1000):
    """
    Jacobi method for dense matrices.

    Args:
        A: Dense coefficient matrix (numpy array).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
    """
    
    n=A.shape[0]
    errors=[]
    x = x0.copy()
    start_time = time.time()
    
    for i in range(max_iter):
        x_new = np.zeros_like(x)
        for j in range(n):
            L = np.dot(A[j, :j], x[:j]) # pour calculer L, on fait la somme des produits A[j, :j] * x[:j], ie le produit scalaire entre la j-ème ligne de A avant l'élément sur la diagonale et les éléments correspondants de x

            U = np.dot(A[j, j+1:], x[j+1:]) # pour calculer U, on fait la somme des produits A[j, j+1:] * x[j+1:], ie le produit scalaire entre la j-ème ligne de A après l'élément sur la diagonale et les éléments correspondants de x
        
            x_new[j] = (b[j] - L - U) / A[j, j] 

        error = np.linalg.norm(x_new - x)
        print(error)
        errors.append(error)

        x = x_new

        if error < tol:
            break

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Temps écoulé: {time_taken}") 
    return x_new, i+1, time_taken



def jacobi_sparse(A, b, x0, tol=1e-6, max_iter=10000):
    """
    Jacobi method for sparse matrices.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
    """
    

    x = x0.copy()
    D_inv=1/A.diagonal()
    LU=A-sparse.diags(A.diagonal()) #-(L+U)
    
    start_time = time.time()
    for i in range(max_iter):
        x_new=D_inv*(b-LU.dot(x))
        
        error = np.linalg.norm(x_new - x)
        
        if error < tol:
            break
        x = x_new
        
    end_time = time.time()
    time_taken = end_time - start_time
    return x_new, i+1, time_taken



# small for loop comparing the times required for both approaches as a function of the dimension n

#dim=[1000]
dim = [1000, 2000, 3000, 5000, 10000]
results = []

for n in dim:
    x0=np.zeros(n,dtype=np.float64)
    A_sparse, A_dense, b = generate_corrected_sparse_tridiagonal_matrix(n)
    
    # Classical Jacobi (dense)
    x_dense, iter_dense, time_dense = jacobi_dense(A_dense, b, x0) 

    # Jacobi for sparse matrix
    x_sparse, iter_sparse, time_sparse = jacobi_sparse(A_sparse, b, x0)
    
    
    print(f"Iterations (dense): {iter_dense}, Time (dense): {time_dense:.4f} seconds")
    print(f"Iterations (sparse): {iter_sparse}, Time (sparse): {time_sparse:.4f} seconds")
    
    results.append((n, time_dense, time_sparse))
 
 

# results
print("n\tDense Time\tSparse Time")
for r in results:
    print(f"{r[0]}\t{r[1]}\t\t{r[2]}")
    

# Trace le temps d'exécution en fonction de la taille de la matrice
    n_val = [r[0] for r in results]
    dense_t = [r[1] for r in results]
    sparse_t = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(n_val, dense_t, label="Jacobi (Dense)", marker='o')
    plt.plot(n_val, sparse_t, label="Jacobi (Sparse)", marker='x')

    plt.xlabel('Taille de la matrice (n)')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Temps d\'exécution des méthodes Jacobi pour matrices dense et sparse')
    plt.legend()
    plt.grid(True)
    plt.show()


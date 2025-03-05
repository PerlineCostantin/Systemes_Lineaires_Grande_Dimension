import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time

def generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=2, off_diagonal_value=-1):
    """
    Generates a sparse tridiagonal matrix, ensuring no overlaps.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        A_dense: equivalent Dense matrix (numpy array)
        b: Right-hand side vector (numpy array).
    """
    
    main_diag = np.full(n, diagonal_value) #diagonale principale
    upp_diag= np.full(n-1,off_diagonal_value) #diagonale supérieure
    low_diag=np.full(n-1,off_diagonal_value) #diagonale inférieure
    
    # Construct sparse matrix
    data=np.concatenate((main_diag, upp_diag,low_diag ))
    #print(data)
    rows = np.concatenate(( np.arange(n),np.arange(n-1),np.arange(1,n))) # might need to use np.concatenate
    #print(rows)
    cols = np.concatenate((np.arange(n), np.arange(1,n),np.arange(n-1)))
    #print(cols)
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
    print(f"As: {As}") 
    print(f"A_dense: {A_dense}") 
    print(f"b: {b}") 
    
    return As, A_dense, b

def generate_sparse_tridiagonal_matrix(n):
    """
    Generates a sparse tridiagonal matrix with the specific values.

    Args:
        n: Dimension of the system (size of the matrix A).

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    ### avec le Laplacien
    
    As, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=2, off_diagonal_value=-1)
    
    h=1/(n+1)
    A=(1/(h*h)) * As 
    print(A)
    A_dense=(1/(h*h)) * A_dense
    

    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    
    # Right-hand side vector
    b = np.random.rand(n)

    return A,  A_dense, b


def jacobi_sparse_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=10000):
    # méthode de Jacobi
   
    
    x = x0.copy()
    errors_J = []
    D_inv=1/A.diagonal()
    LU=A-sparse.diags(A.diagonal()) #-(L+U)
    #print(f"LU: {LU}")
    
    start_time = time.time()
    for i in range(max_iter):
        x_new=D_inv*(b-LU.dot(x))
        
        error_J = np.linalg.norm(x_new - x_exact)
        errors_J.append(error_J) 
        
        if error_J < tol:
            break
        x = x_new
        
    end_time = time.time()
    time_taken = end_time - start_time
        
    return x_new, i+1, errors_J, time_taken


def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=10000):
    # méthode de GS
    n = A.shape[0]
    x = x0.copy()
    errors_GS = []
    D_inv=1/A.diagonal()
    
    start_time = time.time()
    
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            L=(A[i, :i]).dot(x_new[:i])
            U=(A[i, i+1:]).dot(x[i+1:])
            x_new[i] = (b[i] - L - U) / A[i, i]
            
        
        error_GS = np.linalg.norm(x_new - x_exact)
        errors_GS.append(error_GS)
        
        if error_GS < tol:
            break
        x = x_new
                
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Temps écoulé: {time_taken}") 
    
    return x_new, k+1, errors_GS, time_taken




#n=10 #donc h=1/11 donc 1/h² = 121
#x0 = np.zeros(n)


#tests pour différentes tailles de matrices
#dim = [5]
dim = [1000, 2000, 3000, 5000]
results = []


for n in dim:
    x0=np.zeros(n,dtype=np.float64)
    As, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=-1)
    
    #solution exacte
    x_exact = np.linalg.solve(A_dense, b)

    # Jacobi (sparse)
    x_new, iter_J, error_J,time_J = jacobi_sparse_with_error(As, b, x0, x_exact, tol=1e-6, max_iter=10000)
    # GS 
    x_new, iter_GS, error_GS, time_GS=gauss_seidel_sparse_with_error(As, b, x0, x_exact, tol=1e-6, max_iter=10000)
    
    print(f"Iterations (J): {iter_J}, Time (J): {time_J:.4f} seconds")
    print(f"Iterations (GS): {iter_GS}, Time (GS): {time_GS:.4f} seconds")
    
    results.append((n, time_J, time_GS))
    
    

# results
print("n\tJacobi Time\tGauss-Seidel Time")
for r in results:
    print(f"{r[0]}\t{r[1]:.4f}\t\t{r[2]:.4f}")
    
    
#tracer les erreurs
def plot_error(error_J, error_GS, iter_J, iter_GS):
    plt.figure(figsize=(8, 6))

    # première méthode (J)
    plt.semilogy(range(iter_J), error_J, marker='o', linestyle='-',color='blue', label='méthode de Jacobi')  # Log-scale for the y-axis

    # deuxieme méthode (GS)
    plt.semilogy(range(iter_GS), error_GS, marker='o', linestyle='-', label='méthode de GS')  # Log-scale for the y-axis

    
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations Comparison")
    plt.grid(True)
    plt.legend()  # Ajouter une légende pour identifier chaque courbe

    # Afficher le graphique
    plt.show()


plot_error(error_J, error_GS, iter_J, iter_GS)

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
    
    main_diag = np.full(n, diagonal_value)
    upp_diag= np.full(n-1,off_diagonal_value)
    low_diag=np.full(n-1,off_diagonal_value)
    
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
    
    As, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=2, off_diagonal_value=-1)
    
    h=1/(n+1)
    A=(1/(h*h)) * As 
    print(A)
    A_dense=(1/(h*h)) * A_dense
    
    b = np.random.rand(n)

    return A,  A_dense, b


def jacobi_sparse_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=10000): 
    # méthode de Jacobi
    
    x = x0.copy()
    errors_J = []
    D_inv=1/A.diagonal()
    LU=A-sparse.diags(A.diagonal()) #-(L+U)
    
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
    #méthode de GS
    n = A.shape[0]
    x = x0.copy()
    errors_GS = []
    #D_inv=1/A.diagonal()
    
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


def SOR(A, b, x0, x_exact, tol=1e-6, max_iter=10000, omega=0.5):
    # méthode SOR (omega appartient à [0,2])

    n = A.shape[0]
    x = x0.copy()
    errors_SOR = []
    
    start_time = time.time()

    for k in range(max_iter):
        
        x_new = x.copy()
        
        for i in range(n):
            L=(A[i, :i]).dot (x_new[:i])
            U=(A[i, i+1:]).dot(x[i+1:])
            s = (b[i] - L - U) / A[i, i]
            x_new[i]=omega*s+(1-omega)*x[i]
            
        
        error_SOR = np.linalg.norm(x_new - x_exact)
        errors_SOR.append(error_SOR)
        
        if error_SOR < tol:
            break
        x = x_new
            
            
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Temps écoulé: {time_taken}") 
    
    return x, k+1, errors_SOR, time_taken



#tests 
dim = [100]
#dim = [1000, 2000, 3000, 5000]
results = []


for n in dim:
    x0=np.zeros(n,dtype=np.float64)
    #As, A_dense, b = generate_sparse_tridiagonal_matrix(n)
    As, A_dense, b=generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=2, off_diagonal_value=-1)
    
    #solution exacte
    x_exact = np.linalg.solve(A_dense, b)
  
    # Jacobi (sparse)
    x_new, iter_J, error_J,time_J = jacobi_sparse_with_error(As, b, x0, x_exact, tol=1e-6, max_iter=100)
    
    # GS 
    x_new, iter_GS, error_GS, time_GS=gauss_seidel_sparse_with_error(As, b, x0, x_exact, tol=1e-6, max_iter=100)
    
    #SOR
    x, iter_SOR, error_SOR, time_SOR= SOR(As, b, x0, x_exact, tol=1e-6, max_iter=100, omega=1.5)
    
    
    print(f"Iterations (J): {iter_J}, Time (J): {time_J:.4f} seconds")
    print(f"Iterations (GS): {iter_GS}, Time (GS): {time_GS:.4f} seconds")
    print(f"Iterations (SOR): {iter_SOR}, Time (SOR): {time_SOR:.4f} seconds")
    
    results.append((n, time_J, time_GS))
    
    

# results
print("n\tDense Time\tSparse Time")
for r in results:
    print(f"{r[0]}\t{r[1]}\t\t{r[2]}")
    
    

def plot_error(error_J, error_GS, iter_J, iter_GS, error_SOR, iter_SOR):
    plt.figure(figsize=(8, 6))

    # première méthode (J) (en orange)
    plt.semilogy(range(iter_J), error_J, marker='o', linestyle='-',color='blue', label='Jacobi')  

    # deuxieme méthode (GS) (en bleu)
    plt.semilogy(range(iter_GS), error_GS, marker='o', linestyle='-', label='GS')  
    
    #troisième méthode (SOR) (en rouge)
    plt.semilogy(range(iter_SOR), error_SOR, marker='o', linestyle='-', label='SOR')

    # Ajouter titres et légendes
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations Comparison")
    plt.grid(True)
    plt.legend()


    # Afficher le graphique
    plt.show()


plot_error(error_J, error_GS, iter_J, iter_GS, error_SOR, iter_SOR)

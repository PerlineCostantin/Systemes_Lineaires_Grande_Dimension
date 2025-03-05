import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time

# POUR TRACER LES ERREURS EN FONCTION DES ITERATIONS POUR LES DIFFERENTS OMEGAS

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
   
    
    main_diag = np.full(n, diagonal_value)#diagonale principale
    upp_diag= np.full(n-1,off_diagonal_value)#diagonale supérieure
    low_diag=np.full(n-1,off_diagonal_value)#diagonale inférieure
    
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



def SOR(A, b, x0, x_exact, tol=1e-6, max_iter=10000, omega=0.5):
   
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
    
    return x, k+1, errors_SOR, time_taken


def best_omega(A, b, x0, x_exact, tol=1e-6, max_iter=10000):
    #pour trouver la valeur de oméga avec laquelle la méthode converge le plus vite
    best_w=0
    min_iter = max_iter
    val=np.arange(0.1,1.9,0.1) # omega de 0.1 à 1.9, pas de 0.1
    w_errors_SOR = [] #tableau contenant toutes les erreurs pour les différents omégas
    
    for omega in val:
        x, iterations, errors, time_taken = SOR(A, b, x0, x_exact, tol=tol, max_iter=max_iter, omega=omega)
        w_errors_SOR.append(errors)
        
        if (iterations<min_iter): #si le nombre d'itérations actuel est inférieur au nombre d'itérations minimum
            min_iter=iterations #on change le nombre d'itérations minimum
            best_w=omega #on change le meilleur omega
            best_time=time_taken
  
    
    print(f"Best omega: {best_w}, Iterations {min_iter}, Temps écoulé:{best_time:.4f} seconds")     
    return val, w_errors_SOR


def plot_omega_errors(val, w_errors_SOR):
    #pour tracer les erreurs avec les différents omégas et voir avec quel omega la méthode converge le plus vite
    
    plt.figure(figsize=(8, 6))
    
    m=len(val)

    for i in range(m):
        #pour toutes les valeurs de omega, on trace la droite des erreurs en fonction des itérations 
        plt.semilogy(range(len(w_errors_SOR[i])), w_errors_SOR[i], label=f"omega = {val[i]:.1f}") 

    plt.xlabel("Iterations")
    plt.ylabel("Error ")
    plt.title("Error vs Iterations for Different Omega Values")
    plt.grid(True)
    plt.legend()
    plt.show()

# Tests
dim = [100]  
results = []

for n in dim:
    x0 = np.zeros(n, dtype=np.float64)
    #As, A_dense, b = generate_sparse_tridiagonal_matrix(n)
    As, A_dense, b=generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=-1)
    x_exact = np.linalg.solve(A_dense, b)

   
    val, w_errors_SOR = best_omega(As, b, x0, x_exact)

    plot_omega_errors(val, w_errors_SOR)

    # methode de Jacobi et Gauss-Seidel 
    x_new, iter_J, error_J, time_J = jacobi_sparse_with_error(As, b, x0, x_exact, tol=1e-6, max_iter=100)
    x_new, iter_GS, error_GS, time_GS = gauss_seidel_sparse_with_error(As, b, x0, x_exact, tol=1e-6, max_iter=100)

   
    print(f"Iterations (Jacobi): {iter_J}, Time (Jacobi): {time_J:.4f} seconds")
    print(f"Iterations (Gauss-Seidel): {iter_GS}, Time (Gauss-Seidel): {time_GS:.4f} seconds")
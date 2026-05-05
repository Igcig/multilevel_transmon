import numpy as np
import scipy.constants as ct
import qutip as qt
import scipy.linalg as alg

def cos_func(phi, Ej):
    return -Ej*np.cos(phi)

def second_harmonic(phi, A, B):
    return -A*np.cos(phi) - B*np.cos(2*phi)

def single_ABS_channel(phi, Ej, T):
    return -4*Ej*np.sqrt(1 - T*np.sin(phi/2)**2)/T

def numerical_transmon_E(Ej, Ec, ng, n_max, n_E=None, eigvals_only=False, potential="cos", return_H=None):
    if potential == "cos":
        diagonal = 4*Ec*(np.arange(-int(n_max), int(n_max)+1, 1) - ng)**2
        offdiagonal = -Ej*np.ones(int(2*n_max))/2
        H = np.diag(diagonal, 0) + np.diag(offdiagonal, -1) + np.diag(offdiagonal, 1)
    else:
        # For functional forms different than cos(phi) we solve in the phase basis
        n_max = 2*n_max + 1
        dphi = 2*np.pi/n_max
        phi = np.arange(-np.pi, np.pi, dphi)
        if len(phi) > n_max:
            phi = phi[:-1]
        diagonal = potential(phi, Ej) + 4*Ec*(2/dphi**2 + ng**2)
        offdiagonal_up = 4*Ec*(-np.ones(n_max-1)/dphi**2 + ng*1j*np.ones(n_max-1)/dphi)
        offdiagonal_down = 4*Ec*(-np.ones(n_max-1)/dphi**2 - ng*1j*np.ones(n_max-1)/dphi)
        H = np.diag(diagonal, 0) + np.diag(offdiagonal_down, -1) + np.diag(offdiagonal_up, 1)
        H[0,-1] = offdiagonal_down[0]
        H[-1,0] = offdiagonal_up[0]
    
    if return_H is not None:
        return_H[:, :] = H.copy()  # broadcasting

    if potential == "cos":
        if n_E is None:
            return alg.eigh_tridiagonal(diagonal, offdiagonal, eigvals_only=eigvals_only)
        else:
            return alg.eigh_tridiagonal(diagonal, offdiagonal, eigvals_only=eigvals_only, select='i', select_range=(0, n_E-1))
    else:
        if n_E is None:
            return alg.eigh(H, eigvals_only=eigvals_only)
        else:
            return alg.eigh(H, subset_by_index=[0, n_E-1], eigvals_only=eigvals_only)

def numerical_coupled_E(A, N, Ej, Ec, ng, fr, beta, Vrms, n_max, n_E, eigvals_only=True, potential="cos", return_H=None):
    E_transmon, eigenstates_transmon = numerical_transmon_E(Ej, Ec, ng, n_max, n_E=N, potential=potential)
    if potential == "cos":
        # Write N in charge basis as a diagonal matrix
        N_base_n = np.diag(range(-int(n_max), int(n_max)+1, 1))
        N_base_diag = eigenstates_transmon.transpose() @ N_base_n @ eigenstates_transmon
    else:
        # Write N in phase basis is not a diagonal matrix
        n_max = 2*n_max + 1
        dphi = 2*np.pi/n_max
        offdiagonal_up =  -1j*np.ones(n_max-1)/2/dphi
        offdiagonal_down = 1j*np.ones(n_max-1)/2/dphi
        N_base_phi = np.diag(offdiagonal_down, -1) + np.diag(offdiagonal_up, 1)
        N_base_phi[0,-1] = offdiagonal_down[0]
        N_base_phi[-1,0] = offdiagonal_up[0]
        N_base_diag = eigenstates_transmon.transpose() @ N_base_phi @ eigenstates_transmon

    H_transmon = np.diag(E_transmon)

    E_res = fr*np.arange(A)
    g0 = 2*beta*ct.e*Vrms/ct.h
    H = np.zeros((N*A, N*A), dtype=complex)
    for j in range(A):
        H[j*N:(j+1)*N, j*N:(j+1)*N] = H_transmon + E_res[j]*np.eye(N)
        if j + 1 < A:
            H[j*N:(j+1)*N, (j+1)*N:(j+2)*N] = g0*N_base_diag*np.sqrt(j+1)
            H[(j+1)*N:(j+2)*N, j*N:(j+1)*N] = g0*N_base_diag.transpose().conj()*np.sqrt(j+1)
    
    if return_H is not None:
        return_H[:, :] = H.copy()  # broadcasting
    
    return alg.eigh(H, subset_by_index=[0, n_E-1], eigvals_only=eigvals_only)

def sorted_energies(A, N, EJ, EC, ng, fr, g0, n_fqs=1, N_max=400, potential="cos"):
    if not isinstance(EJ, (list, np.ndarray)):
        EJ = [EJ]
    length = len(EJ)
    n_E = int(N*A/2)
    
    even_energies = np.zeros((n_E, length))
    even_states = np.empty((n_E, length), dtype=qt.core.qobj.Qobj)
    odd_energies = np.zeros((n_E, length))
    odd_states = np.empty((n_E, length), dtype=qt.core.qobj.Qobj)
    fq_num = np.zeros((n_fqs, length))
    G = np.empty(length, dtype=qt.core.qobj.Qobj)

    P_a = qt.Qobj(np.diag((-1)**(np.arange(A))))
    P_n = qt.Qobj(np.diag((-1)**(np.arange(N))))
    P = qt.tensor(P_a, P_n)
    H = np.zeros((N*A, N*A), dtype=complex)
    for j, ej in enumerate(EJ):
        (E, V) = numerical_coupled_E(A, N, ej, EC, ng, fr, g0, ct.h/2/ct.e, N_max, N*A, eigvals_only=False, potential=potential, return_H=H)
        G[j] = qt.tensor(qt.qeye(A), qt.Qobj(H[0:N,N:2*N]))
        even_e = []
        even_v = []
        odd_e = []
        odd_v = []
        
        for l, energy in enumerate(E):
            v = qt.Qobj(V[:,l].reshape((-1,1)), dims=[[A, N], [1, 1]])
            parity = np.real(v.dag()*P*v)
            if parity > 0:
                even_e.append(energy)
                even_v.append(v)
            else:
                odd_e.append(energy)
                odd_v.append(v)
        if n_E > len(even_e):
            for _ in  range(n_E - len(even_e)):
                even_e.append(0)
                even_v.append(qt.Qobj(np.zeros(N*A), dims=[[A, N], [1, 1]]))
        if n_E > len(odd_e):
            for _ in  range(n_E - len(odd_e)):
                odd_e.append(0)
                odd_v.append(qt.Qobj(np.zeros(N*A), dims=[[A, N], [1, 1]]))
        even_energies[:n_E, j] = even_e[:n_E]
        even_states[:n_E, j] = even_v[:n_E]
        odd_energies[:n_E, j] = odd_e[:n_E]
        odd_states[:n_E, j] = odd_v[:n_E]
        # The matrix H has been constructed using the real eigenenergies of the transmon for the diagonal
        fq_num[:,j] = np.real(np.diag(H)[1:(n_fqs+1)] - H[0,0])
    
    return fq_num, G, even_energies, even_states, odd_energies, odd_states

def H_JC(fq,gi,fr,A):
    # cavity mode operator
    a = qt.tensor(qt.destroy(A), qt.qeye(2))
    # Hqubit
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0],[0,fq]])) )

    # qubit/atom operators
    sm = qt.tensor(qt.qeye(A), qt.destroy(2)) # sigma-minus operator
    
    H = fr * a.dag()*a + Hqb + gi*a*sm.dag() + np.conj(gi)*a.dag()*sm
    return H

def H_QRM(fq,gi,fr,A):
    # cavity mode operator
    a = qt.tensor(qt.destroy(A), qt.qeye(2))
    # Hqubit
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0],[0,fq]])) )

    # qubit/atom operators
    sm = qt.tensor(qt.qeye(A), qt.destroy(2)) # sigma-minus operator
    
    # the full Hamiltonian!!!
    H = fr * a.dag()*a + Hqb + gi*(a+a.dag())*sm.dag() + np.conj(gi)*(a+a.dag())*sm
    return H

def H_JC_transmon(fq, Ec, g0,fr,A):
    # cavity mode operator
    a = qt.tensor(qt.destroy(A), qt.qeye(2))
    # Hqubit
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0],[0,fq]])) )

    # qubit/atom operators
    sm = qt.tensor(qt.qeye(A), qt.destroy(2)) # sigma-minus operator
    
    g = g0*np.sqrt((fq + Ec)/Ec)/4
    H = fr * a.dag()*a + Hqb + g*a*sm.dag() + np.conj(g)*a.dag()*sm
    return H

def H_twolevel_transmon(fq, Ec, gi, fr, A):
    # cavity mode operator
    a = qt.tensor(qt.destroy(A), qt.qeye(2))
    # Hqubit
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0],[0,fq]])))

    # qubit/atom operators
    sm = qt.tensor(qt.qeye(A), qt.destroy(2)) # sigma-minus operator
    
    g = gi*np.sqrt((fq + Ec)/Ec)/4
    # the full Hamiltonian!!!
    H = fr * a.dag()*a + Hqb + g*(a+a.dag())*sm.dag() + np.conj(g)*(a+a.dag())*sm
    return H

def H_threelevel_transmon(fq,Ec,alpha,gi,fr,A):
    # cavity mode operator
    a = qt.tensor(qt.destroy(A), qt.qeye(3))

    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0,0],[0,fq,0],[0,0,2*fq-3*alpha*Ec]])))

    # qubit/atom operators
    sm = qt.tensor(qt.qeye(A), qt.destroy(3)) # sigma-minus operator
    
    g = gi*np.sqrt((fq + Ec)/Ec)/4
    H = fr * a.dag()*a + Hqb + g*(a+a.dag())*sm.dag() + np.conj(g)*(a+a.dag())*sm
    return H

def hamiltonian_fit(fq, fr, g):
    A = 10
    E1g = np.zeros(len(fq))
    E0e = np.zeros(len(fq))
    for j, f in enumerate(fq):
        _,E0e[j],E1g[j],_ = H_QRM(f,g,fr,A).eigenenergies()[0:4]
    return E1g-E0e

def hamiltonian_fit_variable_g_aprox(fq, fr, g0):
    A = 10
    E1g = np.zeros(len(fq))
    E0e = np.zeros(len(fq))
    for j, f in enumerate(fq):
        g = g0*(1+1/2/fr*(f-fr))
        _,E0e[j],E1g[j],_ = H_QRM(f,g,fr,A).eigenenergies()[0:4]
    return E1g-E0e

def twolevel_transmon_H_fit(fq, fr, Ec, g):
    A = 10
    E1g = np.zeros(len(fq))
    E0e = np.zeros(len(fq))
    for j, f in enumerate(fq):
        _,E0e[j],E1g[j],_ = H_twolevel_transmon(f,Ec,g,fr,A).eigenenergies()[0:4]
    return E1g-E0e

def threelevel_transmon_H_fit(fq, fr, Ec, alpha, g):
    A = 10
    E1g = np.zeros(len(fq))
    E0e = np.zeros(len(fq))
    P_a = qt.Qobj(np.diag((-1)**(np.arange(A))))
    P_n = qt.Qobj(np.diag((-1)**(np.arange(3))))
    P = qt.tensor(P_a, P_n)
    for j, f in enumerate(fq):
        E, V = H_threelevel_transmon(f,Ec,alpha,g,fr,A).eigenstates()
        even_energies = []
        odd_energies = []
        for k, energy in enumerate(E):
            v = V[k]
            parity = v.dag()*P*v
            if np.real(parity) > 0:
                even_energies.append(energy)
            else:
                odd_energies.append(energy)
        E0e[j],E1g[j] = (odd_energies[0] - even_energies[0]), (odd_energies[1] - even_energies[0])
    return E1g-E0e
import numpy as np
import scipy.constants as ct
import qutip as qt
import scipy.linalg as alg

# =============================================================================
# Potential energy functions
# =============================================================================

def cos_func(phi, Ej):
    """Josephson junction cosine potential.

    Parameters
    ----------
    phi : array_like
        Superconducting phase difference.
    Ej : float
        Josephson energy.

    Returns
    -------
    ndarray
        Potential energy -Ej * cos(phi).
    """
    return -Ej*np.cos(phi)


def second_harmonic(phi, A, B):
    """Two-harmonic Josephson potential, e.g. for a junction with second harmonic.

    Parameters
    ----------
    phi : array_like
        Superconducting phase difference.
    A : float
        Amplitude of the first harmonic.
    B : float
        Amplitude of the second harmonic.

    Returns
    -------
    ndarray
        Potential energy -A*cos(phi) - B*cos(2*phi).
    """
    return -A*np.cos(phi) - B*np.cos(2*phi)


def single_ABS_channel(phi, Ej, T):
    """Potential energy of a single Andreev bound state (ABS) channel.

    Parameters
    ----------
    phi : array_like
        Superconducting phase difference.
    Ej : float
        Josephson energy.
    T : float
        Transmission coefficient of the channel (0 < T <= 1).

    Returns
    -------
    ndarray
        ABS energy -4*Ej*sqrt(1 - T*sin(phi/2)^2) / T.
    """
    return -4*Ej*np.sqrt(1 - T*np.sin(phi/2)**2)/T


# =============================================================================
# Transmon Hamiltonian diagonalization
# =============================================================================

def numerical_transmon_E(Ej, Ec, ng, n_max, n_E=None, eigvals_only=False, potential="cos", return_H=None):
    """Compute the energy spectrum of a transmon qubit numerically.

    For the standard cosine potential, the Hamiltonian is built in the charge
    basis as a tridiagonal matrix resulting in much faster computation. For
    arbitrary potentials it is built in the phase basis with periodic boundary
    conditions.

    Parameters
    ----------
    Ej : float
        Josephson energy
    Ec : float
        Charging energy.
    ng : float
        Offset charge (in units of 2e).
    n_max : int
        Charge basis cutoff: charge states from -n_max to n_max are included.
    n_E : int, optional
        Number of eigenstates to return. If None, all are returned.
    eigvals_only : bool, optional
        If True, return only eigenvalues. Default False.
    potential : str or callable, optional
        "cos" for the standard Josephson cosine potential, or a callable
        V(phi, Ej) for a custom potential evaluated in the phase basis.
    return_H : ndarray, optional
        If provided, the Hamiltonian matrix is written into this array in-place.

    Returns
    -------
    eigenvalues : ndarray
        Energy eigenvalues.
    eigenvectors : ndarray
        Eigenvectors (only if eigvals_only=False).
    """
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
        return_H[:, :] = H.copy()

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
    """Compute the energy spectrum of a transmon coupled to a microwave resonator.

    The transmon is diagonalized first; then the coupling to the resonator is
    written in the transmon eigenbasis using the charge operator N.

    Parameters
    ----------
    A : int
        Number of Fock states of the resonator.
    N : int
        Number of transmon levels to keep.
    Ej : float
        Josephson energy.
    Ec : float
        Charging energy.
    ng : float
        Offset charge (in units of 2e).
    fr : float
        Bare resonator frequency.
    beta : float
        Capacitance ratio (dimensionless).
    Vrms : float
        RMS voltage of the resonator vacuum fluctuations.
    n_max : int
        Charge basis cutoff for the transmon diagonalization.
    n_E : int
        Number of eigenstates of the full coupled system to return.
    eigvals_only : bool, optional
        If True, return only eigenvalues. Default True.
    potential : str or callable, optional
        Transmon potential. See `numerical_transmon_E`.
    return_H : ndarray, optional
        If provided, the full coupled Hamiltonian is written into this array in-place.

    Returns
    -------
    eigenvalues : ndarray
        Energy eigenvalues of the coupled system.
    eigenvectors : ndarray
        Eigenvectors (only if eigvals_only=False).
    """
    E_transmon, eigenstates_transmon = numerical_transmon_E(Ej, Ec, ng, n_max, n_E=N, potential=potential)
    if potential == "cos":
        # Write N in charge basis as a diagonal matrix
        N_base_n = np.diag(range(-int(n_max), int(n_max)+1, 1))
        N_base_diag = eigenstates_transmon.transpose() @ N_base_n @ eigenstates_transmon
    else:
        # Write N in phase basis (finite-difference derivative, periodic BC)
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
    g0 = 2*beta*ct.e*Vrms/ct.h  # bare coupling rate (Hz)
    H = np.zeros((N*A, N*A), dtype=complex)
    for j in range(A):
        H[j*N:(j+1)*N, j*N:(j+1)*N] = H_transmon + E_res[j]*np.eye(N)
        if j + 1 < A:
            H[j*N:(j+1)*N, (j+1)*N:(j+2)*N] = g0*N_base_diag*np.sqrt(j+1)
            H[(j+1)*N:(j+2)*N, j*N:(j+1)*N] = g0*N_base_diag.transpose().conj()*np.sqrt(j+1)

    if return_H is not None:
        return_H[:, :] = H.copy()

    return alg.eigh(H, subset_by_index=[0, n_E-1], eigvals_only=eigvals_only)


def sorted_energies(A, N, EJ, EC, ng, fr, g0, n_fqs=1, N_max=400, potential="cos"):
    """Compute and sort eigenenergies of the coupled transmon-resonator system by parity.

    The eigenstates are classified as even or odd under the parity operator
    P = (-1)^(a†a) ⊗ (-1)^(n_transmon_excitations), which is a conserved symmetry of the
    Quantum Rabi-like Hamiltonian.

    Energies and frequencies should all be given in the same units, either in Hz or in eV.

    Parameters
    ----------
    A : int
        Number of Fock states of the resonator.
    N : int
        Number of transmon levels to keep.
    EJ : float or array_like
        Josephson energy, or array of values to sweep over.
    EC : float
        Charging energy.
    ng : float
        Offset charge (in units of 2e).
    fr : float
        Bare resonator frequency.
    g0 : float
        Bare (vacuum) coupling energy.
    n_fqs : int, optional
        Number of transmon transition frequencies to extract. Default 1.
    N_max : int, optional
        Charge basis cutoff for the transmon. Default 400.
    potential : str or callable, optional
        Transmon potential. See `numerical_transmon_E`.

    Returns
    -------
    fq_num : ndarray, shape (n_fqs, len(EJ))
        Transmon transition frequencies extracted from the diagonal of H.
    G : ndarray of Qobj, shape (len(EJ),)
        Off-diagonal coupling block of the Hamiltonian as a QuTiP object.
    even_energies : ndarray, shape (N*A//2, len(EJ))
        Even-parity eigenenergies.
    even_states : ndarray of Qobj
        Even-parity eigenstates.
    odd_energies : ndarray, shape (N*A//2, len(EJ))
        Odd-parity eigenenergies.
    odd_states : ndarray of Qobj
        Odd-parity eigenstates.
    """
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
        even_e, even_v, odd_e, odd_v = [], [], [], []

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
        fq_num[:,j] = np.real(np.diag(H)[1:(n_fqs+1)] - H[0,0])

    return fq_num, G, even_energies, even_states, odd_energies, odd_states


# =============================================================================
# Model Hamiltonians (QuTiP)
# =============================================================================

def H_JC(fq, gi, fr, A):
    """Jaynes-Cummings Hamiltonian (rotating wave approximation).

    Models a two-level qubit coupled to a single resonator mode, keeping only
    energy-conserving interaction terms (a*sigma+ + a†*sigma-).

    Parameters
    ----------
    fq : float
        Qubit frequency.
    gi : float
        Qubit-resonator coupling strength.
    fr : float
        Resonator frequency.
    A : int
        Hilbert space truncation for the resonator (number of Fock states).

    Returns
    -------
    Qobj
        Jaynes-Cummings Hamiltonian as a QuTiP operator.
    """
    a = qt.tensor(qt.destroy(A), qt.qeye(2))
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0],[0,fq]])))
    sm = qt.tensor(qt.qeye(A), qt.destroy(2))
    H = fr * a.dag()*a + Hqb + gi*a*sm.dag() + np.conj(gi)*a.dag()*sm
    return H


def H_QRM(fq, gi, fr, A):
    """Quantum Rabi Model Hamiltonian (beyond rotating wave approximation).

    Extends the Jaynes-Cummings model by including counter-rotating terms
    (a*sigma- + a†*sigma+), relevant in the ultrastrong coupling regime.

    Parameters
    ----------
    fq : float
        Qubit frequency.
    gi : float
        Qubit-resonator coupling strength.
    fr : float
        Resonator frequency.
    A : int
        Hilbert space truncation for the resonator.

    Returns
    -------
    Qobj
        Quantum Rabi Hamiltonian as a QuTiP operator.
    """
    a = qt.tensor(qt.destroy(A), qt.qeye(2))
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0],[0,fq]])))
    sm = qt.tensor(qt.qeye(A), qt.destroy(2))
    H = fr * a.dag()*a + Hqb + gi*(a+a.dag())*sm.dag() + np.conj(gi)*(a+a.dag())*sm
    return H


def H_JC_transmon(fq, Ec, g0, fr, A):
    """Jaynes-Cummings Hamiltonian with transmon-corrected coupling.

    The coupling is renormalized: g = g0 * sqrt((fq + Ec) / Ec) / 4.

    Energies and frequencies should all be given in the same units, either in Hz or in eV.

    Parameters
    ----------
    fq : float
        Qubit (transmon) frequency.
    Ec : float
        Charging energy of the transmon.
    g0 : float
        Bare coupling strength.
    fr : float
        Bare resonator frequency.
    A : int
        Hilbert space truncation for the resonator.

    Returns
    -------
    Qobj
        Transmon-corrected Jaynes-Cummings Hamiltonian.
    """
    a = qt.tensor(qt.destroy(A), qt.qeye(2))
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0],[0,fq]])))
    sm = qt.tensor(qt.qeye(A), qt.destroy(2))
    g = g0*np.sqrt((fq + Ec)/Ec)/4
    H = fr * a.dag()*a + Hqb + g*a*sm.dag() + np.conj(g)*a.dag()*sm
    return H


def H_twolevel_transmon(fq, Ec, gi, fr, A):
    """Quantum Rabi Hamiltonian with transmon-corrected coupling (two-level).

    Same renormalization as `H_JC_transmon` but including counter-rotating terms.

    Energies and frequencies should all be given in the same units, either in Hz or in eV.

    Parameters
    ----------
    fq : float
        Qubit (transmon) frequency.
    Ec : float
        Charging energy of the transmon.
    gi : float
        Bare coupling strength.
    fr : float
        Bare resonator frequency.
    A : int
        Hilbert space truncation for the resonator.

    Returns
    -------
    Qobj
        Two-level transmon Quantum Rabi Hamiltonian.
    """
    a = qt.tensor(qt.destroy(A), qt.qeye(2))
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0],[0,fq]])))
    sm = qt.tensor(qt.qeye(A), qt.destroy(2))
    g = gi*np.sqrt((fq + Ec)/Ec)/4
    H = fr * a.dag()*a + Hqb + g*(a+a.dag())*sm.dag() + np.conj(g)*(a+a.dag())*sm
    return H


def H_threelevel_transmon(fq, Ec, alpha, gi, fr, A):
    """Quantum Rabi Hamiltonian for a three-level transmon.

    The third level energy is set to 2*fq - 3*alpha*Ec, accounting for the
    transmon anharmonicity alpha.

    Energies and frequencies should all be given in the same units, either in Hz or in eV.

    Parameters
    ----------
    fq : float
        01 transition frequency of the transmon.
    Ec : float
        Charging energy.
    alpha : float
        Anharmonicity prefactor (dimensionless).
    gi : float
        Bare coupling strength.
    fr : float
        Bare resonator frequency.
    A : int
        Hilbert space truncation for the resonator.

    Returns
    -------
    Qobj
        Three-level transmon Quantum Rabi Hamiltonian.
    """
    a = qt.tensor(qt.destroy(A), qt.qeye(3))
    Hqb = qt.tensor(qt.qeye(A), qt.Qobj(np.array([[0,0,0],[0,fq,0],[0,0,2*fq-3*alpha*Ec]])))
    sm = qt.tensor(qt.qeye(A), qt.destroy(3))
    g = gi*np.sqrt((fq + Ec)/Ec)/4
    H = fr * a.dag()*a + Hqb + g*(a+a.dag())*sm.dag() + np.conj(g)*(a+a.dag())*sm
    return H


# =============================================================================
# Spectroscopy fitting functions
# =============================================================================

def hamiltonian_fit(fq, fr, g):
    """Compute the spectroscopic transition frequency using the Quantum Rabi Model.

    Returns E_1g - E_0e as a function of qubit frequency, for use as a fit
    function against spectroscopy data.

    Parameters
    ----------
    fq : array_like
        Array of qubit frequencies to evaluate.
    fr : float
        Bare resonator frequency.
    g : float
        Coupling strength.

    Returns
    -------
    ndarray
        Transition frequency E_1g - E_0e at each value of fq.
    """
    A = 10
    E1g = np.zeros(len(fq))
    E0e = np.zeros(len(fq))
    for j, f in enumerate(fq):
        _,E0e[j],E1g[j],_ = H_QRM(f,g,fr,A).eigenenergies()[0:4]
    return E1g-E0e


def hamiltonian_fit_variable_g_aprox(fq, fr, g0):
    """QRM fit with a frequency-dependent coupling approximation.

    The coupling is linearized around the resonator frequency:
    g(fq) = g0 * (1 + (fq - fr) / (2*fr)).

    Parameters
    ----------
    fq : array_like
        Array of qubit frequencies.
    fr : float
        Resonator frequency.
    g0 : float
        Coupling at resonance.

    Returns
    -------
    ndarray
        Transition frequency E_1g - E_0e at each value of fq.
    """
    A = 10
    E1g = np.zeros(len(fq))
    E0e = np.zeros(len(fq))
    for j, f in enumerate(fq):
        g = g0*(1+1/2/fr*(f-fr))
        _,E0e[j],E1g[j],_ = H_QRM(f,g,fr,A).eigenenergies()[0:4]
    return E1g-E0e


def twolevel_transmon_H_fit(fq, fr, Ec, g):
    """Spectroscopy fit using the two-level transmon Quantum Rabi Hamiltonian.

    Energies and frequencies should all be given in the same units, either in Hz or in eV.

    Parameters
    ----------
    fq : array_like
        Array of qubit frequencies.
    fr : float
        Bare resonator frequency.
    Ec : float
        Charging energy of the transmon.
    g : float
        Bare coupling strength.

    Returns
    -------
    ndarray
        Transition frequency E_1g - E_0e at each value of fq.
    """
    A = 10
    E1g = np.zeros(len(fq))
    E0e = np.zeros(len(fq))
    for j, f in enumerate(fq):
        _,E0e[j],E1g[j],_ = H_twolevel_transmon(f,Ec,g,fr,A).eigenenergies()[0:4]
    return E1g-E0e


def threelevel_transmon_H_fit(fq, fr, Ec, alpha, g):
    """Spectroscopy fit using the three-level transmon Quantum Rabi Hamiltonian.

    Eigenstates are sorted by parity before extracting transition frequencies,
    necessary because level crossings can reorder the spectrum.

    Energies and frequencies should all be given in the same units, either in Hz or in eV.

    Parameters
    ----------
    fq : array_like
        Array of qubit frequencies.
    fr : float
        Bare resonator frequency.
    Ec : float
        Charging energy of the transmon.
    alpha : float
        Anharmonicity prefactor (dimensionless).
    g : float
        Bare coupling strength.

    Returns
    -------
    ndarray
        Transition frequency E_1g - E_0e at each value of fq.
    """
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
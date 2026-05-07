//! Spectra calculation library for transmon-resonator systems.
//!
//! This module provides high-performance computation of microwave transmission
//! spectra for a transmon qubit coupled to a microwave resonator. The spectra
//! are computed by summing Lorentzian contributions from all pairs of eigenstates,
//! weighted by transition matrix elements.
//!
//! Three types of transitions are supported:
//! - **Single-photon (01)**: transitions between the ground and first excited manifold.
//! - **Two-photon (2ph)**: transitions involving absorption/emission of two photons.
//! - **Sideband (12)**: transitions between the first and second excited manifold.
//!
//! The Python-facing functions are:
//! - [`calculate_spectra`]: spectrum for a single set of eigenstates.
//! - [`calculate_spectra_2D`]: spectrum swept over a parameter (e.g. EJ).
//! - [`calculate_multiphoton_spectra`]: 1-, 2-, 3-, and 4-photon spectra.
//!
//! Parallelism over frequency points is handled by Rayon.

use std::f64::consts::PI;
use std::ops::Mul;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
use ndarray::Array2;
use numpy::IntoPyArray;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadonlyArray3;
use numpy::PyArray2;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use num_complex::Complex;
use rayon::prelude::*;
use itertools::Itertools;

// =============================================================================
// Python-facing functions
// =============================================================================

/// Compute transmission spectra swept over a parameter axis (e.g. Josephson energy EJ).
///
/// This is the batched version of [`calculate_spectra`]. It iterates over the
/// last axis of the input arrays in parallel using Rayon, returning 2D arrays
/// of shape `(num_ej, num_f)`.
///
/// # Arguments
/// * `A` - Number of Fock states of the resonator.
/// * `N` - Number of transmon levels kept in the simulation.
/// * `peaks_01` - Number of single-photon peaks included in the selection matrix.
/// * `peaks_2ph` - Number of two-photon peaks included in the selection matrix.
/// * `peaks_12` - Number of sideband peaks included in the selection matrix.
/// * `w` - Lorentzian linewidth (FWHM, same units as `f`).
/// * `f` - Frequency axis to evaluate the spectrum on.
/// * `even_state_population` - Thermal population of each even-parity eigenstate.
/// * `odd_state_population` - Thermal population of each odd-parity eigenstate.
/// * `even_energies` - Even-parity eigenenergies, shape `(n_E, num_ej)`.
/// * `even_states` - Even-parity eigenstates, shape `(num_ej, n_E, dim)`.
/// * `odd_energies` - Odd-parity eigenenergies, shape `(n_E, num_ej)`.
/// * `odd_states` - Odd-parity eigenstates, shape `(num_ej, n_E, dim)`.
/// * `G` - Coupling matrix in the transmon eigenbasis, shape `(num_ej, dim, dim)`.
///
/// # Returns
/// Four 2D arrays `(a_sel, t_sel, t2ph_sel, t12_sel)` of shape `(num_ej, num_f)`:
/// - `a_sel`: annihilation-operator spectrum (cavity transmission proxy).
/// - `t_sel`: single-photon transmission spectrum.
/// - `t2ph_sel`: two-photon transmission spectrum.
/// - `t12_sel`: sideband (1→2) transmission spectrum.
#[pyfunction]
fn calculate_spectra_2D(
    py: Python<'_>,
    A: usize,
    N: usize,
    peaks_01: usize,
    peaks_2ph: usize,
    peaks_12: usize,
    w: f64,
    f: Vec<f64>,
    even_state_population: Vec<f64>,
    odd_state_population: Vec<f64>,
    even_energies: PyReadonlyArray2<f64>,
    even_states: PyReadonlyArray3<Complex<f64>>,
    odd_energies: PyReadonlyArray2<f64>,
    odd_states: PyReadonlyArray3<Complex<f64>>,
    G: PyReadonlyArray3<Complex<f64>>
) -> (Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>) {
    let even_energies = even_energies.as_array();
    let odd_energies = odd_energies.as_array();
    let even_states = even_states.as_array();
    let odd_states = odd_states.as_array();
    let G = G.as_array();
    let num_f = f.len();
    let num_ej = G.shape()[0];

    // Build selection matrices (real) and cast to complex for matrix products
    let M = matrix_n0_to_n1(A, N, peaks_01).mapv(|x| Complex::new(x, 0.0));
    let M2ph = matrix_2ph(A, N, peaks_2ph).mapv(|x| Complex::new(x, 0.0));
    let M12 = matrix_n1_to_n2(A, N, peaks_12).mapv(|x| Complex::new(x, 0.0));
    
    let (a_sel, t_sel, t2ph_sel, t12_sel): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
    (0..num_ej).into_par_iter().map(|m| {
        calculate_spectra_rust(
            M.view(),
            M2ph.view(),
            M12.view(),
            w,
            &f,
            &even_state_population,
            &odd_state_population,
            even_energies.index_axis(Axis(1), m),
            even_states.index_axis(Axis(0), m),
            odd_energies.index_axis(Axis(1), m),
            odd_states.index_axis(Axis(0), m),
            G.index_axis(Axis(0), m),
            build_annihilation_tensor_operator(A, N).view()
        )
    }).collect::<Vec<_>>().into_iter().multiunzip();
    
    let stotal_a_selection = Array2::from_shape_vec((num_ej, num_f), a_sel.concat()).unwrap();
    let stotal_t_selection = Array2::from_shape_vec((num_ej, num_f), t_sel.concat()).unwrap();
    let stotal_t2ph_selection = Array2::from_shape_vec((num_ej, num_f), t2ph_sel.concat()).unwrap();
    let stotal_t12_selection = Array2::from_shape_vec((num_ej, num_f), t12_sel.concat()).unwrap();
    
    (stotal_a_selection.into_pyarray(py).to_owned().into(),
     stotal_t_selection.into_pyarray(py).to_owned().into(),
     stotal_t2ph_selection.into_pyarray(py).to_owned().into(),
     stotal_t12_selection.into_pyarray(py).to_owned().into())
}

/// Compute transmission spectra for a single set of eigenstates.
///
/// For each frequency in `f`, the spectrum is obtained by summing Lorentzian
/// lineshapes centered at the transition frequencies between all pairs of
/// even- and odd-parity eigenstates, weighted by the squared modulus of the
/// corresponding transition matrix element.
///
/// # Arguments
/// See [`calculate_spectra_2D`] for argument descriptions. Here `even_energies`,
/// `even_states`, `odd_energies`, `odd_states`, and `G` are 1D/2D arrays
/// corresponding to a single parameter value.
///
/// # Returns
/// Four 1D arrays `(a_sel, t_sel, t2ph_sel, t12_sel)` of length `num_f`.
#[pyfunction]
fn calculate_spectra(
    py: Python<'_>,
    A: usize,
    N: usize,
    peaks_01: usize,
    peaks_2ph: usize,
    peaks_12: usize,
    w: f64,
    f: Vec<f64>,
    even_state_population: Vec<f64>,
    odd_state_population: Vec<f64>,
    even_energies: PyReadonlyArray1<f64>,
    even_states: PyReadonlyArray2<Complex<f64>>,
    odd_energies: PyReadonlyArray1<f64>,
    odd_states: PyReadonlyArray2<Complex<f64>>,
    G: PyReadonlyArray2<Complex<f64>>,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let even_energies = even_energies.as_array();
    let odd_energies = odd_energies.as_array();
    let G = G.as_array();
    let even_states = even_states.as_array();
    let odd_states = odd_states.as_array();

    // Build selection matrices
    let M = matrix_n0_to_n1(A, N, peaks_01).mapv(|x| Complex::new(x, 0.0));
    let M2ph = matrix_2ph(A, N, peaks_2ph).mapv(|x| Complex::new(x, 0.0));
    let M12 = matrix_n1_to_n2(A, N, peaks_12).mapv(|x| Complex::new(x, 0.0));
    
    let (row_a, row_t, row_t2ph, row_t12) = calculate_spectra_rust(
        M.view(),
        M2ph.view(),
        M12.view(),
        w,
        &f,
        &even_state_population,
        &odd_state_population,
        even_energies,
        even_states,
        odd_energies,
        odd_states,
        G,
        build_annihilation_tensor_operator(A, N).view()
    );
    
    (row_a.into_pyarray(py).to_owned().into(),
    row_t.into_pyarray(py).to_owned().into(),
    row_t2ph.into_pyarray(py).to_owned().into(),
    row_t12.into_pyarray(py).to_owned().into())
}

/// Compute 1-, 2-, 3-, and 4-photon transmission spectra.
///
/// Extends [`calculate_spectra`] to higher-order multiphoton transitions.
/// The n-photon coupling matrix is built as the Hadamard product of the
/// n-photon selection matrix with G^n (n-th matrix power of G).
///
/// # Returns
/// Four 1D arrays `(t01, t2ph, t3ph, t4ph)` of length `num_f`, corresponding
/// to 1-, 2-, 3-, and 4-photon transmission respectively.
#[pyfunction]
fn calculate_multiphoton_spectra(
    py: Python<'_>,
    A: usize,
    N: usize,
    w: f64,
    f: Vec<f64>,
    even_state_population: Vec<f64>,
    odd_state_population: Vec<f64>,
    even_energies: PyReadonlyArray1<f64>,
    even_states: PyReadonlyArray2<Complex<f64>>,
    odd_energies: PyReadonlyArray1<f64>,
    odd_states: PyReadonlyArray2<Complex<f64>>,
    G: PyReadonlyArray2<Complex<f64>>,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let even_energies = even_energies.as_array();
    let odd_energies = odd_energies.as_array();
    let G = G.as_array();
    let even_states = even_states.as_array();
    let odd_states = odd_states.as_array();

    // Build selection matrices for each photon order
    let M = matrix_n0_to_n1(A, N, 1).mapv(|x| Complex::new(x, 0.0));
    let M2ph = matrix_nph(A, N, 2).mapv(|x| Complex::new(x, 0.0));
    let M3ph = matrix_nph(A, N, 3).mapv(|x| Complex::new(x, 0.0));
    let M4ph = matrix_nph(A, N, 4).mapv(|x| Complex::new(x, 0.0));
    
    let (row_t01, row_t2ph, row_t3ph, row_t4ph) = calculate_multiphoton_spectra_rust(
        M.view(),
        M2ph.view(),
        M3ph.view(),
        M4ph.view(),
        w,
        &f,
        &even_state_population,
        &odd_state_population,
        even_energies,
        even_states,
        odd_energies,
        odd_states,
        G,
    );
    
    (row_t01.into_pyarray(py).to_owned().into(),
    row_t2ph.into_pyarray(py).to_owned().into(),
    row_t3ph.into_pyarray(py).to_owned().into(),
    row_t4ph.into_pyarray(py).to_owned().into())
}

// =============================================================================
// Core spectrum computation (internal)
// =============================================================================

/// Inner routine that computes the four spectral channels for a single parameter point.
///
/// For each pair of even (`j`) and odd (`k`) eigenstates, four transition matrix
/// elements are computed:
/// - `Aeo[j,k]` = ⟨even_j | (M ⊙ a) | odd_k⟩  (annihilation operator channel)
/// - `Teo[j,k]` = ⟨even_j | (M ⊙ G) | odd_k⟩  (single-photon transmission)
/// - `T2ph[j,k]` = ⟨even_j | (M2ph ⊙ G²) | even_k⟩  (two-photon, even→even)
/// - `T12[j,k]`  = ⟨even_j | (M12 ⊙ G) | odd_k⟩   (sideband 1→2)
///
/// The spectrum at each frequency is then a sum of Lorentzians:
/// `S(f) = Σ_{j,k} |T[j,k]|² * L(f - ΔE_{jk})`
/// where `L` is a Lorentzian of width `w`.
fn calculate_spectra_rust<'a>(
    M: ArrayView2<'a, Complex<f64>>,
    M2ph: ArrayView2<'a, Complex<f64>>,
    M12: ArrayView2<'a, Complex<f64>>,
    w: f64,
    f: &'a [f64],
    even_state_population: &'a [f64],
    odd_state_population: &'a [f64],
    even_energies: ArrayView1<'a, f64>,
    even_states: ArrayView2<'a, Complex<f64>>,
    odd_energies: ArrayView1<'a, f64>,
    odd_states: ArrayView2<'a, Complex<f64>>,
    G: ArrayView2<'a, Complex<f64>>,
    A:ArrayView2<'a, Complex<f64>>
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {

    // Build effective coupling matrices: element-wise product of selection mask and operator
    let Ma = hadamard_product(&M, &A);
    let Mt = hadamard_product(&M, &G);
    let M2ph = hadamard_product(&M2ph, &G.dot(&G));
    let M12 = hadamard_product(&M12, &G);
    
    // Pre-compute all transition matrix elements between eigenstates
    let n_E = even_energies.len();
    let mut Aeo = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut Teo = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut T2ph = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut T12 = Array2::<Complex<f64>>::zeros((n_E, n_E));
    for j in 0..n_E {
        for k in 0..n_E {
            Teo[[j,k]] = even_states.row(j).dag().dot(&Mt.dot(&odd_states.row(k)));
            Aeo[[j,k]] = even_states.row(j).dag().dot(&Ma.dot(&odd_states.row(k)));
            T2ph[[j,k]] = even_states.row(j).dag().dot(&M2ph.dot(&even_states.row(k)));
            T12[[j,k]] = even_states.row(j).dag().dot(&M12.dot(&odd_states.row(k)));
        }
    }

    // Sum Lorentzian contributions at each frequency point (parallelized over f)
    let (row_a, row_t, row_t2ph, row_t12): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
    f.par_iter().map(|freq| {
        let mut a = 0.0;
        let mut t = 0.0;
        let mut t2ph = 0.0;
        let mut t12 = 0.0;
        for k in 0..n_E {
            for j in 0..n_E {
                let evenj = even_energies[j];
                let evenk = even_energies[k];
                let odd = odd_energies[k];
                let pop_sum = even_state_population[j] + even_state_population[k];
                let pop_mix = even_state_population[j] + odd_state_population[k];

                // Lorentzian L(f) = (w/2π) / ((w/2)² + (f - f0)²)
                let lortz = w / (2.0 * PI) / (w.powi(2) / 4.0 + (freq - (evenj - odd).abs()).powi(2));
                let mut lortz2ph = w / (2.0 * PI) / (w.powi(2) / 4.0 + (2.0 * freq - (evenj - evenk)).powi(2));
                lortz2ph += w / (2.0 * PI) / (w.powi(2) / 4.0 + (freq - (evenj - evenk)).powi(2));
                a += Aeo[[j,k]].norm_sqr() * lortz;
                t += Teo[[j,k]].norm_sqr() * lortz;
                t2ph += T2ph[[j,k]].norm_sqr() * lortz2ph * pop_sum;
                t12 += T12[[j,k]].norm_sqr() * lortz * pop_mix;
            }
        }
        (a, t, t2ph, t12)
    }).collect::<Vec<_>>().into_iter().multiunzip();
    (row_a, row_t, row_t2ph, row_t12)
}

/// Inner routine for multiphoton spectra computation.
///
/// Analogous to [`calculate_spectra_rust`] but computes up to 4-photon processes.
/// The n-photon Lorentzian is centered at `n*f = |E_j - E_k|`, and the coupling
/// matrix is built from G^n (n successive applications of G).
fn calculate_multiphoton_spectra_rust<'a>(
    M: ArrayView2<'a, Complex<f64>>,
    M2ph: ArrayView2<'a, Complex<f64>>,
    M3ph: ArrayView2<'a, Complex<f64>>,
    M4ph: ArrayView2<'a, Complex<f64>>,
    w: f64,
    f: &'a [f64],
    even_state_population: &'a [f64],
    odd_state_population: &'a [f64],
    even_energies: ArrayView1<'a, f64>,
    even_states: ArrayView2<'a, Complex<f64>>,
    odd_energies: ArrayView1<'a, f64>,
    odd_states: ArrayView2<'a, Complex<f64>>,
    G: ArrayView2<'a, Complex<f64>>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // n-photon coupling: Hadamard product of selection mask with G^n
    let Mt = hadamard_product(&M, &G);
    let M2ph = hadamard_product(&M2ph, &G.dot(&G));
    let M3ph = hadamard_product(&M3ph, &G.dot(&G.dot(&G)));
    let M4ph = hadamard_product(&M4ph, &G.dot(&G.dot(&G.dot(&G))));

    let n_E = even_energies.len();
    let mut Teo = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut T2ph = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut T3ph = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut T4ph = Array2::<Complex<f64>>::zeros((n_E, n_E));
    for j in 0..n_E {
        for k in 0..n_E {
            Teo[[j,k]] = even_states.row(j).dag().dot(&Mt.dot(&odd_states.row(k)));
            T2ph[[j,k]] = even_states.row(j).dag().dot(&M2ph.dot(&even_states.row(k)));
            T3ph[[j,k]] = even_states.row(j).dag().dot(&M3ph.dot(&odd_states.row(k)));
            T4ph[[j,k]] = even_states.row(j).dag().dot(&M4ph.dot(&even_states.row(k)));
        }
    }

    let (row_t01, row_t2ph, row_t3ph, row_t4ph): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
    f.par_iter().map(|freq| {
        let mut t = 0.0;
        let mut t2ph = 0.0;
        let mut t3ph = 0.0;
        let mut t4ph = 0.0;
        for k in 0..n_E {
            for j in 0..n_E {
                let evenj = even_energies[j];
                let evenk = even_energies[k];
                let odd = odd_energies[k];
                let pop_sum = even_state_population[j] + even_state_population[k];
                let pop_mix = even_state_population[j] + odd_state_population[k];

                let lortz = w / (2.0 * PI) / (w.powi(2) / 4.0 + (freq - (evenj - odd).abs()).powi(2));
                let mut lortz2ph = w / (2.0 * PI) / (w.powi(2) / 4.0 + (2.0 * freq - (evenj - evenk)).powi(2));
                lortz2ph += w / (2.0 * PI) / (w.powi(2) / 4.0 + (freq - (evenj - evenk)).powi(2));
                let mut lortz3ph = w / (2.0 * PI) / (w.powi(2) / 4.0 + (3.0 * freq - (evenj - odd).abs()).powi(2));
                lortz3ph += w / (2.0 * PI) / (w.powi(2) / 4.0 + (freq - (evenj - odd).abs()).powi(2));
                let mut lortz4ph = w / (2.0 * PI) / (w.powi(2) / 4.0 + (4.0 * freq - (evenj - evenk)).powi(2));
                lortz4ph += w / (2.0 * PI) / (w.powi(2) / 4.0 + (2.0 * freq - (evenj - evenk)).powi(2));
                lortz4ph += w / (2.0 * PI) / (w.powi(2) / 4.0 + (freq - (evenj - evenk)).powi(2));
                t += Teo[[j,k]].norm_sqr() * lortz;
                t2ph += T2ph[[j,k]].norm_sqr() * lortz2ph * pop_sum;
                t3ph += T3ph[[j,k]].norm_sqr() * lortz3ph * pop_mix;
                t4ph += T4ph[[j,k]].norm_sqr() * lortz4ph * pop_sum;
            }
        }
        (t, t2ph, t3ph, t4ph)
    }).collect::<Vec<_>>().into_iter().multiunzip();
    (row_t01, row_t2ph, row_t3ph, row_t4ph)
}

// =============================================================================
// Selection matrices
// =============================================================================

/// Build a selection matrix for single-photon (0→1) transitions.
///
/// Places ones at positions `(i*N, i*N + 1)` and their transposes for each
/// of the first `n_peaks` photon-number sectors.
fn matrix_n0_to_n1(A: usize, N: usize, n_peaks: usize) -> Array2<f64> {
    let size = A * N;
    let mut m = Array2::<f64>::zeros((size, size));

    for i in 0..n_peaks {
        let ground = i * N;
        let excited = ground + 1;
        m[[ground, excited]] = 1.0;
        m[[excited, ground]] = 1.0;
    }
    m
}

/// Build a selection matrix for two-photon transitions (0→2 in photon number).
///
/// Places ones at `(i*N, i*N + 2)` and transposes for each of the first
/// `n_peaks` photon-number sectors.
fn matrix_2ph(A: usize, N: usize, n_peaks: usize) -> Array2<f64> {
    let size = A * N;
    let mut m = Array2::<f64>::zeros((size, size));

    for i in 0..n_peaks {
        let ground = i * N;
        let excited = ground + 2;
        m[[ground, excited]] = 1.0;
        m[[excited, ground]] = 1.0;
    }
    m
}

/// Build a selection matrix for an n-photon transition starting from the vacuum.
///
/// Places a single one at `(0, n)` and its transpose, selecting only the
/// `|0⟩ → |n⟩` photon-number transition from the ground state.
fn matrix_nph(A: usize, N: usize, n: usize) -> Array2<f64> {
    let size = A * N;
    let mut m = Array2::<f64>::zeros((size, size));
    m[[0, n]] = 1.0;
    m[[n, 0]] = 1.0;
    m
}

/// Build a selection matrix for sideband (1→2) transitions.
///
/// Places ones at `(i*N + 1, i*N + 2)` and transposes for each of the
/// first `n_peaks` photon-number sectors.
fn matrix_n1_to_n2(A: usize, N: usize, n_peaks: usize) -> Array2<f64> {
    let size = A * N;
    let mut m = Array2::<f64>::zeros((size, size));

    for i in 0..n_peaks {
        let ground = i * N + 1;
        let excited = ground + 1;
        m[[ground, excited]] = 1.0;
        m[[excited, ground]] = 1.0;
    }
    m
}

// =============================================================================
// Linear algebra utilities
// =============================================================================

/// Element-wise (Hadamard) product of two matrices of equal shape.
fn hadamard_product<T, A, B>(a: &ArrayBase<A, Ix2>, b: &ArrayBase<B, Ix2>) -> Array2<T>
where
    T: Copy + Mul<Output = T> + num_traits::identities::Zero,
    A: Data<Elem = T>,
    B: Data<Elem = T>,
{
    assert_eq!(a.dim(), b.dim());
    let mut result = Array2::<T>::zeros(a.raw_dim());
    ndarray::Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &x, &y| *r = x * y);
    result
}

/// Conjugate transpose (†) for 1D quantum state vectors.
pub trait QuantumStateOps {
    fn dag(&self) -> Array1<Complex<f64>>;
}

/// Conjugate transpose (†) for 2D quantum operators.
pub trait QuantumOperatorOps {
    fn dag(&self) -> Array2<Complex<f64>>;
}

impl<S> QuantumStateOps for ArrayBase<S, Ix1>
where
    S: Data<Elem = Complex<f64>>,
{
    fn dag(&self) -> Array1<Complex<f64>> {
        self.t().mapv(|x| x.conj())
    }
}

impl<S> QuantumOperatorOps for ArrayBase<S, Ix2>
where
    S: Data<Elem = Complex<f64>>,
{
    fn dag(&self) -> Array2<Complex<f64>> {
        self.t().mapv(|x| x.conj())
    }
}

/// Bosonic annihilation operator in Fock space of dimension `dim`.
///
/// Returns the matrix with entries `a[i-1, i] = sqrt(i)` for i in 1..dim.
fn destroy(dim: usize) -> Array2<Complex<f64>> {
    let mut a = Array2::<Complex<f64>>::zeros((dim, dim));
    for i in 1..dim {
        a[[i - 1, i]] = Complex::new((i as f64).sqrt(), 0.0);
    }
    a
}

/// Identity matrix of dimension `dim` with complex entries.
fn identity(dim: usize) -> Array2<Complex<f64>> {
    Array2::from_diag(&Array1::from_elem(dim, Complex::new(1.0, 0.0)))
}

/// Kronecker (tensor) product of two complex matrices.
fn kronecker(
    a: &Array2<Complex<f64>>,
    b: &Array2<Complex<f64>>,
) -> Array2<Complex<f64>> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();

    let mut result = Array2::<Complex<f64>>::zeros((a_rows * b_rows, a_cols * b_cols));

    for i in 0..a_rows {
        for j in 0..a_cols {
            let a_ij = a[[i, j]];
            for k in 0..b_rows {
                for l in 0..b_cols {
                    result[[i * b_rows + k, j * b_cols + l]] = a_ij * b[[k, l]];
                }
            }
        }
    }

    result
}

/// Build the cavity annihilation operator in the full tensor-product Hilbert space.
///
/// Returns `a ⊗ I_N`, where `a` is the bosonic annihilation operator on the
/// resonator space of dimension `A`, and `I_N` is the identity on the transmon
/// space of dimension `N`.
fn build_annihilation_tensor_operator(A: usize, N: usize) -> Array2<Complex<f64>> {
    let a = destroy(A);
    let id = identity(N);
    kronecker(&a, &id)
}

// =============================================================================
// PyO3 module definition
// =============================================================================

/// Python module exposing the spectra calculation functions.
#[pymodule]
fn spectra_calculation(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_spectra_2D, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_multiphoton_spectra, m)?)?;
    Ok(())
}
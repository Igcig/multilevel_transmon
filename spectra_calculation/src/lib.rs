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

    // Calculate selection matrices
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

    // Calculate selection matrices
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

    // Calculate selection matrices
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

    let Ma = hadamard_product(&M, &A);
    let Mt = hadamard_product(&M, &G);
    let M2ph = hadamard_product(&M2ph, &G.dot(&G));
    let M12 = hadamard_product(&M12, &G);
    
    let n_E = even_energies.len();
    let mut Aeo = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut Teo = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut T2ph = Array2::<Complex<f64>>::zeros((n_E, n_E));
    let mut T12 = Array2::<Complex<f64>>::zeros((n_E, n_E));
    for j in 0..n_E {
        for k in 0..n_E {
            // Calculate transition matrix elements
            Teo[[j,k]] = even_states.row(j).dag().dot(&Mt.dot(&odd_states.row(k)));
            Aeo[[j,k]] = even_states.row(j).dag().dot(&Ma.dot(&odd_states.row(k)));
            T2ph[[j,k]] = even_states.row(j).dag().dot(&M2ph.dot(&even_states.row(k)));
            T12[[j,k]] = even_states.row(j).dag().dot(&M12.dot(&odd_states.row(k)));
        }
    }
    // let (Teo, Aeo, T2ph, T12) = compute_all_transition_elements(&even_states, &odd_states, &Ma, &Mt, &M2ph, &M12);

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

                let lortz = w / (2.0 * PI) / (w.powi(2) / 4.0 + (freq - (evenj - odd).abs()).powi(2));
                let lortz2ph = w / (2.0 * PI) / (w.powi(2) / 4.0 + (2.0 * freq - (evenj - evenk)).powi(2));
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
            // Calculate transition matrix elements
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
                let lortz2ph = w / (2.0 * PI) / (w.powi(2) / 4.0 + (2.0 * freq - (evenj - evenk)).powi(2));
                let lortz3ph = w / (2.0 * PI) / (w.powi(2) / 4.0 + (3.0 * freq - (evenj - odd).abs()).powi(2));
                let lortz4ph = w / (2.0 * PI) / (w.powi(2) / 4.0 + (4.0 * freq - (evenj - evenk)).powi(2));
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

fn matrix_n0_to_n1(A: usize, N: usize, n_peaks: usize) -> Array2<f64> {
    let size = A * N;
    let mut m = Array2::<f64>::zeros((size, size));

    for i in 0..n_peaks {
        let ground = i * N;
        let excited = ground + 1;

        // Set symmetric entries
        m[[ground, excited]] = 1.0;
        m[[excited, ground]] = 1.0;
    }
    m
}

fn matrix_2ph(A: usize, N: usize, n_peaks: usize) -> Array2<f64> {
    let size = A * N;
    let mut m = Array2::<f64>::zeros((size, size));

    for i in 0..n_peaks {
        let ground = i * N;
        let excited = ground + 2;

        // Set symmetric entries
        m[[ground, excited]] = 1.0;
        m[[excited, ground]] = 1.0;
    }
    m
}

fn matrix_nph(A: usize, N: usize, n: usize) -> Array2<f64> {
    let size = A * N;
    let mut m = Array2::<f64>::zeros((size, size));

    let ground = 0;
    let excited = n;

    // Set symmetric entries
    m[[ground, excited]] = 1.0;
    m[[excited, ground]] = 1.0;
    m
}

fn matrix_n1_to_n2(A: usize, N: usize, n_peaks: usize) -> Array2<f64> {
    let size = A * N;
    let mut m = Array2::<f64>::zeros((size, size));

    for i in 0..n_peaks {
        let ground = i * N + 1;
        let excited = ground + 1;

        // Set symmetric entries
        m[[ground, excited]] = 1.0;
        m[[excited, ground]] = 1.0;
    }
    m
}

/// Element-wise product of two matrices
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

// For 1D quantum states (e.g. eigenstates)
pub trait QuantumStateOps {
    fn dag(&self) -> Array1<Complex<f64>>;
}

// For 2D quantum operators (e.g. matrices)
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

fn destroy(dim: usize) -> Array2<Complex<f64>> {
    let mut a = Array2::<Complex<f64>>::zeros((dim, dim));
    for i in 1..dim {
        a[[i - 1, i]] = Complex::new((i as f64).sqrt(), 0.0);
    }
    a
}

fn identity(dim: usize) -> Array2<Complex<f64>> {
    Array2::from_diag(&Array1::from_elem(dim, Complex::new(1.0, 0.0)))
}

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

fn build_annihilation_tensor_operator(A: usize, N: usize) -> Array2<Complex<f64>> {
    let a = destroy(A);
    let id = identity(N);
    kronecker(&a, &id)
}

/// A Python module implemented in Rust.
#[pymodule]
fn spectra_calculation(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_spectra_2D, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_spectra, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_multiphoton_spectra, m)?)?;
    Ok(())
}
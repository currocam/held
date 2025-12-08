use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::ops::Bound::Included;

// This function computes E[X_iY_iX_jY_j]
#[inline]
pub fn linkage_disequilibrium(genotypes1: &[f64], genotypes2: &[f64]) -> f64 {
    let s = genotypes1.len() as f64;
    let (ld, ld_square) = genotypes1.iter().zip(genotypes2.iter()).fold(
        (0.0, 0.0),
        |(ld_acc, ld_square_acc), (&a, &b)| {
            let prod = a * b;
            (ld_acc + prod, ld_square_acc + prod * prod)
        },
    );
    (ld * ld - ld_square) / (s * (s - 1.0))
}

#[pyclass]
pub struct Bins {
    left_bins: Vec<f64>,
    right_bins: Vec<f64>,
}

#[pymethods]
impl Bins {
    #[new]
    fn new(left_bins: Vec<f64>, right_bins: Vec<f64>) -> PyResult<Self> {
        if left_bins.len() != right_bins.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "left_bins and right_bins must have the same length",
            ));
        }
        Ok(Bins {
            left_bins,
            right_bins,
        })
    }

    fn __len__(&self) -> usize {
        self.left_bins.len()
    }

    #[getter]
    fn left_bins<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.left_bins)
    }

    #[getter]
    fn right_bins<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.right_bins)
    }
}

#[pyclass]
struct StreamingStats {
    counts: Vec<usize>,
    ld: Vec<f64>,
    ld_square: Vec<f64>,
    rolling_map: RollingMap,
}

impl StreamingStats {
    #[inline]
    fn update_bin_inline(
        counts: &mut [usize],
        ld: &mut [f64],
        ld_square: &mut [f64],
        index: usize,
        genotypes1: &[f64],
        genotypes2: &[f64],
    ) {
        let new_value = linkage_disequilibrium(genotypes1, genotypes2);
        counts[index] += 1;
        let delta = new_value - ld[index];
        ld[index] += delta / counts[index] as f64;
        let delta2 = new_value - ld[index];
        ld_square[index] += delta * delta2;
    }
}

struct RollingMap {
    // Maps a given position to a vector of standarized genotypes
    pub map: BTreeMap<u64, Vec<f64>>,
    maf_threshold: f64,
}

impl RollingMap {
    fn default() -> Self {
        RollingMap {
            map: BTreeMap::new(),
            maf_threshold: 0.25,
        }
    }
    #[inline]
    fn insert(&mut self, position: i32, genotypes: &[i32]) {
        let position = position as u64;
        // Compute MAF
        let total: i32 = genotypes.iter().sum();
        let n = genotypes.len();
        let n_f64 = n as f64;
        let allele_frequency = total as f64 / (2.0 * n_f64);
        let maf = allele_frequency.min(1.0 - allele_frequency);
        if maf < self.maf_threshold {
            return;
        }
        let mean = 2.0 * allele_frequency;
        let std_dev = (2.0 * allele_frequency * (1.0 - allele_frequency)).sqrt();
        let mut standarized = Vec::with_capacity(n);
        for &gt in genotypes {
            standarized.push((gt as f64 - mean) / std_dev);
        }
        self.map.insert(position, standarized);
    }
}

#[pymethods]
impl StreamingStats {
    #[new]
    fn new(n_bins: usize) -> Self {
        StreamingStats {
            counts: vec![0; n_bins],
            ld: vec![0.0; n_bins],
            ld_square: vec![0.0; n_bins],
            rolling_map: RollingMap::default(),
        }
    }

    fn update_batch(
        &mut self,
        genotypes: PyReadonlyArray2<i32>,
        positions: PyReadonlyArray1<i32>,
        bins: &Bins,
    ) -> PyResult<()> {
        let genotypes = genotypes.as_array();
        let positions = positions.as_array();
        let shape = genotypes.shape();
        if positions.len() != shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} rows in genotypes, got {}",
                positions.len(),
                shape[0]
            )));
        }

        let min_distance = *bins.left_bins.first().unwrap();
        let max_distance = *bins.right_bins.last().unwrap();

        // Add rows from the batch
        for i in 0..shape[0] {
            let position = positions[i];
            let row: Vec<i32> = genotypes.row(i).to_vec();
            self.rolling_map.insert(position, &row);
        }

        // Iterate over the rolling map
        while let Some((&position1, _)) = self.rolling_map.map.first_key_value() {
            let min_next = (position1 as f64 + min_distance) as u64;
            let max_next = (position1 as f64 + max_distance) as u64;
            if let Some((&last_position, _)) = self.rolling_map.map.last_key_value() {
                if max_next > last_position {
                    break;
                }
            } else {
                break;
            }
            let (_, genotypes1) = self.rolling_map.map.remove_entry(&position1).unwrap();
            let mut bin_index = 0;

            // Iterate across relevant values directly without collecting
            for (position2, genotypes2) in self
                .rolling_map
                .map
                .range((Included(&min_next), Included(&max_next)))
            {
                let distance = (*position2 - position1) as f64;
                while bin_index < bins.left_bins.len() && distance > bins.right_bins[bin_index] {
                    bin_index += 1;
                }
                if bin_index >= bins.left_bins.len() {
                    break;
                }
                if distance >= bins.left_bins[bin_index] && distance <= bins.right_bins[bin_index] {
                    Self::update_bin_inline(
                        &mut self.counts,
                        &mut self.ld,
                        &mut self.ld_square,
                        bin_index,
                        &genotypes1,
                        genotypes2,
                    );
                }
            }
        }
        Ok(())
    }

    fn finalize<'py>(
        &mut self,
        py: Python<'py>,
        bins: &Bins,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let min_distance = *bins.left_bins.first().unwrap();
        let max_distance = *bins.right_bins.last().unwrap();
        // We are not gonna get new samples, so we have to finish processing
        while !self.rolling_map.map.is_empty() {
            let (position1, genotypes1) = self.rolling_map.map.pop_first().unwrap();
            let min_next = (position1 as f64 + min_distance) as u64;
            let max_next = (position1 as f64 + max_distance) as u64;
            let mut bin_index = 0;

            // Iterate across relevant values directly without collecting
            for (position2, genotypes2) in self
                .rolling_map
                .map
                .range((Included(&min_next), Included(&max_next)))
            {
                let distance = (*position2 - position1) as f64;
                while bin_index < bins.left_bins.len() && distance > bins.right_bins[bin_index] {
                    bin_index += 1;
                }
                if bin_index >= bins.left_bins.len() {
                    break;
                }
                if distance >= bins.left_bins[bin_index] && distance <= bins.right_bins[bin_index] {
                    Self::update_bin_inline(
                        &mut self.counts,
                        &mut self.ld,
                        &mut self.ld_square,
                        bin_index,
                        &genotypes1,
                        genotypes2,
                    );
                }
            }
        }
        let mut mean = self.ld.clone();
        let mut var = self.ld_square.clone();
        for i in 0..self.counts.len() {
            if self.counts[i] > 1 {
                var[i] /= self.counts[i] as f64;
            } else {
                var[i] = f64::NAN;
                mean[i] = f64::NAN;
            }
        }
        // Return a matrix with 3 columns: mean, variance, and count
        let count: Vec<f64> = self.counts.iter().map(|&c| c as f64).collect();
        let matrix: Vec<Vec<f64>> = (0..mean.len())
            .map(|i| vec![mean[i], var[i], count[i]])
            .collect();
        Ok(PyArray2::from_vec2(py, &matrix)?)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn held(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StreamingStats>()?;
    m.add_class::<Bins>()?;
    Ok(())
}

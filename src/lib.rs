use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
struct StreamingStats {
    sums: Vec<f64>,
    counts: Vec<usize>,
    n_cols: usize,
}

#[pymethods]
impl StreamingStats {
    #[new]
    fn new(n_cols: usize) -> Self {
        StreamingStats {
            sums: vec![0.0; n_cols],
            counts: vec![0; n_cols],
            n_cols,
        }
    }

    fn update(&mut self, matrix: PyReadonlyArray2<i32>) -> PyResult<()> {
        let array = matrix.as_array();
        let shape = array.shape();

        if shape[1] != self.n_cols {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} columns, got {}",
                self.n_cols, shape[1]
            )));
        }

        for row in array.rows() {
            for (col_idx, &value) in row.iter().enumerate() {
                self.sums[col_idx] += value as f64;
                self.counts[col_idx] += 1;
            }
        }

        Ok(())
    }

    fn finalize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let means: Vec<f64> = self
            .sums
            .iter()
            .zip(self.counts.iter())
            .map(
                |(&sum, &count)| {
                    if count > 0 { sum / count as f64 } else { 0.0 }
                },
            )
            .collect();

        Ok(PyArray1::from_slice(py, &means))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
mod held {
    #[pymodule_export]
    use super::StreamingStats;
}


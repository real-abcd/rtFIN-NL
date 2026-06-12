use anyhow::Result;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyList;

pub struct Embedder {
    inner: Py<PyAny>,
}

unsafe impl Send for Embedder {}
unsafe impl Sync for Embedder {}

impl Embedder {
    pub fn new(model: &str, device: &str) -> Result<Self> {
        Python::with_gil(|py| -> PyResult<_> {
            let st = py
                .import("sentence_transformers")?
                .getattr("SentenceTransformer")?
                .call1((model,))?;
            st.call_method1("to", (device,))?;
            Ok(Embedder { inner: st.into_py(py) })
        })
        .map_err(Into::into)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        Python::with_gil(|py| -> PyResult<Vec<f32>> {
            self.inner
                .as_ref(py)
                .call_method1("encode", (text,))
                .map_err(|e| PyRuntimeError::new_err(format!("텍스트인코딩안됨 ㅠㅠ: {}", e)))?
                .extract()
        })
        .map_err(Into::into)
    }

    /// 배치 임베딩 (`&[impl AsRef<str> + Debug]` 형태로 넘기세요)
    pub fn encode_batch<T>(&self, texts: &[T]) -> Result<Vec<f32>>
    where
        T: AsRef<str> + std::fmt::Debug,
    {
        Python::with_gil(|py| -> PyResult<Vec<f32>> {
            let rust_list: Vec<String> =
                texts.iter().map(|s| s.as_ref().to_string()).collect();
            let py_list = PyList::new(py, &rust_list);
            self.inner
                .as_ref(py)
                .call_method1("encode", (py_list,))
                .map_err(|e| PyRuntimeError::new_err(format!("배치못돌렸음 ㅠㅠ: {:?}", e)))?
                .extract()
        })
        .map_err(Into::into)
    }
}

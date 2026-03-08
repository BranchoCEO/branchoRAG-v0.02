use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use walkdir::WalkDir;

// --- BLOCK 1: DATA ---
#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<String>,
    edges: Vec<(usize, usize)>, 
}

#[pyclass]
struct BranchoRAG {
    data: GraphData,
}

// --- BLOCK 2: METHODS ---
#[pymethods]
impl BranchoRAG {
    #[new]
    fn new() -> Self {
        BranchoRAG {
            data: GraphData { nodes: Vec::new(), edges: Vec::new() },
        }
    }

    /// Scans a directory recursively, ignoring common "junk" folders.
    fn scan_folder(&mut self, path: String) -> PyResult<()> {
        // Folders to skip to prevent indexing thousands of unnecessary files
        let ignore_list = ["target", ".git", ".venv", "__pycache__", "env", "node_modules"];

        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            let path_str = entry.path().display().to_string();

            // Skip if the path contains any of the ignored directory names
            if ignore_list.iter().any(|&dir| path_str.contains(dir)) {
                continue;
            }

            if entry.file_type().is_file() {
                if !self.data.nodes.contains(&path_str) {
                    self.data.nodes.push(path_str);
                }
            }
        }
        Ok(())
    }

    fn save_memory(&self, filename: String) -> PyResult<()> {
        let json = serde_json::to_string_pretty(&self.data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    fn node_count(&self) -> usize {
        self.data.nodes.len()
    }
}

// --- BLOCK 3: THE MODULE ---
#[pymodule]
fn branchorag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BranchoRAG>()?;
    Ok(())
}

use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use walkdir::WalkDir;

// --- BLOCK 1: DATA ---
#[derive(Serialize, Deserialize)]
struct FileNode {
    path: String,
    content: String,
    embedding: Vec<f32>,  // 384-dim vector from sentence-transformers; empty until embed() is called
}

#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<FileNode>,
    edges: Vec<(usize, usize)>,
}

#[pyclass]
struct BranchoRAG {
    data: GraphData,
}

const IGNORE_LIST: &[&str] = &["target", ".git", ".venv", "__pycache__", "env", "node_modules"];
const MAX_FILE_BYTES: u64 = 1_000_000; // 1MB — skip minified JS, logs, binaries, etc.

// --- BLOCK 2: METHODS ---
#[pymethods]
impl BranchoRAG {
    #[new]
    fn new() -> Self {
        BranchoRAG {
            data: GraphData { nodes: Vec::new(), edges: Vec::new() },
        }
    }

    fn scan_folder(&mut self, path: String) -> PyResult<()> {

        // Track paths we've already seen to avoid duplicates
        let mut seen: HashSet<String> = self.data.nodes.iter().map(|n| n.path.clone()).collect();

        // filter_entry prunes ignored dirs entirely — WalkDir won't descend into them at all,
        // which is much faster than checking every file inside .git, node_modules, etc.
        for entry in WalkDir::new(path).into_iter().filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            !IGNORE_LIST.contains(&name)
        }).filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            let path_str = entry.path().display().to_string();
            if seen.contains(&path_str) {
                continue;
            }

            // Check size before reading to avoid loading huge files into memory
            let size = entry.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
            if size > MAX_FILE_BYTES {
                continue;
            }

            // read_to_string naturally skips binary files that aren't valid UTF-8
            if let Ok(content) = fs::read_to_string(entry.path()) {
                seen.insert(path_str.clone());
                self.data.nodes.push(FileNode { path: path_str, content, embedding: Vec::new() });
            }
        }
        Ok(())
    }

    /// Returns all file contents in order so Python can batch-encode them.
    fn get_contents(&self) -> Vec<String> {
        self.data.nodes.iter().map(|n| n.content.clone()).collect()
    }

    /// Accepts the full list of embeddings from Python and writes them back into each node.
    /// Expects exactly one Vec<f32> per node, in the same order as get_contents() returned them.
    fn set_embeddings(&mut self, embeddings: Vec<Vec<f32>>) -> PyResult<()> {
        if embeddings.len() != self.data.nodes.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} embeddings, got {}",
                self.data.nodes.len(),
                embeddings.len()
            )));
        }
        for (node, emb) in self.data.nodes.iter_mut().zip(embeddings.into_iter()) {
            node.embedding = emb;
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

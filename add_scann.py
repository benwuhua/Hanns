import re

with open("src/ffi.rs", "r") as f:
    code = f.read()

# 1. Add ScaNNIndex to imports
code = re.sub(
    r'use crate::faiss::\{(.*?)\};',
    r'use crate::faiss::{\1, ScaNNIndex};',
    code
)

# 2. Add CIndexType::Scann
if 'Scann = 9' not in code:
    code = re.sub(
        r'(DiskAnn\s*=\s*8,)',
        r'\1\n    Scann = 9,',
        code
    )

# 3. Add Scann fields to CIndexConfig
if 'num_partitions:' not in code:
    code = re.sub(
        r'(pub beamwidth: usize,)',
        r'\1\n    // ScaNN 参数\n    pub num_partitions: usize,\n    pub num_centroids: usize,\n    pub reorder_k: usize,\n    pub anisotropic_alpha: f32,\n',
        code
    )

# 4. Add Scann defaults
if 'num_partitions: 16' not in code:
    code = re.sub(
        r'(beamwidth: 8,)',
        r'\1\n            // ScaNN 默认参数\n            num_partitions: 16,\n            num_centroids: 256,\n            reorder_k: 100,\n            anisotropic_alpha: 0.2,',
        code
    )

# 5. Add scann field to IndexWrapper
if 'scann: Option<ScaNNIndex>' not in code:
    code = re.sub(
        r'(diskann: Option<DiskAnnIndex>,)',
        r'\1\n    scann: Option<ScaNNIndex>,',
        code
    )

# 6. Add scann: None to Some(Self { ... })
code = re.sub(r'diskann: (None|Some\(diskann\))(, dim \})', r'diskann: \1, scann: None\2', code)

# 7. Add CIndexType::Scann to match block
if 'CIndexType::Scann =>' not in code:
    scann_match = """
            CIndexType::Scann => {
                let mut index_config = IndexConfig {
                    index_type: IndexType::Scann,
                    metric_type: metric,
                    dim,
                    params: IndexParams::default(),
                };
                let scann = ScaNNIndex::new(&index_config).ok()?;
                Some(Self { flat: None, hnsw: None, ivf_flat: None, ivf_sq8: None, ivf_pq: None, binary_flat: None, binary_ivf: None, sparse: None, diskann: None, scann: Some(scann), dim })
            }"""
    code = re.sub(r'(CIndexType::DiskAnn => \{.*?\n            \})', r'\1\n' + scann_match, code, flags=re.DOTALL)

# 8. Add to add()
if 'self.scann' not in code[:code.find('fn train')]:
    code = re.sub(
        r'(\} else if let Some\(ref mut idx\) = self\.diskann \{\n\s*// DiskAnn.*?\n\s*idx\.add\(vectors, ids\)\.map_err\(\|\_\| CError::Internal\)\n\s*\}) else \{',
        r'\1 else if let Some(ref mut idx) = self.scann {\n            idx.add(vectors, ids).map_err(|_| CError::Internal)\n        } else {',
        code, flags=re.DOTALL
    )

# 9. Add to train()
if 'self.scann' not in code[code.find('fn train'):code.find('fn search')]:
    code = re.sub(
        r'(\} else if let Some\(ref mut idx\) = self\.diskann \{\n\s*// DiskAnn.*?\n\s*idx\.train\(vectors\)\.map_err\(\|\_\| CError::Internal\)\?;\n\s*Ok\(\(\)\)\n\s*\}) else \{',
        r'\1 else if let Some(ref mut idx) = self.scann {\n            idx.train(vectors).map_err(|_| CError::Internal)?;\n            Ok(())\n        } else {',
        code, flags=re.DOTALL
    )

# 10. Add to search()
if 'self.scann' not in code[code.find('fn search'):code.find('fn ntotal')]:
    code = re.sub(
        r'(\} else if let Some\(ref idx\) = self\.diskann \{\n\s*// DiskAnn.*?\n\s*idx\.search\(query, &req\)\.map_err\(\|\_\| CError::Internal\)\n\s*\}) else \{',
        r'\1 else if let Some(ref idx) = self.scann {\n            idx.search(query, &req).map_err(|_| CError::Internal)\n        } else {',
        code, flags=re.DOTALL
    )

# 11. Add to ntotal()
if 'self.scann' not in code[code.find('fn ntotal'):]:
    code = re.sub(
        r'(\} else if let Some\(ref idx\) = self\.diskann \{\n\s*idx\.ntotal\(\)\n\s*\}) else \{',
        r'\1 else if let Some(ref idx) = self.scann {\n            idx.ntotal()\n        } else {',
        code, flags=re.DOTALL
    )

with open("src/ffi.rs", "w") as f:
    f.write(code)

print("ScaNN FFI added successfully.")

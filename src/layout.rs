//! 缓存友好的数据布局

/// AoS (Array of Structures) - 传统布局
#[derive(Clone, Debug)]
pub struct VectorAoS {
    pub id: i64,
    pub data: Vec<f32>,
}

/// SoA (Structure of Arrays) - 缓存友好
pub struct VectorSoA {
    pub ids: Vec<i64>,
    pub data: Vec<f32>,
}

impl VectorSoA {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            data: Vec::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize, dim: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity * dim),
        }
    }
    
    pub fn push(&mut self, id: i64, vector: &[f32]) {
        self.ids.push(id);
        self.data.extend_from_slice(vector);
    }
    
    pub fn get(&self, idx: usize, dim: usize) -> Option<(&i64, &[f32])> {
        if idx < self.ids.len() {
            Some((&self.ids[idx], &self.data[idx * dim..(idx + 1) * dim]))
        } else {
            None
        }
    }
    
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    
    pub fn dim(&self, total_dim: usize) -> usize {
        if self.ids.is_empty() { 0 } else { total_dim }
    }
}

impl Default for VectorSoA {
    fn default() -> Self {
        Self::new()
    }
}

/// 列式存储（用于按列访问）
pub struct ColumnStore {
    columns: Vec<Vec<f32>>,
    num_rows: usize,
}

impl ColumnStore {
    pub fn new(num_cols: usize) -> Self {
        Self {
            columns: vec![Vec::new(); num_cols],
            num_rows: 0,
        }
    }
    
    pub fn add_row(&mut self, values: &[f32]) {
        assert_eq!(values.len(), self.columns.len());
        
        for (i, v) in values.iter().enumerate() {
            self.columns[i].push(*v);
        }
        self.num_rows += 1;
    }
    
    pub fn get_column(&self, col: usize) -> &[f32] {
        &self.columns[col]
    }
    
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        self.columns.get(col).and_then(|c| c.get(row)).copied()
    }
    
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
    
    pub fn num_cols(&self) -> usize {
        self.columns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soa() {
        let mut soa = VectorSoA::new();
        
        soa.push(1, &[1.0, 2.0, 3.0]);
        soa.push(2, &[4.0, 5.0, 6.0]);
        
        assert_eq!(soa.len(), 2);
        
        let (id, vec) = soa.get(0, 3).unwrap();
        assert_eq!(*id, 1);
        assert_eq!(vec, &[1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_column_store() {
        let mut store = ColumnStore::new(3);
        
        store.add_row(&[1.0, 2.0, 3.0]);
        store.add_row(&[4.0, 5.0, 6.0]);
        
        assert_eq!(store.num_rows(), 2);
        
        assert_eq!(store.get(0, 0), Some(1.0));
        assert_eq!(store.get(1, 2), Some(6.0));
    }
}

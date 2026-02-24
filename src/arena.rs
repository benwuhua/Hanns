//! Arena 内存分配器

/// Arena 分配器
pub struct Arena {
    chunks: Vec<Vec<u8>>,
    chunk_size: usize,
}

impl Arena {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            chunk_size: chunk_size.max(1024),
        }
    }
    
    pub fn alloc(&mut self, size: usize) -> Vec<u8> {
        let size = size.max(1);
        
        // 尝试找合适的块
        for chunk in &mut self.chunks {
            if chunk.len() >= size {
                return chunk.drain(..size).collect();
            }
        }
        
        // 创建新块
        let new_size = size.max(self.chunk_size);
        let mut new_chunk = vec![0u8; new_size];
        let result = new_chunk.drain(..size).collect();
        self.chunks.push(new_chunk);
        result
    }
    
    pub fn reset(&mut self) {
        self.chunks.clear();
    }
    
    pub fn total_used(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arena() {
        let mut arena = Arena::new(1024);
        let _a = arena.alloc(100);
        assert!(arena.total_used() > 0);
    }
    
    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new(1024);
        arena.alloc(100);
        arena.reset();
        assert_eq!(arena.total_used(), 0);
    }
}

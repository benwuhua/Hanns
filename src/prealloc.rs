//! 预分配向量

/// 预分配向量
pub struct PreallocVec<T> {
    data: Vec<T>,
    cap: usize,
}

impl<T: Clone> PreallocVec<T> {
    pub fn with_capacity(cap: usize) -> Self {
        Self { data: Vec::new(), cap }
    }
    
    pub fn push(&mut self, val: T) {
        if self.data.len() < self.cap {
            self.data.push(val);
        }
    }
    
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn as_slice(&self) -> &[T] { &self.data }
}

impl<T: Clone> Default for PreallocVec<T> {
    fn default() -> Self { Self::with_capacity(1024) }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prealloc() {
        let mut v = PreallocVec::with_capacity(3);
        v.push(1);
        v.push(2);
        assert_eq!(v.len(), 2);
    }
}

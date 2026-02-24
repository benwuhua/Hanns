//! 环形缓冲区

pub struct RingBuffer<T> {
    data: Vec<T>,
    head: usize,
    tail: usize,
    cap: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity.max(1)),
            head: 0,
            tail: 0,
            cap: capacity.max(1),
        }
    }
    
    pub fn push(&mut self, item: T) {
        if self.data.len() < self.cap {
            self.data.push(item);
        } else {
            self.data[self.head] = item;
            self.head = (self.head + 1) % self.cap;
            self.tail = (self.tail + 1) % self.cap;
        }
    }
    
    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() { return None; }
        let item = self.data.remove(self.tail);
        self.tail = (self.tail + 1) % self.cap;
        Some(item)
    }
    
    pub fn len(&self) -> usize { self.data.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring() {
        let mut r = RingBuffer::new(3);
        r.push(1);
        r.push(2);
        assert_eq!(r.len(), 2);
    }
}

use std::cell::RefCell;

pub struct VisitedList {
    gen: u16,
    visited: Vec<u16>,
}

impl VisitedList {
    pub fn new(n: usize) -> Self {
        Self {
            gen: 1,
            visited: vec![0u16; n],
        }
    }

    #[inline]
    pub fn is_visited(&self, id: u32) -> bool {
        let idx = id as usize;
        idx < self.visited.len() && self.visited[idx] == self.gen
    }

    #[inline]
    pub fn mark(&mut self, id: u32) {
        let idx = id as usize;
        if idx < self.visited.len() {
            self.visited[idx] = self.gen;
        }
    }

    /// O(1) reset; falls back to O(n) clear when generation wraps.
    pub fn reset(&mut self, required_n: usize) {
        if required_n > self.visited.len() {
            self.visited.resize(required_n, 0);
        }
        if self.gen == u16::MAX {
            self.visited.fill(0);
            self.gen = 1;
        } else {
            self.gen += 1;
        }
    }
}

thread_local! {
    static VISITED: RefCell<VisitedList> = RefCell::new(VisitedList::new(0));
}

pub fn with_visited<R>(n: usize, f: impl FnOnce(&mut VisitedList) -> R) -> R {
    VISITED.with(|v| {
        let mut v = v.borrow_mut();
        v.reset(n);
        f(&mut v)
    })
}

#[inline]
pub fn reset_thread_local(n: usize) {
    VISITED.with(|v| v.borrow_mut().reset(n));
}

#[inline]
pub fn is_visited_thread_local(id: u32) -> bool {
    VISITED.with(|v| v.borrow().is_visited(id))
}

#[inline]
pub fn mark_thread_local(id: u32) {
    VISITED.with(|v| v.borrow_mut().mark(id));
}

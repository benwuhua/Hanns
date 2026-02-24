//! 一次性值

pub struct OnceCell<T> {
    value: Option<T>,
}

impl<T> OnceCell<T> {
    pub fn new() -> Self { Self { value: None } }
    
    pub fn set(&mut self, value: T) -> Result<(), ()> {
        if self.value.is_some() { Err(()) } else { self.value = Some(value); Ok(()) }
    }
    
    pub fn get(&self) -> Option<&T> { self.value.as_ref() }
}

impl<T> Default for OnceCell<T> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_once() {
        let mut c = OnceCell::new();
        assert!(c.set(42).is_ok());
        assert_eq!(c.get(), Some(&42));
    }
}

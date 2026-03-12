//! 一次性值

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnceCellSetError;

impl Display for OnceCellSetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "value already initialized")
    }
}

impl Error for OnceCellSetError {}

pub struct OnceCell<T> {
    value: Option<T>,
}

impl<T> OnceCell<T> {
    pub fn new() -> Self {
        Self { value: None }
    }

    pub fn set(&mut self, value: T) -> Result<(), OnceCellSetError> {
        if self.value.is_some() {
            Err(OnceCellSetError)
        } else {
            self.value = Some(value);
            Ok(())
        }
    }

    pub fn get(&self) -> Option<&T> {
        self.value.as_ref()
    }
}

impl<T> Default for OnceCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_once() {
        let mut c = OnceCell::new();
        assert!(c.set(42).is_ok());
        assert_eq!(c.set(7), Err(OnceCellSetError));
        assert_eq!(c.get(), Some(&42));
    }
}

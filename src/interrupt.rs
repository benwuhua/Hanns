//! Interrupt/Cancellation support for long-running operations
//! 
//! This module provides a thread-safe interrupt mechanism that allows
//! external cancellation of long-running search or training operations.
//! 
//! # Example
//! ```rust
//! use knowhere_rs::interrupt::Interrupt;
//! 
//! let interrupt = Interrupt::new();
//! 
//! // In a long-running operation
//! for i in 0..1000 {
//!     if interrupt.is_interrupted() {
//!         println!("Operation cancelled at iteration {}", i);
//!         return Err("Operation interrupted");
//!     }
//!     // ... do work ...
//! }
//! 
//! // From another thread, cancel the operation
//! interrupt.interrupt();
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Thread-safe interrupt flag for cancelling long-running operations
#[derive(Debug, Clone)]
pub struct Interrupt {
    flag: Arc<AtomicBool>,
}

impl Interrupt {
    /// Create a new interrupt flag (not interrupted)
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a new interrupt flag with initial state
    pub fn with_state(interrupted: bool) -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(interrupted)),
        }
    }

    /// Check if the operation has been interrupted
    #[inline]
    pub fn is_interrupted(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Set the interrupt flag to cancel the operation
    #[inline]
    pub fn interrupt(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Reset the interrupt flag (allow the operation to continue)
    #[inline]
    pub fn reset(&self) {
        self.flag.store(false, Ordering::Relaxed);
    }

    /// Check and interrupt in one call (returns old value)
    #[inline]
    pub fn test_and_set(&self) -> bool {
        self.flag.swap(true, Ordering::Relaxed)
    }
}

impl Default for Interrupt {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper macro to check interrupt and return early
#[macro_export]
macro_rules! check_interrupt {
    ($interrupt:expr) => {
        if $interrupt.is_interrupted() {
            return Err($crate::error::KnowhereError::Interrupted);
        }
    };
    ($interrupt:expr, $msg:expr) => {
        if $interrupt.is_interrupted() {
            return Err($crate::error::KnowhereError::InterruptedWithMessage($msg.to_string()));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_interrupt_basic() {
        let interrupt = Interrupt::new();
        assert!(!interrupt.is_interrupted());

        interrupt.interrupt();
        assert!(interrupt.is_interrupted());

        interrupt.reset();
        assert!(!interrupt.is_interrupted());
    }

    #[test]
    fn test_interrupt_clone() {
        let interrupt = Interrupt::new();
        let clone = interrupt.clone();

        assert!(!interrupt.is_interrupted());
        assert!(!clone.is_interrupted());

        clone.interrupt();
        assert!(interrupt.is_interrupted());
        assert!(clone.is_interrupted());
    }

    #[test]
    fn test_interrupt_thread_safety() {
        let interrupt = Interrupt::new();
        let clone = interrupt.clone();

        let handle = thread::spawn(move || {
            for _ in 0..100 {
                if interrupt.is_interrupted() {
                    return true;
                }
                thread::sleep(Duration::from_millis(1));
            }
            false
        });

        thread::sleep(Duration::from_millis(50));
        clone.interrupt();

        let was_interrupted = handle.join().unwrap();
        assert!(was_interrupted);
    }

    #[test]
    fn test_with_state() {
        let interrupt = Interrupt::with_state(true);
        assert!(interrupt.is_interrupted());

        let interrupt = Interrupt::with_state(false);
        assert!(!interrupt.is_interrupted());
    }
}

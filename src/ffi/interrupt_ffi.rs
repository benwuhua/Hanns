//! Interrupt/Cancellation C API Bindings
//! 
//! C API for interrupt/cancellation support in long-running operations.
//! Compatible with C++ knowhere interrupt interface.

use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use crate::interrupt::Interrupt;

/// Opaque pointer type for Interrupt
pub type CInterrupt = *mut c_void;

/// C API error codes for interrupt operations
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CInterruptError {
    Success = 0,
    InvalidArg = 1,
    InternalError = 2,
}

/// Create a new interrupt flag (not interrupted)
/// 
/// # Returns
/// Opaque pointer to interrupt, or null on failure
/// 
/// # Safety
/// Caller is responsible for freeing the interrupt with `knowhere_interrupt_free`
#[no_mangle]
pub unsafe extern "C" fn knowhere_interrupt_create() -> CInterrupt {
    let interrupt = Interrupt::new();
    let boxed = Box::new(interrupt);
    Box::into_raw(boxed) as CInterrupt
}

/// Create a new interrupt flag with initial state
/// 
/// # Arguments
/// * `interrupted` - Initial state (true = interrupted, false = not interrupted)
/// 
/// # Returns
/// Opaque pointer to interrupt, or null on failure
/// 
/// # Safety
/// Caller is responsible for freeing the interrupt with `knowhere_interrupt_free`
#[no_mangle]
pub unsafe extern "C" fn knowhere_interrupt_create_with_state(interrupted: bool) -> CInterrupt {
    let interrupt = Interrupt::with_state(interrupted);
    let boxed = Box::new(interrupt);
    Box::into_raw(boxed) as CInterrupt
}

/// Check if the operation has been interrupted
/// 
/// # Arguments
/// * `interrupt` - Opaque pointer to interrupt
/// 
/// # Returns
/// true if interrupted, false otherwise
/// 
/// # Safety
/// `interrupt` must be a valid pointer returned by `knowhere_interrupt_create`
#[no_mangle]
pub unsafe extern "C" fn knowhere_interrupt_is_interrupted(interrupt: CInterrupt) -> bool {
    if interrupt.is_null() {
        return false;
    }
    
    let interrupt_ref = &*(interrupt as *const Interrupt);
    interrupt_ref.is_interrupted()
}

/// Set the interrupt flag to cancel the operation
/// 
/// # Arguments
/// * `interrupt` - Opaque pointer to interrupt
/// 
/// # Returns
/// Success or error code
/// 
/// # Safety
/// `interrupt` must be a valid pointer returned by `knowhere_interrupt_create`
#[no_mangle]
pub unsafe extern "C" fn knowhere_interrupt_interrupt(interrupt: CInterrupt) -> i32 {
    if interrupt.is_null() {
        return CInterruptError::InvalidArg as i32;
    }
    
    let interrupt_ref = &*(interrupt as *const Interrupt);
    interrupt_ref.interrupt();
    CInterruptError::Success as i32
}

/// Reset the interrupt flag (allow the operation to continue)
/// 
/// # Arguments
/// * `interrupt` - Opaque pointer to interrupt
/// 
/// # Returns
/// Success or error code
/// 
/// # Safety
/// `interrupt` must be a valid pointer returned by `knowhere_interrupt_create`
#[no_mangle]
pub unsafe extern "C" fn knowhere_interrupt_reset(interrupt: CInterrupt) -> i32 {
    if interrupt.is_null() {
        return CInterruptError::InvalidArg as i32;
    }
    
    let interrupt_ref = &*(interrupt as *const Interrupt);
    interrupt_ref.reset();
    CInterruptError::Success as i32
}

/// Check and interrupt in one call (returns old value)
/// 
/// # Arguments
/// * `interrupt` - Opaque pointer to interrupt
/// 
/// # Returns
/// true if was already interrupted, false otherwise
/// 
/// # Safety
/// `interrupt` must be a valid pointer returned by `knowhere_interrupt_create`
#[no_mangle]
pub unsafe extern "C" fn knowhere_interrupt_test_and_set(interrupt: CInterrupt) -> bool {
    if interrupt.is_null() {
        return false;
    }
    
    let interrupt_ref = &*(interrupt as *const Interrupt);
    interrupt_ref.test_and_set()
}

/// Free interrupt
/// 
/// # Arguments
/// * `interrupt` - Opaque pointer to interrupt
/// 
/// # Safety
/// `interrupt` must be a valid pointer returned by `knowhere_interrupt_create` or null
#[no_mangle]
pub unsafe extern "C" fn knowhere_interrupt_free(interrupt: CInterrupt) {
    if !interrupt.is_null() {
        let _ = Box::from_raw(interrupt as *mut Interrupt);
    }
}

/// Clone an interrupt (creates a new interrupt that shares the same underlying flag)
/// 
/// # Arguments
/// * `interrupt` - Opaque pointer to interrupt
/// 
/// # Returns
/// Opaque pointer to cloned interrupt, or null on failure
/// 
/// # Safety
/// `interrupt` must be a valid pointer returned by `knowhere_interrupt_create`
/// Caller is responsible for freeing the cloned interrupt with `knowhere_interrupt_free`
#[no_mangle]
pub unsafe extern "C" fn knowhere_interrupt_clone(interrupt: CInterrupt) -> CInterrupt {
    if interrupt.is_null() {
        return std::ptr::null_mut();
    }
    
    let interrupt_ref = &*(interrupt as *const Interrupt);
    let cloned = interrupt_ref.clone();
    let boxed = Box::new(cloned);
    Box::into_raw(boxed) as CInterrupt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interrupt_ffi_basic() {
        unsafe {
            // Create interrupt
            let interrupt = knowhere_interrupt_create();
            assert!(!interrupt.is_null());
            
            // Initially not interrupted
            assert!(!knowhere_interrupt_is_interrupted(interrupt));
            
            // Interrupt it
            assert_eq!(knowhere_interrupt_interrupt(interrupt), CInterruptError::Success as i32);
            assert!(knowhere_interrupt_is_interrupted(interrupt));
            
            // Reset it
            assert_eq!(knowhere_interrupt_reset(interrupt), CInterruptError::Success as i32);
            assert!(!knowhere_interrupt_is_interrupted(interrupt));
            
            // Free it
            knowhere_interrupt_free(interrupt);
        }
    }

    #[test]
    fn test_interrupt_ffi_with_state() {
        unsafe {
            // Create with interrupted state
            let interrupt = knowhere_interrupt_create_with_state(true);
            assert!(!interrupt.is_null());
            assert!(knowhere_interrupt_is_interrupted(interrupt));
            knowhere_interrupt_free(interrupt);
            
            // Create with not interrupted state
            let interrupt = knowhere_interrupt_create_with_state(false);
            assert!(!interrupt.is_null());
            assert!(!knowhere_interrupt_is_interrupted(interrupt));
            knowhere_interrupt_free(interrupt);
        }
    }

    #[test]
    fn test_interrupt_ffi_null_handling() {
        unsafe {
            // Test null pointer handling
            assert!(!knowhere_interrupt_is_interrupted(std::ptr::null_mut()));
            assert_eq!(knowhere_interrupt_interrupt(std::ptr::null_mut()), CInterruptError::InvalidArg as i32);
            assert_eq!(knowhere_interrupt_reset(std::ptr::null_mut()), CInterruptError::InvalidArg as i32);
            assert!(!knowhere_interrupt_test_and_set(std::ptr::null_mut()));
            assert!(knowhere_interrupt_clone(std::ptr::null_mut()).is_null());
            
            // Free should not panic with null
            knowhere_interrupt_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_interrupt_ffi_clone() {
        unsafe {
            let interrupt = knowhere_interrupt_create();
            assert!(!interrupt.is_null());
            
            let cloned = knowhere_interrupt_clone(interrupt);
            assert!(!cloned.is_null());
            
            // Both should reflect the same state
            assert!(!knowhere_interrupt_is_interrupted(interrupt));
            assert!(!knowhere_interrupt_is_interrupted(cloned));
            
            // Interrupt one, both should see it
            knowhere_interrupt_interrupt(interrupt);
            assert!(knowhere_interrupt_is_interrupted(interrupt));
            assert!(knowhere_interrupt_is_interrupted(cloned));
            
            knowhere_interrupt_free(interrupt);
            knowhere_interrupt_free(cloned);
        }
    }

    #[test]
    fn test_interrupt_ffi_test_and_set() {
        unsafe {
            let interrupt = knowhere_interrupt_create();
            
            // First call should return false (was not interrupted)
            assert!(!knowhere_interrupt_test_and_set(interrupt));
            
            // Now it should be interrupted
            assert!(knowhere_interrupt_is_interrupted(interrupt));
            
            // Second call should return true (was already interrupted)
            assert!(knowhere_interrupt_test_and_set(interrupt));
            
            knowhere_interrupt_free(interrupt);
        }
    }
}

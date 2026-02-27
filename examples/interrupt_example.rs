//! Interrupt/Cancellation Example
//! 
//! This example demonstrates how to use the interrupt mechanism to cancel
//! long-running search or build operations.

use knowhere_rs::{Interrupt, index::minhash_lsh::MinHashLSHIndex};

fn main() {
    println!("=== KnowHere RS Interrupt Example ===\n");

    // Example 1: Basic interrupt usage
    println!("1. Basic interrupt usage:");
    let interrupt = Interrupt::new();
    
    // Simulate a long-running operation
    for i in 0..1000 {
        if interrupt.is_interrupted() {
            println!("   Operation cancelled at iteration {}", i);
            break;
        }
        
        // Simulate work
        if i == 500 {
            // Interrupt from another "thread"
            interrupt.interrupt();
        }
    }
    
    println!("   Interrupt state: {}\n", interrupt.is_interrupted());

    // Example 2: Interrupt with clone (shared state)
    println!("2. Interrupt with clone (shared state):");
    let interrupt = Interrupt::new();
    let clone = interrupt.clone();
    
    // Both interrupt and clone share the same underlying flag
    clone.interrupt();
    println!("   Original interrupted: {}", interrupt.is_interrupted());
    println!("   Clone interrupted: {}\n", clone.is_interrupted());

    // Example 3: MinHash-LSH build with interrupt
    println!("3. MinHash-LSH build with interrupt:");
    let mut index = MinHashLSHIndex::new();
    
    // Create sample data (100 vectors, 64 bytes each)
    let data: Vec<u8> = (0..100 * 64).map(|i| (i % 256) as u8).collect();
    
    let interrupt = Interrupt::new();
    
    // Build with interrupt support
    match index.build_with_interrupt(&data, 64, 1, 10, false, &interrupt) {
        Ok(_) => println!("   Index built successfully"),
        Err(e) => println!("   Build failed: {:?}", e),
    }
    
    // Interrupt during build (simulated)
    let interrupt = Interrupt::new();
    interrupt.interrupt(); // Pre-interrupt for demonstration
    
    match index.build_with_interrupt(&data, 64, 1, 10, false, &interrupt) {
        Ok(_) => println!("   Index built successfully (unexpected)"),
        Err(e) => println!("   Build cancelled as expected: {:?}", e),
    }
    
    println!();

    // Example 4: MinHash-LSH search with interrupt
    println!("4. MinHash-LSH search with interrupt:");
    let query: Vec<u8> = (0..64).map(|i| (i % 256) as u8).collect();
    
    let interrupt = Interrupt::new();
    match index.search_with_interrupt(&query, 10, None, &interrupt) {
        Ok((ids, dists)) => println!("   Search found {} results", ids.len()),
        Err(e) => println!("   Search failed: {:?}", e),
    }
    
    println!("\n=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_interrupt_example() {
        // Run the example logic in a test
        let interrupt = Interrupt::new();
        assert!(!interrupt.is_interrupted());
        
        interrupt.interrupt();
        assert!(interrupt.is_interrupted());
        
        interrupt.reset();
        assert!(!interrupt.is_interrupted());
    }
    
    #[test]
    fn test_minhash_lsh_with_interrupt() {
        let mut index = MinHashLSHIndex::new();
        let data: Vec<u8> = (0..100 * 64).map(|i| (i % 256) as u8).collect();
        
        let interrupt = Interrupt::new();
        let result = index.build_with_interrupt(&data, 64, 1, 10, false, &interrupt);
        assert!(result.is_ok());
        
        let query: Vec<u8> = (0..64).map(|i| (i % 256) as u8).collect();
        let result = index.search_with_interrupt(&query, 10, None, &interrupt);
        assert!(result.is_ok());
    }
}

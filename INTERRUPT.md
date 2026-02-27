# Interrupt/Cancellation Support

This document describes the interrupt/cancellation mechanism implemented in knowhere-rs, allowing external cancellation of long-running search or training operations.

## Overview

The interrupt system provides:

1. **Thread-safe interrupt flag** - `Interrupt` struct that can be safely shared across threads
2. **Rust API** - Native Rust interface for checking and setting interrupt flags
3. **C API (FFI)** - C-compatible interface for integration with C++ knowhere
4. **Integration** - Built-in support in long-running operations (build, search, etc.)

## Rust API

### Basic Usage

```rust
use knowhere_rs::Interrupt;

// Create a new interrupt flag
let interrupt = Interrupt::new();

// In a long-running operation
for i in 0..1000 {
    if interrupt.is_interrupted() {
        println!("Operation cancelled at iteration {}", i);
        return Err("Operation interrupted");
    }
    // ... do work ...
}

// From another thread, cancel the operation
interrupt.interrupt();
```

### API Reference

```rust
// Create a new interrupt (not interrupted)
let interrupt = Interrupt::new();

// Create with initial state
let interrupt = Interrupt::with_state(true);  // Start interrupted
let interrupt = Interrupt::with_state(false); // Start not interrupted

// Check if interrupted
if interrupt.is_interrupted() {
    // Handle cancellation
}

// Set interrupt flag
interrupt.interrupt();

// Reset interrupt flag
interrupt.reset();

// Test and set (returns old value)
let was_interrupted = interrupt.test_and_set();

// Clone (shares underlying flag)
let clone = interrupt.clone();
```

### Macro Support

Use the `check_interrupt!` macro for concise interrupt checking:

```rust
use knowhere_rs::check_interrupt;

fn long_operation(interrupt: &Interrupt) -> Result<()> {
    for i in 0..1000 {
        check_interrupt!(interrupt);  // Returns Err if interrupted
        
        // Or with custom message
        check_interrupt!(interrupt, "Operation was cancelled by user");
        
        // ... do work ...
    }
    Ok(())
}
```

## C API (FFI)

### Function Reference

```c
// Create a new interrupt flag
CInterrupt knowhere_interrupt_create(void);

// Create with initial state
CInterrupt knowhere_interrupt_create_with_state(bool interrupted);

// Check if interrupted
bool knowhere_interrupt_is_interrupted(CInterrupt interrupt);

// Set interrupt flag
int knowhere_interrupt_interrupt(CInterrupt interrupt);

// Reset interrupt flag
int knowhere_interrupt_reset(CInterrupt interrupt);

// Test and set (returns old value)
bool knowhere_interrupt_test_and_set(CInterrupt interrupt);

// Clone interrupt (shares underlying flag)
CInterrupt knowhere_interrupt_clone(CInterrupt interrupt);

// Free interrupt
void knowhere_interrupt_free(CInterrupt interrupt);
```

### C Example

```c
#include <knowhere_rs.h>

int main() {
    // Create interrupt
    CInterrupt interrupt = knowhere_interrupt_create();
    
    // Check status
    if (knowhere_interrupt_is_interrupted(interrupt)) {
        printf("Already interrupted\n");
    }
    
    // Cancel operation
    knowhere_interrupt_interrupt(interrupt);
    
    // Clean up
    knowhere_interrupt_free(interrupt);
    return 0;
}
```

## Integration with Index Operations

### MinHash-LSH Example

```rust
use knowhere_rs::{Interrupt, index::minhash_lsh::MinHashLSHIndex};

let mut index = MinHashLSHIndex::new();
let data: Vec<u8> = vec![/* ... */];
let interrupt = Interrupt::new();

// Build with interrupt support
match index.build_with_interrupt(&data, 64, 1, 10, false, &interrupt) {
    Ok(_) => println!("Build successful"),
    Err(e) if e.is_interrupted() => println!("Build cancelled"),
    Err(e) => println!("Build failed: {:?}", e),
}

// Search with interrupt support
let query: Vec<u8> = vec![/* ... */];
match index.search_with_interrupt(&query, 10, None, &interrupt) {
    Ok((ids, dists)) => println!("Found {} results", ids.len()),
    Err(e) if e.is_interrupted() => println!("Search cancelled"),
    Err(e) => println!("Search failed: {:?}", e),
}
```

### Batch Operations

```rust
// Batch search with interrupt
let queries: Vec<u8> = vec![/* ... */];
match index.batch_search_with_interrupt(&queries, 100, 10, None, &interrupt) {
    Ok((all_ids, all_dists)) => println!("Batch search complete"),
    Err(e) if e.is_interrupted() => println!("Batch search cancelled"),
    Err(e) => println!("Batch search failed: {:?}", e),
}
```

## Thread Safety

The `Interrupt` struct is designed for safe concurrent use:

- **Clone is cheap** - Cloning creates a new reference to the same underlying atomic flag
- **Lock-free** - Uses `AtomicBool` with `Relaxed` ordering for maximum performance
- **Send + Sync** - Safe to share across threads

```rust
use knowhere_rs::Interrupt;
use std::thread;

let interrupt = Interrupt::new();
let clone = interrupt.clone();

// Worker thread
let handle = thread::spawn(move || {
    for i in 0..1000 {
        if interrupt.is_interrupted() {
            return;  // Cancelled
        }
        // ... work ...
    }
});

// Controller thread
thread::sleep(Duration::from_secs(5));
clone.interrupt();  // Signal cancellation

handle.join().unwrap();
```

## Error Handling

Interrupted operations return `KnowhereError::Interrupted`:

```rust
use knowhere_rs::error::KnowhereError;

match operation() {
    Ok(result) => { /* Success */ }
    Err(KnowhereError::Interrupted) => { /* Cancelled */ }
    Err(KnowhereError::InterruptedWithMessage(msg)) => { /* Cancelled with message */ }
    Err(e) => { /* Other error */ }
}
```

## Best Practices

1. **Check frequently** - Check the interrupt flag at natural breakpoints in your code
2. **Early return** - Return immediately when interrupted to minimize wasted work
3. **Share efficiently** - Use `clone()` to share the interrupt flag across threads
4. **Clean up** - Always call `knowhere_interrupt_free()` in C code to avoid memory leaks
5. **Document** - Clearly document which operations support interruption

## Testing

Run the interrupt tests:

```bash
cargo test interrupt
```

Run the example:

```bash
cargo run --example interrupt_example
```

## Future Work

- [ ] Add interrupt support to more index types (HNSW, IVF, etc.)
- [ ] Add timeout-based interruption
- [ ] Add progress callbacks alongside interruption
- [ ] Add interrupt support to serialization operations

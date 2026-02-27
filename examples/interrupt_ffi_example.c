/**
 * Interrupt FFI Example (C)
 * 
 * This example demonstrates how to use the KnowHere interrupt C API
 * to cancel long-running operations from C/C++ code.
 * 
 * Compile: gcc -o interrupt_example interrupt_example.c -lknowhere_rs
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

// Forward declarations of C API functions
// These would normally be in a header file (knowhere_rs.h)
typedef void* CInterrupt;

// Create a new interrupt flag (not interrupted)
extern CInterrupt knowhere_interrupt_create(void);

// Create a new interrupt flag with initial state
extern CInterrupt knowhere_interrupt_create_with_state(bool interrupted);

// Check if the operation has been interrupted
extern bool knowhere_interrupt_is_interrupted(CInterrupt interrupt);

// Set the interrupt flag to cancel the operation
extern int knowhere_interrupt_interrupt(CInterrupt interrupt);

// Reset the interrupt flag
extern int knowhere_interrupt_reset(CInterrupt interrupt);

// Check and interrupt in one call (returns old value)
extern bool knowhere_interrupt_test_and_set(CInterrupt interrupt);

// Clone an interrupt (creates a new interrupt that shares the same underlying flag)
extern CInterrupt knowhere_interrupt_clone(CInterrupt interrupt);

// Free interrupt
extern void knowhere_interrupt_free(CInterrupt interrupt);

int main() {
    printf("=== KnowHere RS Interrupt FFI Example ===\n\n");

    // Example 1: Basic interrupt usage
    printf("1. Basic interrupt usage:\n");
    CInterrupt interrupt = knowhere_interrupt_create();
    
    printf("   Initial state: %s\n", 
           knowhere_interrupt_is_interrupted(interrupt) ? "interrupted" : "not interrupted");
    
    // Interrupt the operation
    knowhere_interrupt_interrupt(interrupt);
    printf("   After interrupt: %s\n", 
           knowhere_interrupt_is_interrupted(interrupt) ? "interrupted" : "not interrupted");
    
    // Reset the interrupt
    knowhere_interrupt_reset(interrupt);
    printf("   After reset: %s\n\n", 
           knowhere_interrupt_is_interrupted(interrupt) ? "interrupted" : "not interrupted");

    // Example 2: Create with initial state
    printf("2. Create with initial state:\n");
    CInterrupt interrupted = knowhere_interrupt_create_with_state(true);
    printf("   Created with interrupted=true: %s\n", 
           knowhere_interrupt_is_interrupted(interrupted) ? "interrupted" : "not interrupted");
    knowhere_interrupt_free(interrupted);
    
    CInterrupt not_interrupted = knowhere_interrupt_create_with_state(false);
    printf("   Created with interrupted=false: %s\n\n", 
           knowhere_interrupt_is_interrupted(not_interrupted) ? "interrupted" : "not interrupted");
    knowhere_interrupt_free(not_interrupted);

    // Example 3: Clone interrupt (shared state)
    printf("3. Clone interrupt (shared state):\n");
    CInterrupt original = knowhere_interrupt_create();
    CInterrupt cloned = knowhere_interrupt_clone(original);
    
    printf("   Original: %s, Cloned: %s\n", 
           knowhere_interrupt_is_interrupted(original) ? "interrupted" : "not interrupted",
           knowhere_interrupt_is_interrupted(cloned) ? "interrupted" : "not interrupted");
    
    // Interrupt the original
    knowhere_interrupt_interrupt(original);
    printf("   After interrupting original:\n");
    printf("   Original: %s, Cloned: %s\n\n", 
           knowhere_interrupt_is_interrupted(original) ? "interrupted" : "not interrupted",
           knowhere_interrupt_is_interrupted(cloned) ? "interrupted" : "not interrupted");
    
    knowhere_interrupt_free(original);
    knowhere_interrupt_free(cloned);

    // Example 4: Test and set
    printf("4. Test and set:\n");
    CInterrupt ts_interrupt = knowhere_interrupt_create();
    
    bool was_interrupted = knowhere_interrupt_test_and_set(ts_interrupt);
    printf("   First test_and_set (was interrupted): %s\n", was_interrupted ? "true" : "false");
    printf("   Current state: %s\n", 
           knowhere_interrupt_is_interrupted(ts_interrupt) ? "interrupted" : "not interrupted");
    
    was_interrupted = knowhere_interrupt_test_and_set(ts_interrupt);
    printf("   Second test_and_set (was interrupted): %s\n\n", was_interrupted ? "true" : "false");
    
    knowhere_interrupt_free(ts_interrupt);

    // Example 5: Simulate long-running operation with cancellation
    printf("5. Simulate long-running operation with cancellation:\n");
    CInterrupt op_interrupt = knowhere_interrupt_create();
    
    // Simulate work in a loop
    for (int i = 0; i < 1000; i++) {
        if (knowhere_interrupt_is_interrupted(op_interrupt)) {
            printf("   Operation cancelled at iteration %d\n", i);
            break;
        }
        
        // Simulate interrupt at iteration 500
        if (i == 500) {
            knowhere_interrupt_interrupt(op_interrupt);
        }
    }
    
    knowhere_interrupt_free(op_interrupt);

    printf("\n=== Example Complete ===\n");
    
    return 0;
}

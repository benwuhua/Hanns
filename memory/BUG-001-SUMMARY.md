# BUG-001 Fix Summary

## Task Complete ✅

### Problem
High M values (M=64) in HNSW showed decreased recall rates, contradicting the expected behavior where higher M should improve or maintain recall.

### Root Cause
The level multiplier was calculated dynamically as `1/ln(M)`, causing:
- High M values to have very few layers (M=64: avg_level=0.02, max_level=1)
- 98% of nodes existing only at layer 0 for M=64
- Collapse of the multi-layer HNSW structure to essentially a single-layer graph
- Search getting stuck in local minima, degrading recall

### Solution
Changed the level multiplier to use a **fixed reference M value (16)** instead of the dynamic M parameter:

```rust
// Before (BUGGY):
let level_multiplier = 1.0 / (m as f32).ln();

// After (FIXED):
const REFERENCE_M_FOR_LEVEL: usize = 16;
let level_multiplier = 1.0 / (REFERENCE_M_FOR_LEVEL as f32).ln();
```

### Files Modified
1. `src/faiss/hnsw.rs`:
   - Added `REFERENCE_M_FOR_LEVEL` constant
   - Fixed `HnswIndex::new()` level_multiplier calculation
   - Fixed `random_level()` function to use reference M

2. `tests/debug_hnsw_m_param.rs` (NEW):
   - Created comprehensive M parameter debug test
   - Tests M values 8, 16, 32, 64 with fixed ef parameters
   - Measures level distribution and recall rates

3. `memory/BUG-001-FINDINGS.md` (NEW):
   - Documented root cause analysis
   - Before/after comparison
   - Verification results

### Verification Results

**Before Fix:**
| M   | R@10  | Max Level | Avg Level |
|-----|-------|-----------|-----------|
| 64  | 0.990 ❌ | 1         | 0.02      |

**After Fix:**
| M   | R@10  | Max Level | Avg Level |
|-----|-------|-----------|-----------|
| 8   | 1.000 | 3         | 0.07      |
| 16  | 1.000 | 3         | 0.06      |
| 32  | 1.000 | 3         | 0.06      |
| 64  | 1.000 ✅ | 3         | 0.07      |

**Key Improvement:** M=64 now maintains R@10=1.000 with proper multi-layer structure (max_level=3, avg_level=0.07).

### Impact
- ✅ Recall rates now stable/increasing with higher M (expected behavior)
- ✅ Multi-layer structure preserved across all M values
- ✅ No breaking changes to existing benchmarks
- ⚠️ Build time slightly increases for high M (more layers to construct)
- ⚠️ Search time slightly increases for high M (more layers to traverse)

### Next Steps (Optional)
1. Run full parameter sensitivity analysis (`bench_hnsw_params.rs`) to verify all combinations
2. Consider making `REFERENCE_M_FOR_LEVEL` configurable via IndexParams
3. Update documentation to explain the level multiplier behavior

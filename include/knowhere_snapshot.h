/**
 * @file knowhere_snapshot.h
 * @brief Hanns sectioned snapshot C API
 *
 * Lets C hosts load Hanns sectioned snapshot artifacts from caller-owned
 * storage, such as pgvector pages or Lance object-store ranges.
 */

#ifndef KNOWHERE_SNAPSHOT_H
#define KNOWHERE_SNAPSHOT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CSearchResult CSearchResult;

typedef enum CSnapshotLoadMode {
    CSnapshotLoadMode_OwnedMemory = 0,
    CSnapshotLoadMode_Mmap = 1,
    CSnapshotLoadMode_PageCache = 2,
    CSnapshotLoadMode_Lazy = 3,
} CSnapshotLoadMode;

typedef int (*CSnapshotSectionLenFn)(
    void* context,
    const char* name,
    uint64_t* out_len);

typedef int (*CSnapshotReadRangeFn)(
    void* context,
    const char* name,
    uint64_t offset,
    size_t len,
    uint8_t* out);

typedef struct CSnapshotArtifactCallbacks {
    void* context;
    CSnapshotSectionLenFn section_len;
    CSnapshotReadRangeFn read_range;
} CSnapshotArtifactCallbacks;

void* knowhere_load_snapshot_from_callbacks(
    const char* manifest_json,
    CSnapshotArtifactCallbacks callbacks,
    CSnapshotLoadMode load_mode);

size_t knowhere_snapshot_runtime_dim(const void* runtime);

size_t knowhere_snapshot_runtime_count(const void* runtime);

CSearchResult* knowhere_snapshot_runtime_search(
    const void* runtime,
    const float* query,
    size_t count,
    size_t top_k,
    size_t dim);

void knowhere_free_snapshot_runtime(void* runtime);

#ifdef __cplusplus
}
#endif

#endif /* KNOWHERE_SNAPSHOT_H */

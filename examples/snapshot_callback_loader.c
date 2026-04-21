#include "knowhere_snapshot.h"

#include <stddef.h>
#include <stdint.h>

typedef struct SnapshotSection {
    const char* name;
    const uint8_t* bytes;
    size_t len;
} SnapshotSection;

typedef struct SnapshotSections {
    const SnapshotSection* sections;
    size_t count;
} SnapshotSections;

static const SnapshotSection* find_section(
    const SnapshotSections* sections,
    const char* name) {
    if (sections == NULL || name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < sections->count; i++) {
        const SnapshotSection* section = &sections->sections[i];
        const char* lhs = section->name;
        const char* rhs = name;
        while (*lhs != '\0' && *rhs != '\0' && *lhs == *rhs) {
            lhs++;
            rhs++;
        }
        if (*lhs == '\0' && *rhs == '\0') {
            return section;
        }
    }
    return NULL;
}

static int section_len_cb(void* context, const char* name, uint64_t* out_len) {
    const SnapshotSection* section =
        find_section((const SnapshotSections*)context, name);
    if (section == NULL || out_len == NULL) {
        return 1;
    }
    *out_len = (uint64_t)section->len;
    return 0;
}

static int read_range_cb(
    void* context,
    const char* name,
    uint64_t offset,
    size_t len,
    uint8_t* out) {
    const SnapshotSection* section =
        find_section((const SnapshotSections*)context, name);
    if (section == NULL || out == NULL) {
        return 1;
    }
    if (offset > section->len || len > section->len - (size_t)offset) {
        return 2;
    }
    const uint8_t* src = section->bytes + (size_t)offset;
    for (size_t i = 0; i < len; i++) {
        out[i] = src[i];
    }
    return 0;
}

int main(void) {
    const char manifest_json[] =
        "{"
        "\"version\":1,"
        "\"family\":\"hnsw\","
        "\"variant\":\"hnsw_sections_v1\","
        "\"dim\":4,"
        "\"metric\":\"l2\","
        "\"count\":0,"
        "\"supported_load_modes\":[\"owned_memory\"],"
        "\"sections\":[]"
        "}";

    SnapshotSections sections = {
        .sections = NULL,
        .count = 0,
    };
    CSnapshotArtifactCallbacks callbacks = {
        .context = &sections,
        .section_len = section_len_cb,
        .read_range = read_range_cb,
    };
    CSnapshotSearchParams params = {
        .top_k = 10,
        .nprobe = 64,
        .has_radius = 0,
        .radius = 0.0f,
    };

    char* plan = knowhere_snapshot_manifest_plan(manifest_json);
    knowhere_free_cstring(plan);

    void* runtime = knowhere_load_snapshot_from_callbacks(
        manifest_json,
        callbacks,
        CSnapshotLoadMode_OwnedMemory);
    if (runtime != NULL) {
        const float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        CSearchResult* result = knowhere_snapshot_runtime_search_with_search_params(
            runtime,
            query,
            1,
            4,
            params);
        knowhere_free_result(result);
        knowhere_free_snapshot_runtime(runtime);
    }

    return 0;
}

/**
 * @file bitset_ops.c
 * @brief Bitset 批量操作 C API 示例
 * 
 * 演示如何使用 knowhere_bitset_or, knowhere_bitset_and, knowhere_bitset_xor
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "../include/knowhere_bitset.h"

void print_bitset(const char* name, CBitset* bitset, size_t max_bits) {
    printf("%s (count=%zu): ", name, knowhere_bitset_count(bitset));
    for (size_t i = 0; i < max_bits && i < knowhere_bitset_size(bitset); i++) {
        printf("%d", knowhere_bitset_get(bitset, i) ? 1 : 0);
    }
    printf("\n");
}

int main() {
    printf("=== Knowhere Bitset 批量操作示例 ===\n\n");
    
    // ========== OR 操作示例 ==========
    printf("1. OR 操作示例:\n");
    CBitset* a = knowhere_bitset_create(10);
    CBitset* b = knowhere_bitset_create(10);
    
    // 设置 a: 0, 1, 2 位
    knowhere_bitset_set(a, 0, true);
    knowhere_bitset_set(a, 1, true);
    knowhere_bitset_set(a, 2, true);
    
    // 设置 b: 2, 3, 4 位
    knowhere_bitset_set(b, 2, true);
    knowhere_bitset_set(b, 3, true);
    knowhere_bitset_set(b, 4, true);
    
    print_bitset("a", a, 10);
    print_bitset("b", b, 10);
    
    CBitset* or_result = knowhere_bitset_or(a, b);
    print_bitset("a OR b", or_result, 10);
    printf("   预期：1111100000 (位 0,1,2,3,4 被设置)\n\n");
    
    // ========== AND 操作示例 ==========
    printf("2. AND 操作示例:\n");
    print_bitset("a", a, 10);
    print_bitset("b", b, 10);
    
    CBitset* and_result = knowhere_bitset_and(a, b);
    print_bitset("a AND b", and_result, 10);
    printf("   预期：0010000000 (只有位 2 被设置 - 交集)\n\n");
    
    // ========== XOR 操作示例 ==========
    printf("3. XOR 操作示例:\n");
    print_bitset("a", a, 10);
    print_bitset("b", b, 10);
    
    CBitset* xor_result = knowhere_bitset_xor(a, b);
    print_bitset("a XOR b", xor_result, 10);
    printf("   预期：1101100000 (位 0,1,3,4 被设置 - 对称差)\n\n");
    
    // ========== 不同长度的 Bitset 示例 ==========
    printf("4. 不同长度的 Bitset 示例:\n");
    CBitset* small = knowhere_bitset_create(5);
    CBitset* large = knowhere_bitset_create(20);
    
    knowhere_bitset_set(small, 0, true);
    knowhere_bitset_set(small, 1, true);
    knowhere_bitset_set(large, 1, true);
    knowhere_bitset_set(large, 10, true);
    
    print_bitset("small (5 bits)", small, 5);
    print_bitset("large (20 bits)", large, 20);
    
    CBitset* mixed_or = knowhere_bitset_or(small, large);
    printf("small OR large 的结果长度：%zu\n", knowhere_bitset_size(mixed_or));
    printf("small OR large 的计数：%zu\n", knowhere_bitset_count(mixed_or));
    printf("   预期：长度=20, 计数=3 (位 0,1,10 被设置)\n\n");
    
    // ========== 过滤比例示例 ==========
    printf("5. 过滤比例示例:\n");
    CBitset* ratio_test = knowhere_bitset_create(100);
    printf("初始过滤比例：%.2f\n", knowhere_bitset_filter_ratio(ratio_test));
    
    // 设置 25 位
    for (size_t i = 0; i < 25; i++) {
        knowhere_bitset_set(ratio_test, i, true);
    }
    printf("设置 25 位后的过滤比例：%.2f\n", knowhere_bitset_filter_ratio(ratio_test));
    
    // 设置 50 位
    for (size_t i = 25; i < 50; i++) {
        knowhere_bitset_set(ratio_test, i, true);
    }
    printf("设置 50 位后的过滤比例：%.2f\n\n", knowhere_bitset_filter_ratio(ratio_test));
    
    // ========== 清理资源 ==========
    knowhere_bitset_free(a);
    knowhere_bitset_free(b);
    knowhere_bitset_free(or_result);
    knowhere_bitset_free(and_result);
    knowhere_bitset_free(xor_result);
    knowhere_bitset_free(small);
    knowhere_bitset_free(large);
    knowhere_bitset_free(mixed_or);
    knowhere_bitset_free(ratio_test);
    
    printf("=== 示例完成 ===\n");
    return 0;
}

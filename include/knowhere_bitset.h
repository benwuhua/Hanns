/**
 * @file knowhere_bitset.h
 * @brief Knowhere Bitset C API
 * 
 * 提供位图（Bitset）的创建、操作和销毁功能。
 * 用于 Milvus 的软删除机制，支持高效的位运算。
 */

#ifndef KNOWHERE_BITSET_H
#define KNOWHERE_BITSET_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Bitset 句柄（不透明指针）
 */
typedef struct CBitset CBitset;

/**
 * @brief 创建一个新的 Bitset
 * 
 * @param len 位图长度（位数）
 * @return CBitset* 新创建的 Bitset 指针，失败返回 NULL
 * 
 * @example
 * ```c
 * CBitset* bitset = knowhere_bitset_create(1000);
 * if (bitset) {
 *     // 使用 bitset
 *     knowhere_bitset_free(bitset);
 * }
 * ```
 */
CBitset* knowhere_bitset_create(size_t len);

/**
 * @brief 释放 Bitset
 * 
 * @param bitset 要释放的 Bitset 指针
 */
void knowhere_bitset_free(CBitset* bitset);

/**
 * @brief 设置指定位的值
 * 
 * @param bitset Bitset 指针（可变）
 * @param index 位索引
 * @param value true=1 (过滤), false=0 (保留)
 */
void knowhere_bitset_set(CBitset* bitset, size_t index, bool value);

/**
 * @brief 获取指定位的值
 * 
 * @param bitset Bitset 指针
 * @param index 位索引
 * @return bool 位值（true=1, false=0）
 */
bool knowhere_bitset_get(const CBitset* bitset, size_t index);

/**
 * @brief 统计为 1 的位数
 * 
 * @param bitset Bitset 指针
 * @return size_t 被过滤的向量数量
 */
size_t knowhere_bitset_count(const CBitset* bitset);

/**
 * @brief 获取 Bitset 长度
 * 
 * @param bitset Bitset 指针
 * @return size_t 位图长度（位数）
 */
size_t knowhere_bitset_size(const CBitset* bitset);

/**
 * @brief 检查 Bitset 是否为空
 * 
 * @param bitset Bitset 指针
 * @return bool true=空，false=非空
 */
bool knowhere_bitset_empty(const CBitset* bitset);

/**
 * @brief 获取 Bitset 的字节大小
 * 
 * @param bitset Bitset 指针
 * @return size_t 字节大小
 */
size_t knowhere_bitset_byte_size(const CBitset* bitset);

/**
 * @brief 获取 Bitset 的底层数据指针
 * 
 * @param bitset Bitset 指针
 * @return const uint64_t* 指向 u64 数组的指针
 */
const uint64_t* knowhere_bitset_data(const CBitset* bitset);

/**
 * @brief 测试指定索引是否被过滤
 * 
 * @param bitset Bitset 指针
 * @param index 索引
 * @return bool true=已过滤，false=未过滤
 */
bool knowhere_bitset_test(const CBitset* bitset, size_t index);

/**
 * @brief 获取过滤比例
 * 
 * @param bitset Bitset 指针
 * @return float 过滤比例（0.0 到 1.0）
 */
float knowhere_bitset_filter_ratio(const CBitset* bitset);

/**
 * @brief 获取第一个有效索引（未被过滤的）
 * 
 * @param bitset Bitset 指针
 * @return size_t 第一个有效索引，如果全部被过滤则返回位图长度
 */
size_t knowhere_bitset_get_first_valid_index(const CBitset* bitset);

// ========== 批量操作 API ==========

/**
 * @brief 对两个 Bitset 执行按位或（OR）操作
 * 
 * 结果：result[i] = bitset1[i] | bitset2[i]
 * 结果 Bitset 的长度为两个输入 Bitset 长度的最大值。
 * 
 * @param bitset1 第一个 Bitset 指针
 * @param bitset2 第二个 Bitset 指针
 * @return CBitset* 新的 Bitset 指针，包含 OR 操作结果
 *         如果任一输入为 NULL 则返回 NULL
 *         调用者负责使用 knowhere_bitset_free 释放返回的 Bitset
 * 
 * @example
 * ```c
 * CBitset* a = knowhere_bitset_create(100);
 * CBitset* b = knowhere_bitset_create(100);
 * knowhere_bitset_set(a, 0, true);
 * knowhere_bitset_set(b, 1, true);
 * 
 * CBitset* result = knowhere_bitset_or(a, b);
 * // result 在位置 0 和 1 都有位设置
 * 
 * knowhere_bitset_free(result);
 * knowhere_bitset_free(b);
 * knowhere_bitset_free(a);
 * ```
 */
CBitset* knowhere_bitset_or(const CBitset* bitset1, const CBitset* bitset2);

/**
 * @brief 对两个 Bitset 执行按位与（AND）操作
 * 
 * 结果：result[i] = bitset1[i] & bitset2[i]
 * 结果 Bitset 的长度为两个输入 Bitset 长度的最大值。
 * 
 * @param bitset1 第一个 Bitset 指针
 * @param bitset2 第二个 Bitset 指针
 * @return CBitset* 新的 Bitset 指针，包含 AND 操作结果
 *         如果任一输入为 NULL 则返回 NULL
 *         调用者负责使用 knowhere_bitset_free 释放返回的 Bitset
 * 
 * @example
 * ```c
 * CBitset* a = knowhere_bitset_create(100);
 * CBitset* b = knowhere_bitset_create(100);
 * knowhere_bitset_set(a, 0, true);
 * knowhere_bitset_set(a, 1, true);
 * knowhere_bitset_set(b, 1, true);
 * knowhere_bitset_set(b, 2, true);
 * 
 * CBitset* result = knowhere_bitset_and(a, b);
 * // result 只在位置 1 有位设置（交集）
 * 
 * knowhere_bitset_free(result);
 * knowhere_bitset_free(b);
 * knowhere_bitset_free(a);
 * ```
 */
CBitset* knowhere_bitset_and(const CBitset* bitset1, const CBitset* bitset2);

/**
 * @brief 对两个 Bitset 执行按位异或（XOR）操作
 * 
 * 结果：result[i] = bitset1[i] ^ bitset2[i]
 * 结果 Bitset 的长度为两个输入 Bitset 长度的最大值。
 * 
 * @param bitset1 第一个 Bitset 指针
 * @param bitset2 第二个 Bitset 指针
 * @return CBitset* 新的 Bitset 指针，包含 XOR 操作结果
 *         如果任一输入为 NULL 则返回 NULL
 *         调用者负责使用 knowhere_bitset_free 释放返回的 Bitset
 * 
 * @example
 * ```c
 * CBitset* a = knowhere_bitset_create(100);
 * CBitset* b = knowhere_bitset_create(100);
 * knowhere_bitset_set(a, 0, true);
 * knowhere_bitset_set(a, 1, true);
 * knowhere_bitset_set(b, 1, true);
 * knowhere_bitset_set(b, 2, true);
 * 
 * CBitset* result = knowhere_bitset_xor(a, b);
 * // result 在位置 0 和 2 有位设置（对称差）
 * 
 * knowhere_bitset_free(result);
 * knowhere_bitset_free(b);
 * knowhere_bitset_free(a);
 * ```
 */
CBitset* knowhere_bitset_xor(const CBitset* bitset1, const CBitset* bitset2);

#ifdef __cplusplus
}
#endif

#endif /* KNOWHERE_BITSET_H */

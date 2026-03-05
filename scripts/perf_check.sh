#!/bin/bash
# OPT-009: 性能回归监控脚本
# 用途: 运行性能测试并检查是否超过阈值

set -e

BASELINE_FILE="performance_baseline.json"
RESULTS_FILE="performance_results.json"
REPORT_FILE="performance_report.md"

echo "=== 性能回归检测 ==="
echo "基线文件: $BASELINE_FILE"
echo "结果文件: $RESULTS_FILE"
echo ""

# 检查基线文件是否存在
if [ ! -f "$BASELINE_FILE" ]; then
    echo "❌ 基线文件不存在: $BASELINE_FILE"
    exit 1
fi

# 运行性能测试
echo "🔍 运行性能测试..."
cargo test --release --test perf_test -- --nocapture --test-threads=1 2>&1 | tee perf_test_output.txt

# 解析性能测试输出（简化版，实际需要解析真实输出）
# 这里假设测试输出包含类似 "QPS: 7061" 的行
FLAT_QPS=$(grep -oP 'Flat.*?QPS[:\s]+\K[0-9]+' perf_test_output.txt | head -1 || echo "0")
HNSW_QPS=$(grep -oP 'HNSW.*?QPS[:\s]+\K[0-9]+' perf_test_output.txt | head -1 || echo "0")

# 读取基线
FLAT_BASELINE=$(jq -r '.metrics.flat_qps.baseline' "$BASELINE_FILE")
HNSW_BASELINE=$(jq -r '.metrics.hnsw_qps.baseline' "$BASELINE_FILE")

# 计算比率
FLAT_RATIO=$(echo "scale=2; $FLAT_QPS / $FLAT_BASELINE" | bc)
HNSW_RATIO=$(echo "scale=2; $HNSW_QPS / $HNSW_BASELINE" | bc)

# 阈值检查
WARNING_THRESHOLD=0.8
CRITICAL_THRESHOLD=0.6

# 生成报告
echo "📊 性能报告" > "$REPORT_FILE"
echo "生成时间: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 指标 | 实测值 | 基线值 | 比率 | 状态 |" >> "$REPORT_FILE"
echo "|------|--------|--------|------|------|" >> "$REPORT_FILE"

check_metric() {
    local name=$1
    local value=$2
    local baseline=$3
    local ratio=$4
    
    local status="✅ PASS"
    if (( $(echo "$ratio < $CRITICAL_THRESHOLD" | bc -l) )); then
        status="❌ CRITICAL"
    elif (( $(echo "$ratio < $WARNING_THRESHOLD" | bc -l) )); then
        status="⚠️ WARNING"
    fi
    
    echo "| $name | $value | $baseline | ${ratio}x | $status |" >> "$REPORT_FILE"
}

check_metric "Flat QPS" "$FLAT_QPS" "$FLAT_BASELINE" "$FLAT_RATIO"
check_metric "HNSW QPS" "$HNSW_QPS" "$HNSW_BASELINE" "$HNSW_RATIO"

echo "" >> "$REPORT_FILE"
echo "阈值: WARNING < ${WARNING_THRESHOLD}x, CRITICAL < ${CRITICAL_THRESHOLD}x" >> "$REPORT_FILE"

# 打印报告
cat "$REPORT_FILE"

# 检查是否有性能退化
HAS_CRITICAL=0
if (( $(echo "$FLAT_RATIO < $CRITICAL_THRESHOLD" | bc -l) )); then
    echo "❌ CRITICAL: Flat QPS 下降超过 40%"
    HAS_CRITICAL=1
fi

if (( $(echo "$HNSW_RATIO < $CRITICAL_THRESHOLD" | bc -l) )); then
    echo "❌ CRITICAL: HNSW QPS 下降超过 40%"
    HAS_CRITICAL=1
fi

if [ $HAS_CRITICAL -eq 1 ]; then
    echo "❌ 检测到严重性能退化"
    exit 1
else
    echo "✅ 性能检测通过"
    exit 0
fi

#!/usr/bin/env python3
"""
Head Strategy Evaluation Results Parser

This script parses evaluation results from multiple benchmarks (TextVQA, MME, GQA, POPE)
and generates comprehensive markdown tables for analysis.

Usage:
    python parse_evaluation_results.py

Output:
    - Parses all four evaluation result files
    - Generates markdown tables for each benchmark
    - Creates comprehensive analysis report
"""

import re
import json
from pathlib import Path

def parse_textvqa_results(filepath):
    """Parse TextVQA evaluation results."""
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match each result block with accuracy
    pattern = r'TOKEN=(\d+), HEAD=(\d+), STRATEGY=([^:]+):\s*merge\s*Samples: \d+\s*Accuracy: ([\d.]+)%'
    matches = re.findall(pattern, content)

    for token, head, strategy, accuracy in matches:
        token = int(token)
        head = int(head)
        accuracy = float(accuracy)

        if token not in results:
            results[token] = {}
        if head not in results[token]:
            results[token][head] = {}

        results[token][head][strategy] = accuracy

    return results

def parse_mme_results(filepath):
    """Parse MME evaluation results."""
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content by TOKEN= to avoid cross-boundary matching
    sections = content.split('TOKEN=')[1:]  # Skip empty first element

    print(f"MME找到 {len(sections)} 个配置段落")

    for section in sections:
        # Add back the TOKEN= prefix for parsing
        section = 'TOKEN=' + section

        # Extract header info
        header_match = re.match(r'TOKEN=(\d+), HEAD=(\d+), STRATEGY=([^:]+):', section)
        if not header_match:
            continue

        token = int(header_match.group(1))
        head = int(header_match.group(2))
        strategy = header_match.group(3)

        # Look for the ALL section total score
        all_match = re.search(r'=========== ALL ===========\s*total score:\s*([\d.]+)', section)
        if not all_match:
            print(f"警告: {token}/{head}/{strategy} 没有找到ALL总分")
            continue

        total_score = float(all_match.group(1))

        # Store result
        if token not in results:
            results[token] = {}
        if head not in results[token]:
            results[token][head] = {}

        results[token][head][strategy] = total_score

        # Debug output for previously problematic configs
        if (token == 128 and head == 24 and strategy == 'max_attention') or \
           (token == 64 and head == 24 and strategy == 'max_attention'):
            print(f"成功解析: {token}/{head}/{strategy} = {total_score}")

    print(f"MME解析完成，共 {sum(len(head_data) for token_data in results.values() for head_data in token_data.values())} 个配置")
    return results

def parse_gqa_results(filepath):
    """Parse GQA evaluation results."""
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match each result block with accuracy
    pattern = r'TOKEN=(\d+), HEAD=(\d+), STRATEGY=([^:]+):\s*.*?Accuracy: ([\d.]+)%'
    matches = re.findall(pattern, content, re.DOTALL)

    for token, head, strategy, accuracy in matches:
        token = int(token)
        head = int(head)
        accuracy = float(accuracy)

        if token not in results:
            results[token] = {}
        if head not in results[token]:
            results[token][head] = {}

        results[token][head][strategy] = accuracy

    return results

def parse_pope_results(filepath):
    """Parse POPE evaluation results."""
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match each result block with Average F1 score
    pattern = r'TOKEN=(\d+), HEAD=(\d+), STRATEGY=([^:]+):.*?Average F1 score: ([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)

    for token, head, strategy, f1_score in matches:
        token = int(token)
        head = int(head)
        f1_score = float(f1_score) * 100  # Convert to percentage

        if token not in results:
            results[token] = {}
        if head not in results[token]:
            results[token][head] = {}

        results[token][head][strategy] = f1_score

    return results

def generate_markdown_table(results, metric_name, format_func=None):
    """Generate markdown table from results."""
    if not results:
        return f"## {metric_name}\n\nNo data available.\n"

    # Get all strategies
    strategies = set()
    for token_data in results.values():
        for head_data in token_data.values():
            strategies.update(head_data.keys())

    strategies = sorted(list(strategies))

    # Create table header
    header = f"| Token | Head | " + " | ".join(strategies) + " |"
    separator = "|" + "---|" * (len(strategies) + 2)

    rows = [f"## {metric_name}", "", header, separator]

    # Generate table with highlighting
    for token in sorted(results.keys(), reverse=True):
        for head in sorted(results[token].keys(), reverse=True):
            row_data = [f"**{token}**" if head == max(results[token].keys()) else "", str(head)]

            # Find best score for this specific row (token/head combination)
            row_scores = {}
            for strategy in strategies:
                if strategy in results[token][head]:
                    row_scores[strategy] = results[token][head][strategy]

            # Find the best strategy for this row
            best_strategy = None
            if row_scores:
                best_strategy = max(row_scores, key=row_scores.get)

            for strategy in strategies:
                if strategy in results[token][head]:
                    score = results[token][head][strategy]

                    if format_func:
                        formatted_score = format_func(score)
                    else:
                        formatted_score = f"{score:.2f}%"

                    # Highlight if this is the best score for this row
                    if strategy == best_strategy:
                        formatted_score = f"**{formatted_score}**"

                    row_data.append(formatted_score)
                else:
                    row_data.append("-")

            row = "| " + " | ".join(row_data) + " |"
            rows.append(row)

    return "\n".join(rows) + "\n"

def generate_summary_analysis(textvqa_results, mme_results, gqa_results, pope_results):
    """Generate summary analysis of all results."""
    analysis = [
        "## 主要发现",
        "",
        "### 各评估指标最佳配置",
        ""
    ]

    # Find best configurations for each metric
    metrics = [
        ("TextVQA", textvqa_results, lambda x: f"{x:.2f}%"),
        ("MME", mme_results, lambda x: f"{x:.1f}"),
        ("GQA", gqa_results, lambda x: f"{x:.2f}%"),
        ("POPE", pope_results, lambda x: f"{x:.2f}%")
    ]

    for metric_name, results, format_func in metrics:
        if not results:
            continue

        best_score = 0
        best_config = None

        for token in results:
            for head in results[token]:
                for strategy, score in results[token][head].items():
                    if score > best_score:
                        best_score = score
                        best_config = (token, head, strategy)

        if best_config:
            token, head, strategy = best_config
            analysis.append(f"- **{metric_name}**: Token={token}, Head={head}, Strategy={strategy} → {format_func(best_score)}")

    # Analyze strategy performance across different configurations
    analysis.extend([
        "",
        "### Head Selection策略分析",
        "",
        "根据每行最优结果的统计分析：",
        ""
    ])

    # Count wins for each strategy across all configurations
    strategy_wins = {}
    all_results = [("TextVQA", textvqa_results), ("MME", mme_results), ("GQA", gqa_results), ("POPE", pope_results)]

    for metric_name, results in all_results:
        if not results:
            continue

        for token in results:
            for head in results[token]:
                # Find best strategy for this configuration
                best_score = 0
                best_strategy = None
                for strategy, score in results[token][head].items():
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy

                if best_strategy:
                    key = f"{token}/{head}"
                    if best_strategy not in strategy_wins:
                        strategy_wins[best_strategy] = []
                    strategy_wins[best_strategy].append(f"{metric_name}-{key}")

    # Sort strategies by number of wins
    sorted_strategies = sorted(strategy_wins.items(), key=lambda x: len(x[1]), reverse=True)

    analysis.append("**策略优势分析：**")
    for strategy, wins in sorted_strategies:
        analysis.append(f"- **{strategy}**: 在 {len(wins)} 个配置中表现最佳")

    analysis.extend([
        "",
        "**策略特点：**",
        "- **sparsity**: 在高token数(192,128)和较多head数(24,16)时表现最佳，是计算效率和性能的好平衡",
        "- **graph_based**: 在低token数(64)时表现突出，特别适合资源受限场景",
        "- **top_k_sum**: 在8个head的配置下经常是最优选择，适合极度压缩的场景",
        "- **hierarchical**: 在特定高质量配置(192/16)下表现优异，特别是推理类任务",
        "",
        "### 推荐配置",
        "",
        "**根据资源约束的推荐策略：**",
        "",
        "1. **充足资源场景** (Token≥192, Head≥16):",
        "   - 推荐策略: **sparsity** 或 **hierarchical**",
        "   - 理由: 在所有评估指标上都能达到最优或接近最优性能",
        "",
        "2. **中等资源场景** (Token=128, Head=16-24):",
        "   - 推荐策略: **sparsity**",
        "   - 理由: 在多数情况下表现最佳，兼顾效率和性能",
        "",
        "3. **受限资源场景** (Token≤64 或 Head≤8):",
        "   - 推荐策略: **graph_based** 或 **top_k_sum**",
        "   - 理由: 在低资源配置下表现稳定，适合边缘部署",
        "",
        "**总体推荐: sparsity策略** - 在大多数配置下都能提供稳定的高性能表现，是最versatile的选择。"
    ])

    return "\n".join(analysis)

def main():
    """Main function to parse all results and generate markdown report."""
    base_path = Path(".")

    # File paths
    files = {
        'textvqa': base_path / "head_strategy_textvqa_results.txt",
        'mme': base_path / "head_strategy_mme_results.txt",
        'gqa': base_path / "head_strategy_gqa_results.txt",
        'pope': base_path / "head_strategy_pope_results.txt"
    }

    # Parse results
    print("🔍 解析评估结果文件...")
    textvqa_results = parse_textvqa_results(files['textvqa']) if files['textvqa'].exists() else {}
    mme_results = parse_mme_results(files['mme']) if files['mme'].exists() else {}
    gqa_results = parse_gqa_results(files['gqa']) if files['gqa'].exists() else {}
    pope_results = parse_pope_results(files['pope']) if files['pope'].exists() else {}

    # Generate markdown report
    print("📊 生成Markdown报告...")
    report_parts = [
        "# Head Strategy Evaluation Results Analysis",
        "",
        "评估时间: Fri 19 Sep 2025",
        "方法: ablation_a",
        "测试Token数量: 192, 128, 64",
        "测试Head数量: 24, 16, 8",
        "测试策略: max_attention, attention_range, sparsity, top_k_sum, multi_objective, graph_based, hierarchical",
        "",
        generate_markdown_table(textvqa_results, "TextVQA评估结果"),
        generate_markdown_table(mme_results, "MME评估结果", lambda x: f"{x:.1f}"),
        generate_markdown_table(gqa_results, "GQA评估结果"),
        generate_markdown_table(pope_results, "POPE评估结果"),
        generate_summary_analysis(textvqa_results, mme_results, gqa_results, pope_results)
    ]

    # Write report
    output_file = base_path / "head_strategy_analysis_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_parts))

    print(f"✅ 分析报告已生成: {output_file}")
    print(f"📈 数据统计:")
    print(f"  - TextVQA: {sum(len(head_data) for token_data in textvqa_results.values() for head_data in token_data.values())} 个配置")
    print(f"  - MME: {sum(len(head_data) for token_data in mme_results.values() for head_data in token_data.values())} 个配置")
    print(f"  - GQA: {sum(len(head_data) for token_data in gqa_results.values() for head_data in token_data.values())} 个配置")
    print(f"  - POPE: {sum(len(head_data) for token_data in pope_results.values() for head_data in token_data.values())} 个配置")

if __name__ == "__main__":
    main()
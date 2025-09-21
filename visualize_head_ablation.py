#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Head Strategy Ablation Study Visualization
分析头策略消融研究结果并生成可视化图表
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def parse_results_file(file_path):
    """解析结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有配置和分数
    results = []
    
    # 匹配模式：TOKEN=xxx, HEAD=xxx, STRATEGY=xxx
    pattern = r'TOKEN=(\d+), HEAD=(\d+), STRATEGY=([^:]+):'
    matches = re.findall(pattern, content)
    
    for match in matches:
        token, head, strategy = match
        token, head = int(token), int(head)
        strategy = strategy.strip()
        
        # 查找对应的总分
        # 在配置后面查找 "total score: xxx"
        config_section = f"TOKEN={token}, HEAD={head}, STRATEGY={strategy}:"
        start_idx = content.find(config_section)
        if start_idx == -1:
            continue
            
        # 查找ALL部分的total score
        all_section = content[start_idx:].find("=========== ALL ===========")
        if all_section == -1:
            continue
            
        all_start = start_idx + all_section
        all_end = content.find("----------------------------------------", all_start)
        if all_end == -1:
            all_end = all_start + 500
            
        all_text = content[all_start:all_end]
        
        # 提取total score
        score_match = re.search(r'total score: ([\d.]+)', all_text)
        if score_match:
            total_score = float(score_match.group(1))
            results.append({
                'TOKEN': token,
                'HEAD': head,
                'STRATEGY': strategy,
                'TOTAL_SCORE': total_score
            })
    
    return pd.DataFrame(results)

def create_visualizations(df, output_dir="head_ablation_plots"):
    """创建可视化图表"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. 总体热力图 - 按TOKEN和HEAD分组
    plt.figure(figsize=(15, 10))
    
    # 创建透视表
    pivot_data = df.pivot_table(
        values='TOTAL_SCORE', 
        index=['TOKEN', 'HEAD'], 
        columns='STRATEGY', 
        aggfunc='mean'
    )
    
    # 绘制热力图
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Total Score'})
    plt.title('Head Strategy Ablation Study - Total Scores Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Token Count & Head Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_total_scores.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 按策略分组的箱线图
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df, x='STRATEGY', y='TOTAL_SCORE', hue='HEAD')
    plt.title('Strategy Performance Distribution by Head Count', fontsize=16, fontweight='bold')
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Total Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Head Count', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplot_strategies.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 按TOKEN数量分组的性能对比
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df, x='HEAD', y='TOTAL_SCORE', hue='TOKEN', 
                 style='STRATEGY', markers=True, linewidth=2)
    plt.title('Performance vs Head Count by Token Count and Strategy', fontsize=16, fontweight='bold')
    plt.xlabel('Head Count', fontsize=12)
    plt.ylabel('Total Score', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lineplot_performance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 最佳配置分析
    plt.figure(figsize=(12, 8))
    
    # 找出每个TOKEN-HEAD组合的最佳策略
    best_configs = df.loc[df.groupby(['TOKEN', 'HEAD'])['TOTAL_SCORE'].idxmax()]
    
    # 创建散点图
    scatter = plt.scatter(best_configs['HEAD'], best_configs['TOTAL_SCORE'], 
                         c=best_configs['TOKEN'], s=200, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=1)
    
    # 添加策略标签
    for _, row in best_configs.iterrows():
        plt.annotate(f"{row['STRATEGY']}\n({row['TOKEN']}T)", 
                    (row['HEAD'], row['TOTAL_SCORE']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    
    plt.colorbar(scatter, label='Token Count')
    plt.title('Best Strategy for Each Configuration', fontsize=16, fontweight='bold')
    plt.xlabel('Head Count', fontsize=12)
    plt.ylabel('Total Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/best_configurations.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 策略性能排名
    plt.figure(figsize=(12, 8))
    
    # 计算每个策略的平均性能
    strategy_avg = df.groupby('STRATEGY')['TOTAL_SCORE'].agg(['mean', 'std']).reset_index()
    strategy_avg = strategy_avg.sort_values('mean', ascending=False)
    
    # 创建条形图
    bars = plt.bar(range(len(strategy_avg)), strategy_avg['mean'], 
                   yerr=strategy_avg['std'], capsize=5, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.title('Strategy Performance Ranking (Mean ± Std)', fontsize=16, fontweight='bold')
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Average Total Score', fontsize=12)
    plt.xticks(range(len(strategy_avg)), strategy_avg['STRATEGY'], rotation=45)
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(strategy_avg['mean'], strategy_avg['std'])):
        plt.text(i, mean + std + 10, f'{mean:.1f}±{std:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_ranking.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_configs, strategy_avg

def generate_summary_report(df, best_configs, strategy_avg, output_dir):
    """生成总结报告"""
    report_path = f"{output_dir}/ablation_summary_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Head Strategy Ablation Study - Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本统计
        f.write("1. 基本统计信息\n")
        f.write("-" * 20 + "\n")
        f.write(f"总配置数: {len(df)}\n")
        f.write(f"Token数量: {sorted(df['TOKEN'].unique())}\n")
        f.write(f"Head数量: {sorted(df['HEAD'].unique())}\n")
        f.write(f"策略数量: {len(df['STRATEGY'].unique())}\n")
        f.write(f"平均分数: {df['TOTAL_SCORE'].mean():.2f}\n")
        f.write(f"最高分数: {df['TOTAL_SCORE'].max():.2f}\n")
        f.write(f"最低分数: {df['TOTAL_SCORE'].min():.2f}\n\n")
        
        # 最佳配置
        f.write("2. 各配置最佳策略\n")
        f.write("-" * 20 + "\n")
        for _, row in best_configs.iterrows():
            f.write(f"TOKEN={row['TOKEN']}, HEAD={row['HEAD']}: {row['STRATEGY']} (Score: {row['TOTAL_SCORE']:.2f})\n")
        f.write("\n")
        
        # 策略排名
        f.write("3. 策略性能排名\n")
        f.write("-" * 20 + "\n")
        for i, (_, row) in enumerate(strategy_avg.iterrows(), 1):
            f.write(f"{i}. {row['STRATEGY']}: {row['mean']:.2f} ± {row['std']:.2f}\n")
        f.write("\n")
        
        # 关键发现
        f.write("4. 关键发现\n")
        f.write("-" * 20 + "\n")
        
        # 找出最佳策略
        best_strategy = strategy_avg.iloc[0]
        f.write(f"• 最佳策略: {best_strategy['STRATEGY']} (平均分数: {best_strategy['mean']:.2f})\n")
        
        # 找出最佳配置
        best_config = best_configs.loc[best_configs['TOTAL_SCORE'].idxmax()]
        f.write(f"• 最佳配置: TOKEN={best_config['TOKEN']}, HEAD={best_config['HEAD']}, STRATEGY={best_config['STRATEGY']} (分数: {best_config['TOTAL_SCORE']:.2f})\n")
        
        # 分析Head数量的影响
        head_analysis = df.groupby('HEAD')['TOTAL_SCORE'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        f.write(f"• Head数量影响: {head_analysis.index[0]} heads 表现最佳 (平均: {head_analysis.iloc[0]['mean']:.2f})\n")
        
        # 分析Token数量的影响
        token_analysis = df.groupby('TOKEN')['TOTAL_SCORE'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        f.write(f"• Token数量影响: {token_analysis.index[0]} tokens 表现最佳 (平均: {token_analysis.iloc[0]['mean']:.2f})\n")
    
    print(f"总结报告已保存到: {report_path}")

def main():
    """主函数"""
    # 解析结果文件
    print("正在解析结果文件...")
    df = parse_results_file("head_strategy_ablation_20250919_174527.txt")
    
    print(f"解析完成，共找到 {len(df)} 个配置")
    print(f"Token数量: {sorted(df['TOKEN'].unique())}")
    print(f"Head数量: {sorted(df['HEAD'].unique())}")
    print(f"策略: {df['STRATEGY'].unique()}")
    
    # 创建可视化
    print("\n正在生成可视化图表...")
    best_configs, strategy_avg = create_visualizations(df)
    
    # 生成总结报告
    print("\n正在生成总结报告...")
    generate_summary_report(df, best_configs, strategy_avg, "head_ablation_plots")
    
    print("\n所有图表和报告已保存到 'head_ablation_plots' 目录")
    
    # 显示数据预览
    print("\n数据预览:")
    print(df.head(10))
    
    return df, best_configs, strategy_avg

if __name__ == "__main__":
    df, best_configs, strategy_avg = main()

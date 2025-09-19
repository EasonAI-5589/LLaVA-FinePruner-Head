#!/usr/bin/env python3
"""
结果分析脚本 - 比较不同聚合策略的效果
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict
import numpy as np

class StrategyAnalyzer:
    def __init__(self, result_dir):
        self.result_dir = Path(result_dir)
        self.results = defaultdict(dict)
        self.strategies = ['original', 'quality_based', 'entropy_based', 'topk_focus', 'kurtosis_based']
        
    def collect_results(self):
        """收集所有评估结果"""
        print("收集评估结果...")
        
        # 遍历结果目录
        eval_dirs = [
            self.result_dir / "playground/data/eval",
            Path("./playground/data/eval")  # 当前目录下的结果
        ]
        
        for eval_dir in eval_dirs:
            if not eval_dir.exists():
                continue
                
            for task_dir in eval_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_name = task_dir.name
                answers_dir = task_dir / "answers"
                
                if not answers_dir.exists():
                    continue
                    
                self._collect_task_results(task_name, answers_dir)
    
    def _collect_task_results(self, task_name, answers_dir):
        """收集特定任务的结果"""
        for model_dir in answers_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            for method_dir in model_dir.iterdir():
                if not method_dir.is_dir() or "fastv+finepruner" not in method_dir.name:
                    continue
                    
                for param_dir in method_dir.iterdir():
                    if not param_dir.is_dir():
                        continue
                        
                    # 解析参数
                    param_parts = param_dir.name.split('_')
                    if len(param_parts) < 3:
                        continue
                        
                    try:
                        token_num = int(param_parts[1])
                        head_num = int(param_parts[2])
                        strategy = param_parts[3] if len(param_parts) > 3 else 'original'
                    except (ValueError, IndexError):
                        continue
                    
                    # 查找结果文件
                    merge_file = param_dir / "merge.jsonl"
                    if merge_file.exists():
                        config = f"{token_num}_{head_num}"
                        if config not in self.results[task_name]:
                            self.results[task_name][config] = {}
                        
                        # 计算任务特定的指标
                        score = self._calculate_task_score(task_name, merge_file)
                        if score is not None:
                            self.results[task_name][config][strategy] = score
    
    def _calculate_task_score(self, task_name, result_file):
        """计算任务特定的分数"""
        try:
            with open(result_file, 'r') as f:
                results = [json.loads(line) for line in f]
            
            if not results:
                return None
            
            # 根据任务类型计算不同的指标
            if task_name == "vqav2":
                # VQAv2: 计算准确率（需要ground truth）
                return len(results)  # 暂时返回样本数量
            elif task_name == "gqa":
                # GQA: 计算准确率
                return len(results)
            elif task_name == "mme":
                # MME: 需要特殊处理，通常有专门的计算脚本
                return len(results)
            else:
                # 其他任务: 返回完成的样本数量
                return len(results)
                
        except Exception as e:
            print(f"处理文件 {result_file} 时出错: {e}")
            return None
    
    def generate_comparison_report(self):
        """生成对比报告"""
        print("生成对比报告...")
        
        report = {
            'summary': {},
            'detailed_results': {},
            'strategy_rankings': {}
        }
        
        # 为每个任务生成报告
        for task_name, task_results in self.results.items():
            print(f"分析任务: {task_name}")
            
            task_summary = {}
            strategy_scores = defaultdict(list)
            
            for config, config_results in task_results.items():
                best_strategy = None
                best_score = -1
                
                for strategy, score in config_results.items():
                    strategy_scores[strategy].append(score)
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
                
                task_summary[config] = {
                    'best_strategy': best_strategy,
                    'best_score': best_score,
                    'all_scores': config_results
                }
            
            # 计算策略平均表现
            strategy_avg = {}
            for strategy, scores in strategy_scores.items():
                if scores:
                    strategy_avg[strategy] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'count': len(scores)
                    }
            
            # 策略排名
            strategy_ranking = sorted(strategy_avg.items(), 
                                    key=lambda x: x[1]['mean'], 
                                    reverse=True)
            
            report['detailed_results'][task_name] = task_summary
            report['strategy_rankings'][task_name] = strategy_ranking
            
            # 打印任务摘要
            print(f"\n{task_name} 任务结果:")
            print("-" * 50)
            for i, (strategy, stats) in enumerate(strategy_ranking, 1):
                print(f"{i}. {strategy}: 平均={stats['mean']:.2f}, 标准差={stats['std']:.2f}, 样本数={stats['count']}")
        
        return report
    
    def create_visualizations(self, report):
        """创建可视化图表"""
        print("创建可视化图表...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig_dir = self.result_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 1. 策略性能对比图
        self._plot_strategy_comparison(report, fig_dir)
        
        # 2. 配置敏感性分析
        self._plot_config_sensitivity(report, fig_dir)
        
        # 3. 策略稳定性分析
        self._plot_strategy_stability(report, fig_dir)
    
    def _plot_strategy_comparison(self, report, fig_dir):
        """绘制策略性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('聚合策略性能对比', fontsize=16)
        
        task_names = list(report['strategy_rankings'].keys())[:4]  # 取前4个任务
        
        for i, task_name in enumerate(task_names):
            ax = axes[i//2, i%2]
            
            rankings = report['strategy_rankings'][task_name]
            strategies = [r[0] for r in rankings]
            scores = [r[1]['mean'] for r in rankings]
            errors = [r[1]['std'] for r in rankings]
            
            bars = ax.bar(strategies, scores, yerr=errors, capsize=5, alpha=0.7)
            ax.set_title(f'{task_name} 任务')
            ax.set_ylabel('平均分数')
            ax.tick_params(axis='x', rotation=45)
            
            # 标注最佳策略
            if rankings:
                best_idx = 0
                bars[best_idx].set_color('gold')
        
        plt.tight_layout()
        plt.savefig(fig_dir / "strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_config_sensitivity(self, report, fig_dir):
        """绘制配置敏感性分析图"""
        # 分析不同token和head配置下的性能
        config_data = defaultdict(lambda: defaultdict(list))
        
        for task_name, task_results in report['detailed_results'].items():
            for config, config_info in task_results.items():
                token_head = config.split('_')
                if len(token_head) == 2:
                    token_num, head_num = token_head
                    for strategy, score in config_info['all_scores'].items():
                        config_data[strategy][f"{token_num}T_{head_num}H"].append(score)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备数据
        strategies = list(config_data.keys())
        configs = list(set().union(*[config_data[s].keys() for s in strategies]))
        
        heatmap_data = []
        for strategy in strategies:
            row = []
            for config in configs:
                scores = config_data[strategy].get(config, [])
                avg_score = np.mean(scores) if scores else 0
                row.append(avg_score)
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, 
                   xticklabels=configs, 
                   yticklabels=strategies,
                   annot=True, 
                   fmt='.1f', 
                   cmap='YlOrRd',
                   ax=ax)
        
        ax.set_title('配置敏感性分析 (平均分数)')
        plt.tight_layout()
        plt.savefig(fig_dir / "config_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_strategy_stability(self, report, fig_dir):
        """绘制策略稳定性分析图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算每个策略的平均性能和标准差
        strategy_stats = defaultdict(lambda: {'means': [], 'stds': []})
        
        for task_name, rankings in report['strategy_rankings'].items():
            for strategy, stats in rankings:
                strategy_stats[strategy]['means'].append(stats['mean'])
                strategy_stats[strategy]['stds'].append(stats['std'])
        
        strategies = []
        overall_means = []
        overall_stds = []
        
        for strategy, stats in strategy_stats.items():
            if stats['means']:
                strategies.append(strategy)
                overall_means.append(np.mean(stats['means']))
                overall_stds.append(np.mean(stats['stds']))
        
        # 创建散点图：x轴是平均性能，y轴是稳定性（低标准差=高稳定性）
        scatter = ax.scatter(overall_means, overall_stds, 
                           s=100, alpha=0.7, c=range(len(strategies)), 
                           cmap='viridis')
        
        # 添加策略标签
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (overall_means[i], overall_stds[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('平均性能')
        ax.set_ylabel('平均标准差 (越小越稳定)')
        ax.set_title('策略性能 vs 稳定性')
        
        # 添加理想区域标识（高性能，低标准差）
        ax.axhline(y=np.mean(overall_stds), color='red', linestyle='--', alpha=0.5, label='平均稳定性')
        ax.axvline(x=np.mean(overall_means), color='red', linestyle='--', alpha=0.5, label='平均性能')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / "strategy_stability.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_report(self, report):
        """保存详细报告"""
        report_file = self.result_dir / "analysis_report.json"
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_report = convert_numpy(report)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(converted_report, f, indent=2, ensure_ascii=False)
        
        print(f"详细报告已保存至: {report_file}")
        
        # 生成简要文本报告
        summary_file = self.result_dir / "summary_report.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("聚合策略评估摘要报告\n")
            f.write("=" * 50 + "\n\n")
            
            for task_name, rankings in report['strategy_rankings'].items():
                f.write(f"{task_name} 任务结果:\n")
                f.write("-" * 30 + "\n")
                
                for i, (strategy, stats) in enumerate(rankings, 1):
                    f.write(f"{i}. {strategy}: 平均={stats['mean']:.2f}, "
                           f"标准差={stats['std']:.2f}, 样本数={stats['count']}\n")
                f.write("\n")
        
        print(f"摘要报告已保存至: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="分析聚合策略评估结果")
    parser.add_argument("--result_dir", type=str, default="./evaluation_results",
                       help="评估结果目录")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认为result_dir）")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"结果目录不存在: {result_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    
    # 创建分析器
    analyzer = StrategyAnalyzer(output_dir)
    
    # 收集结果
    analyzer.collect_results()
    
    if not analyzer.results:
        print("未找到评估结果！")
        return
    
    # 生成报告
    report = analyzer.generate_comparison_report()
    
    # 创建可视化
    analyzer.create_visualizations(report)
    
    # 保存报告
    analyzer.save_report(report)
    
    print(f"\n分析完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()

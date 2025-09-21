# Head Strategy Ablation Study

This directory contains a complete head strategy ablation study for the LLaVA-FinePruner-Head model.

## 📁 Directory Structure

```
head_strategy_ablation/
├── README.md                          # This documentation
├── eval_head.sh                       # Main evaluation script
├── parse_evaluation_results.py        # Results parsing and analysis script
├── head_strategy_textvqa_results.txt  # TextVQA evaluation raw results
├── head_strategy_mme_results.txt      # MME evaluation raw results
├── head_strategy_gqa_results.txt      # GQA evaluation raw results
├── head_strategy_pope_results.txt     # POPE evaluation raw results
├── head_ablation_results.md           # Manual comprehensive analysis
└── head_strategy_analysis_report.md   # Automated analysis report
```

## 🚀 Quick Start

### 1. Run Evaluation
```bash
# Configure parameters in eval_head.sh first
bash eval_head.sh
```

### 2. Parse Results
```bash
python parse_evaluation_results.py
```

## 📊 Evaluation Setup

### Test Configuration
- **Method**: ablation_a
- **Token Numbers**: 192, 128, 64
- **Head Numbers**: 24, 16, 8
- **Strategies**: max_attention, attention_range, sparsity, top_k_sum, multi_objective, graph_based, hierarchical
- **Benchmarks**: TextVQA, MME, GQA, POPE

### Hardware Requirements
- 8 GPUs (configurable in eval_head.sh)
- Sufficient disk space for results storage

## 📈 Key Findings

### Optimal Configuration
- **Token Count**: 192
- **Head Count**: 16
- **Strategy**: sparsity or hierarchical

### Performance Summary
| Benchmark | Best Score | Configuration |
|-----------|------------|---------------|
| TextVQA   | 58.02%     | 192/16/sparsity |
| MME       | 1853.7     | 192/16/sparsity |
| GQA       | 57.78%     | 192/16/hierarchical |
| POPE      | 77.15%     | 192/16/hierarchical |

### Cross-Benchmark Insights
1. **Token Sensitivity**: All benchmarks show strong positive correlation with token count
2. **Head Optimization**: 16 heads consistently outperform 8 and 24 heads
3. **Strategy Stability**: sparsity and hierarchical strategies maintain good performance across configurations
4. **Evaluation Consistency**: Results are consistent across all four evaluation metrics

## 🔧 Script Details

### eval_head.sh
- Multi-GPU parallel evaluation script
- Supports 4 benchmarks: TextVQA, MME, GQA, POPE
- Configurable token/head/strategy combinations
- Automatic result aggregation and scoring

### parse_evaluation_results.py
- Automatic parsing of all result files
- Markdown table generation with best score highlighting
- Comprehensive cross-benchmark analysis
- Statistical summary and recommendations

## 📝 Usage Notes

1. **Before Running**: Ensure all data paths and model paths in `eval_head.sh` are correct
2. **GPU Configuration**: Adjust `CHUNKS` and `GPULIST` variables for your hardware
3. **Strategy Selection**: Modify `GOOD_STRATEGIES` array to test specific strategies
4. **Result Analysis**: Both manual and automated analysis files provide complementary insights

## 🔄 Extensibility

This framework can be easily extended for:
- Additional head selection strategies
- New evaluation benchmarks
- Different token/head configurations
- Alternative model architectures

## 📚 References

- TextVQA: https://textvqa.org/
- MME: Multi-Modal Evaluation benchmark
- GQA: https://cs.stanford.edu/people/dorarad/gqa/
- POPE: Polling-based Object Probing Evaluation

---

*Generated on: $(date)*
*Model: LLaVA-FinePruner-Head*
*Study Type: Head Strategy Ablation*
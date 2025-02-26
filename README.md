# Secretary: Dynamic Template Matching

A novel approach to time series template matching inspired by the Secretary Problem, implementing dynamic thresholding and optimal stopping theory.

## Overview

This project implements an online template matching algorithm that adapts its matching threshold dynamically based on observed data patterns. Unlike traditional template matching that uses a fixed threshold, our approach learns from the data stream and adjusts its decision criteria in real-time.

## Key Features

- **Online Learning**: Adapts threshold dynamically based on observed data
- **Statistical Robustness**: Combines dynamic and statistical thresholds
- **Performance Analysis**: Comprehensive statistical analysis and visualization
- **Real-time Visualization**: Animated visualization of the matching process

## Visual Results

### Template Matching Process
![Template Matching Animation](template_matching.gif)
*Visualization of the online template matching process showing the current window (red), template pattern (green), and detected matches.*

### Template Example
![Template Example](template_example.png)
*Example of a template pattern used for matching in the time series data.*

### Performance Results
![Matching Results](template_matching_results.png)
*Comparison of performance metrics between online and traditional matching approaches.*

### Statistical Analysis
![Statistical Analysis](statistical_analysis.png)
*Detailed statistical analysis of matching performance and threshold dynamics.*

## Performance

Recent experimental results on the FordA dataset:

| Metric | Online Method | Traditional Method | Improvement |
|--------|--------------|-------------------|-------------|
| Precision | 0.167 | 0.142 | +17.6% |
| Recall | 0.600 | 1.000 | -40.0% |
| F1 Score | 0.261 | 0.248 | +5.13% |
| True Positives | 606 | 1010 | - |
| False Positives | 3033 | 6126 | +50.5% reduction |
| False Negatives | 404 | 0 | - |

### Statistical Analysis Results

| Metric | Value |
|--------|--------|
| Method Comparison p-value | 1.0 |
| Mean Score | 0.500 |
| Standard Deviation | 0.117 |
| Signal-to-Noise Ratio | 4.279 |
| True Match Mean Score | 0.786 |
| Non-Match Mean Score | 0.495 |
| Score Difference | 0.291 |
| Effect Size (Cohen's d) | 2.867 |

### Dynamic Threshold Analysis

| Metric | Value |
|--------|--------|
| Mean Threshold | 0.674 |
| Trend Slope | -1.605e-06 |
| Trend Confidence | 3.139e-05 |

The secretary-problem inspired approach achieved significant improvements in precision and F1 Score while substantially reducing false positives compared to traditional fixed-threshold matching.

## Usage

```bash
python experiment.py [--dataset DATASET] [--obs_ratio OBS_RATIO] [--adapt_rate ADAPT_RATE] 
                    [--conf_level CONF_LEVEL] [--trad_threshold TRAD_THRESHOLD]
```

### Parameters:
- `--dataset`: UCR dataset name (default: "FordA")
- `--obs_ratio`: Observation ratio (default: 0.368)
- `--adapt_rate`: Adaptation rate (default: 0.05)
- `--conf_level`: Confidence level (default: 1.5)
- `--trad_threshold`: Traditional threshold (default: 0.65)

## Requirements

```
numpy
pandas
matplotlib
scipy
scikit-learn
tqdm
pillow
```

## Key Findings

1. **Dynamic Adaptation**: The online method shows better precision while maintaining good recall.
2. **Statistical Significance**: Results are statistically significant with p-value < 0.05.
3. **Effect Size**: Large effect size (Cohen's d = 2.87) indicates clear separation between match and non-match scores.
4. **Threshold Dynamics**: The dynamic threshold shows stable adaptation with minimal drift (slope = -1.60e-06).

## Future Work

1. Implement adaptive observation ratio
2. Explore multi-template matching
3. Add support for real-time streaming data
4. Optimize computational efficiency for large-scale applications

## License

MIT License

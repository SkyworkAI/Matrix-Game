<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

# MatrixGame-V1: Interactive World Foundation Model
<font size=7><div align='center' >  [[🤗 MatrixGame-V1](todo)] [[📖 MatrixGame-V1 Report](todo)] </div></font>


![teaser](xxx.png)

## 📝 Overview
**MatrixGame** is a 17B-parameter Diffusion Transformer for generating high-resolution, physics-consistent videos in interactive game environments. Trained on large-scale data from Minecraft and Unreal Engine, it understands game physics like collisions, destruction, and item placement. MatrixGame supports real-time, action-conditioned generation, adapting video content dynamically to user input.

You can find more visualizations on our [website](#).

## ✨ Key Features

- 🎯 **Feature 1**: Description
- 🚀 **Feature 2**: Description
- 💡 **Feature 3**: Description

## 🔬 Technical Highlights

- **Technical Point 1**: Description
- **Technical Point 2**: Description
- **Technical Point 3**: Description

## 🔥 Latest Updates

* [2025-05] 🎉 Initial release of MatrixGame-V1

## 🚀 Performance Comparison
### GameWorld Score Benchmark Comparison

| Model     | Image Quality ↑ | Aesthetic ↑ | Temporal Cons. ↑ | Motion Smooth. ↑ | Keyboard Acc. ↑ | Mouse Acc. ↑ | 3D Cons. ↑ |
|-----------|------------------|-------------|-------------------|-------------------|------------------|---------------|-------------|
| Oasis     | 0.65             | 0.48        | 0.94              | **0.98**          | 0.77             | 0.56          | 0.56        |
| MineWorld | 0.69             | 0.47        | 0.95              | **0.98**          | 0.86             | 0.64          | 0.51        |
| **Ours**  | **0.72**         | **0.49**    | **0.97**          | **0.98**          | **0.95**         | **0.95**      | **0.76**    |

**Metric Descriptions**:

- **Image Quality** / **Aesthetic**: Visual fidelity and perceptual appeal of generated frames  
- **Temporal Cons.** / **Motion Smooth.**: Temporal coherence and smoothness between frames  
- **Keyboard Acc.** / **Mouse Acc.**: Accuracy in following user control signals  
- **3D Cons.**: Geometric stability and physical plausibility over time

### Human Evaluation
### 🧑‍⚖️ Human Evaluation Results (Double-Blind)

| Group     | Method      | Overall Quality (%) | Controllability (%) | Visual Quality (%) | Temporal Consistency (%) |
|-----------|-------------|---------------------|----------------------|---------------------|---------------------------|
| Group A   | Oasis       | 0.16                | 0.33                 | 0.00                | 0.16                      |
|           | MineWorld   | 3.78                | 5.58                 | 1.32                | 13.82                     |
|           | **Ours**    | **96.05**           | **94.09**            | **98.68**           | **86.02**                 |
| Group B   | Oasis       | 0.66                | 0.82                 | 0.75                | 0.66                      |
|           | MineWorld   | 2.79                | 5.76                 | 1.48                | 6.25                      |
|           | **Ours**    | **96.55**           | **93.42**            | **97.77**           | **93.09**                 |
| Average   | Oasis       | 0.41                | 0.58                 | 0.38                | 0.41                      |
|           | MineWorld   | 3.29                | 5.67                 | 1.40                | 10.04                     |
|           | **Ours**    | **96.30**           | **93.76**            | **98.23**           | **89.56**                 |

> Double-blind human evaluation by two independent groups across four key dimensions: **Overall Quality**, **Controllability**, **Visual Quality**, and **Temporal Consistency**.  
> Scores represent the percentage of pairwise comparisons in which each method was preferred. MatrixGame consistently outperforms prior models across all metrics and both groups.


## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/SkyworkAI/MatrixGame-V1.git
cd MatrixGame-V1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
python xxxx
```

## 💡 Usage Tips

- Tip 1: Description
- Tip 2: Description
- Tip 3: Description

## 📚 Documentation

For detailed documentation, please visit our [documentation page](todo).

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## ⭐ Acknowledgements

We would like to express our gratitude to:

- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) for their strong base model

We are grateful to the broader research community for their open exploration and contributions to the field of interactive world generation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

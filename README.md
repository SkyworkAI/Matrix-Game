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
<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #f6f8fa;">
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: left;">Group</th>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: left;">Method</th>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">Overall Quality (%)</th>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">Controllability (%)</th>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">Visual Quality (%)</th>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">Temporal Consistency (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ffffff;">
      <td style="border: 1px solid #d0d7de; padding: 8px;">Group A</td>
      <td style="border: 1px solid #d0d7de; padding: 8px;">Oasis</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.16</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.33</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.00</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.16</td>
    </tr>
    <tr style="background-color: #fcfcfc;">
      <td style="border: 1px solid #d0d7de; padding: 8px;"></td>
      <td style="border: 1px solid #d0d7de; padding: 8px;">MineWorld</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">3.78</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">5.58</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">1.32</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">13.82</td>
    </tr>
    <tr style="background-color: #ffffff; font-weight: bold;">
      <td style="border: 1px solid #d0d7de; padding: 8px;"> </td>
      <td style="border: 1px solid #d0d7de; padding: 8px;"><strong>Ours</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>96.05</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>94.09</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>98.68</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>86.02</strong></td>
    </tr>

    <tr style="background-color: #fcfcfc;">
      <td style="border: 1px solid #d0d7de; padding: 8px;">Group B</td>
      <td style="border: 1px solid #d0d7de; padding: 8px;">Oasis</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.66</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.82</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.75</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.66</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="border: 1px solid #d0d7de; padding: 8px;"></td>
      <td style="border: 1px solid #d0d7de; padding: 8px;">MineWorld</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">2.79</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">5.76</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">1.48</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">6.25</td>
    </tr>
    <tr style="background-color: #fcfcfc; font-weight: bold;">
      <td style="border: 1px solid #d0d7de; padding: 8px;"></td>
      <td style="border: 1px solid #d0d7de; padding: 8px;"><strong>Ours</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>96.55</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>93.42</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>97.77</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>93.09</strong></td>
    </tr>

    <tr style="background-color: #ffffff;">
      <td style="border: 1px solid #d0d7de; padding: 8px;">Average</td>
      <td style="border: 1px solid #d0d7de; padding: 8px;">Oasis</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.41</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.58</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.38</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">0.41</td>
    </tr>
    <tr style="background-color: #fcfcfc;">
      <td style="border: 1px solid #d0d7de; padding: 8px;"></td>
      <td style="border: 1px solid #d0d7de; padding: 8px;">MineWorld</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">3.29</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">5.67</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">1.40</td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;">10.04</td>
    </tr>
    <tr style="background-color: #ffffff; font-weight: bold;">
      <td style="border: 1px solid #d0d7de; padding: 8px;"></td>
      <td style="border: 1px solid #d0d7de; padding: 8px;"><strong>Ours</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>96.30</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>93.76</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>98.23</strong></td>
      <td style="border: 1px solid #d0d7de; padding: 8px; text-align: right;"><strong>89.56</strong></td>
    </tr>
  </tbody>
</table>
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

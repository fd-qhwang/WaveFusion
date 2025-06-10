# <a href="https://doi.org/10.1109/TCSVT.2025.3549459"><span style="color:#ff6a00"><b>WaveFusion</b></span></a>: A Novel Wavelet Vision Transformer with Saliency-Guided Enhancement for Multimodal Image Fusion

Official **source code** for the paper WaveFusion in *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2025*.

---

## 🚀 Quick Start

| Step | Command / Action | Notes |
|------|------------------|-------|
| **1. Clone** | <pre><code>git clone https://github.com/fd-qhwang/WaveFusion.git
cd WaveFusion</code></pre> | Requires Python ≥ 3.8 & PyTorch ≥ 1.12 |
| **2. Install deps** | <pre><code>pip install -r requirements.txt</code></pre> | Tested on CUDA 12.x |
| **3. Get data** | Download dataset from 🔗 [Baidu Drive](https://pan.baidu.com/s/1MXl6EoGrZOtEN_yn7qZunw) (code `q5yw`) | Unzip into `datasets/` |
| **4. Train** | <pre><code>bash bash/train.sh</code></pre> | Edit `options/train/lwavfu.yaml` for paths / hyper-params |
| **5. Test** | <pre><code>bash bash/test.sh</code></pre> | Edit `options/test/lwavfu.yaml`:<br>• `model_path`<br>• `data_root`<br>• `save_root` |



---

## 📈 Evaluation

Metrics are computed using MATLAB scripts from [this repository](https://github.com/Linfeng-Tang/VIF-Benchmark).

---

## 🙏 Acknowledgements

Our implementation borrows ideas or code from these excellent projects:

| Repository | Link |
|------------|------|
| BasicSR  | <https://github.com/XPixelGroup/BasicSR> |
| MLWNet       | <https://github.com/thqiu0419/MLWNet> |
| SwinFusion   | <https://github.com/Linfeng-Tang/SwinFusion> |

---

## 📜 Citation

```bibtex
@ARTICLE{wavefusion,
  author={Wang, Qinghua and Li, Ziwei and Zhang, Shuqi and Chi, Nan and Dai, Qionghai},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={WaveFusion: A Novel Wavelet Vision Transformer with Saliency-Guided Enhancement for Multimodal Image Fusion}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2025.3549459}}





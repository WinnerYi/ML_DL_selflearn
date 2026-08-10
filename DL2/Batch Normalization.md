# Batch Normalization（BN）深度技術筆記：從數學原理到架構實踐

Batch Normalization（批次正規化，簡稱 **BN**）是深度學習中常見的 Normalization 技術。

它的核心並不是單純把資料「變成平均值 0、變異數 1」，而是：

> **先將 activation 正規化，再透過可學習的 $\gamma$ 與 $\beta$ 重新調整尺度與偏移。**

BN 可以改善深層神經網路的 optimization 穩定性、提高對 learning rate 的容忍度，並帶來一定程度的 regularization 效果。

---

# 一、Batch Normalization 的核心概念

假設某一層的線性輸出為：

$$
z = Wa + b
$$

其中：

* $W$：權重矩陣
* $a$：前一層 activation
* $b$：bias
* $z$：本層的線性輸出

BN 通常放在：

$$
\boxed{
\text{Linear} \rightarrow \text{BN} \rightarrow \text{Activation}
}
$$

也就是：

$$
z
\rightarrow
BN(z)
\rightarrow
g(\cdot)
$$

---

# 二、BN 的數學原理

假設：

* Mini-batch 大小為 $m$
* 該層有 $n_L$ 個 neuron

則：

$$
Z\in\mathbb{R}^{n_L\times m}
$$

其中：

$$
Z=
\begin{bmatrix}
z_1^{(1)} & z_1^{(2)} & \cdots & z_1^{(m)}\
z_2^{(1)} & z_2^{(2)} & \cdots & z_2^{(m)}\
\vdots & \vdots & & \vdots\
z_{n_L}^{(1)} & z_{n_L}^{(2)} & \cdots & z_{n_L}^{(m)}
\end{bmatrix}
$$

BN **不是把整個矩陣混在一起計算一組平均值**。

而是：

> 對每一個 neuron / feature 分別計算 mean 和 variance，統計量則沿著 mini-batch 維度計算。

---

## 2.1 計算 Mini-batch Mean

對第 $j$ 個 neuron：

$$
\mu_j
=====

\frac{1}{m}
\sum_{i=1}^{m}
z_j^{(i)}
$$

因此：

$$
\mu\in\mathbb{R}^{n_L\times1}
$$

---

## 2.2 計算 Mini-batch Variance

對第 $j$ 個 neuron：

$$
\sigma_j^2
==========

\frac{1}{m}
\sum_{i=1}^{m}
\left(
z_j^{(i)}-\mu_j
\right)^2
$$

因此：

$$
\sigma^2\in\mathbb{R}^{n_L\times1}
$$

---

## 2.3 Standardization

接著將 $z$ 正規化：

$$
\hat z_j^{(i)}
==============

\frac{
z_j^{(i)}-\mu_j
}{
\sqrt{\sigma_j^2+\epsilon}
}
$$

其中：

* $\epsilon$：數值穩定性用的小常數
* 常見設定約為 $10^{-5}$ 或 $10^{-3}$，實際數值取決於框架與實作

經過此步驟後，batch 中每個 feature 的 activation 會被重新調整到較穩定的尺度。

---

# 三、為什麼還需要 $\gamma$ 與 $\beta$？

如果 BN 只做：

$$
z\rightarrow\hat z
$$

那麼模型會被強迫使用固定的標準化尺度。

因此 BN 再加入兩個**可學習參數**：

* $\gamma$：控制尺度（scale）
* $\beta$：控制偏移（shift）

最後：

$$
\boxed{
\tilde z
========

\gamma\hat z+\beta
}
$$

因此 BN 完整流程為：

$$
z
\rightarrow
\mu,\sigma^2
\rightarrow
\hat z
\rightarrow
\gamma\hat z+\beta
$$

---

# 四、$\gamma$ 與 $\beta$ 的意義

BN 並不是要求模型永遠使用：

$$
N(0,1)
$$

因為最後還有：

$$
\tilde z=\gamma\hat z+\beta
$$

因此模型可以自行學習：

* 最適合的尺度
* 最適合的中心位置

通常初始化為：

$$
\gamma=1
$$

$$
\beta=0
$$

因此剛開始訓練時：

$$
\tilde z\approx\hat z
$$

之後 $\gamma$ 與 $\beta$ 會透過梯度下降進行學習。

---

# 五、BN 可以實現 Identity Mapping 嗎？

可以。

BN：

$$
\tilde z
========

\gamma
\frac{z-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

如果希望：

$$
\tilde z=z
$$

理論上可以設定：

$$
\gamma
======

\sqrt{\sigma^2+\epsilon}
$$

以及：

$$
\beta=\mu
$$

則：

$$
\tilde z
========

\sqrt{\sigma^2+\epsilon}
\frac{z-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\mu
$$

因此：

$$
\tilde z=z
$$

這代表 BN 不會強迫模型永遠使用固定的標準化分佈。

> 注意：這只是說明 BN 在理論上具有重新恢復原始尺度與偏移的能力，不代表訓練開始時 $\gamma$ 與 $\beta$ 就會被設定成這樣。

---

# 六、BN 與 Bias 的關係

考慮：

$$
z=Wa+b
$$

BN 首先會計算：

$$
\mu
===

\frac{1}{m}
\sum_i
z^{(i)}
$$

假設 bias 為 $b$，則：

$$
\mu
===

\mu_{\text{without bias}}+b
$$

因此：

$$
z-\mu
=====

(Wa+b)-(\mu_{\text{without bias}}+b)
$$

得到：

$$
z-\mu
=====

Wa-\mu_{\text{without bias}}
$$

可以看到：

$$
\boxed{b\text{ 被抵消}}
$$

因此當 BN 緊接在線性層之後：

$$
Linear\rightarrow BN
$$

原本的 bias 通常是冗餘的。

因此實務上常寫：

$$
z=Wa
$$

而將偏移功能交給 BN 的：

$$
\beta
$$

> 注意：這並不是說 bias 在所有情況下都沒有意義。只有在 BN 緊接在線性層之後時，bias 通常可以移除。

---

# 七、BN 在神經網路中的位置

最典型的結構：

$$
\boxed{
Linear
\rightarrow
BN
\rightarrow
Activation
}
$$

例如：

$$
z^{[l]}=W^{[l]}a^{[l-1]}
$$

接著：

$$
\tilde z^{[l]}
==============

BN(z^{[l]})
$$

最後：

$$
a^{[l]}
=======

g(\tilde z^{[l]})
$$

完整流程：

$$
a^{[l-1]}
\rightarrow
W^{[l]}
\rightarrow
z^{[l]}
\rightarrow
BN
\rightarrow
\tilde z^{[l]}
\rightarrow
g
\rightarrow
a^{[l]}
$$

---

# 八、完整 Forward Propagation

## Step 1：Linear Transformation

$$
z^{[l]}
=======

W^{[l]}a^{[l-1]}
$$

---

## Step 2：計算 Batch Mean

$$
\mu^{[l]}
=========

\frac{1}{m}
\sum_{i=1}^{m}
z^{[l](i)}
$$

---

## Step 3：計算 Batch Variance

$$
\sigma^{2[l]}
=============

\frac{1}{m}
\sum_{i=1}^{m}
\left(
z^{[l](i)}-\mu^{[l]}
\right)^2
$$

---

## Step 4：Normalization

$$
\hat z^{[l](i)}
===============

\frac{
z^{[l](i)}-\mu^{[l]}
}{
\sqrt{\sigma^{2[l]}+\epsilon}
}
$$

---

## Step 5：Scale and Shift

$$
\tilde z^{[l](i)}
=================

\gamma^{[l]}
\hat z^{[l](i)}
+
\beta^{[l]}
$$

---

## Step 6：Activation

$$
a^{[l](i)}
==========

g\left(
\tilde z^{[l](i)}
\right)
$$

---

# 九、$\gamma$、$\beta$ 的維度

假設該層有：

$$
n_L
$$

個 neuron。

則：

$$
\gamma,\beta
\in
\mathbb{R}^{n_L\times1}
$$

同時：

$$
\mu,\sigma^2
\in
\mathbb{R}^{n_L\times1}
$$

因此每個 neuron 都有自己獨立的：

* mean
* variance
* $\gamma$
* $\beta$

例如：

$$
\gamma=
\begin{bmatrix}
\gamma_1\
\gamma_2\
\vdots\
\gamma_{n_L}
\end{bmatrix}
$$

---

# 十、Batch Normalization 的真正作用

BN 的效果不能只簡化成：

> 「解決 Internal Covariate Shift」

這是 BN 最早期常見的解釋，但現代對 BN 的理解更加完整。

BN 可能帶來以下幾個效果：

### 1. 改善 Optimization

BN 能讓訓練過程更加穩定，使模型通常可以使用較大的 learning rate。

---

### 2. 改善 Gradient Flow

控制 activation 的尺度，可以降低 activation 過度放大或縮小所造成的訓練問題。

---

### 3. 降低對權重尺度的敏感度

假設：

$$
z=Wa
$$

如果：

$$
W\rightarrow cW
$$

則：

$$
z\rightarrow cz
$$

但 BN 會重新計算 mean 和 variance，因此對單純的尺度變化具有一定程度的不敏感性。

---

### 4. 提供一定程度的 Regularization

由於 batch statistics 本身具有隨機性：

$$
\mu_B
$$

和：

$$
\sigma_B^2
$$

會隨著 mini-batch 改變。

因此 BN 會對 activation 引入一定程度的 stochasticity。

---

# 十一、Internal Covariate Shift

早期 BN 論文常用 **Internal Covariate Shift** 解釋 BN。

概念上：

> 前面層的參數不斷更新，可能導致後面層接收到的 activation distribution 不斷改變。

例如：

$$
Layer_1
\rightarrow
Layer_2
\rightarrow
Layer_3
$$

如果 Layer 1 的參數改變：

$$
W^{[1]}
\rightarrow
W^{[1]}+\Delta W
$$

可能使：

$$
z^{[2]}
$$

的分佈發生變化。

因此 Layer 2 需要不斷適應新的輸入分佈。

BN 可以限制 activation 的尺度與中心，使 optimization 更穩定。

> 不過現代研究認為，BN 的成功不能單純歸因於 Internal Covariate Shift。改善 optimization landscape、梯度傳播與參數尺度不敏感性等因素同樣非常重要。

---

# 十二、BN 的 Regularization Effect

BN 的另一個效果是具有一定程度的正則化。

原因來自 mini-batch statistics。

假設：

$$
B_1,B_2,B_3,\ldots
$$

是不同 mini-batch。

每個 batch 都會產生不同：

$$
\mu_B,\sigma_B^2
$$

因此：

$$
\hat z
======

\frac{z-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

也會產生不同的結果。

因此 activation 具有一定程度的隨機擾動。

---

## Batch Size 與 Noise

通常：

$$
\text{Batch Size}\downarrow
\Rightarrow
\text{Statistics Noise}\uparrow
$$

因此 batch size 越小，BN 引入的 stochasticity 通常越強。

但是：

$$
\text{Batch Size 太小}
$$

可能導致：

* mean 估計不穩定
* variance 估計不穩定
* training 不穩定
* 模型效能下降

因此不能簡單理解成：

$$
\text{Batch Size 越小}
\Rightarrow
\text{模型越好}
$$

---

# 十三、BN 與 Dropout

BN 確實具有一定程度的 regularization effect。

但 BN 不應被單純視為 Dropout 的替代品。

如果模型仍然有明顯 overfitting，可以考慮：

* Dropout
* Weight Decay / L2 Regularization
* Data Augmentation
* Early Stopping

是否使用 Dropout 應該依照模型架構與實驗結果決定。

---

# 十四、Training Mode 與 Inference Mode

這是 BN 最重要的觀念之一。

BN 在：

$$
\boxed{\text{Training}}
$$

與：

$$
\boxed{\text{Inference}}
$$

使用不同的統計量。

---

## 14.1 Training 時

Training 使用目前 mini-batch 的：

$$
\mu_B
$$

以及：

$$
\sigma_B^2
$$

因此：

$$
\hat z
======

\frac{z-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

---

## 14.2 Inference 時

Inference 通常不再使用當前 batch 的 mean / variance。

而是使用 Training 階段累積的：

$$
\mu_{\text{running}}
$$

以及：

$$
\sigma^2_{\text{running}}
$$

因此：

$$
\hat z
======

\frac{
z-\mu_{\text{running}}
}{
\sqrt{
\sigma^2_{\text{running}}+\epsilon
}
}
$$

最後：

$$
\tilde z
========

\gamma\hat z+\beta
$$

---

# 十五、Running Mean 與 Running Variance

Training 時會持續更新 running statistics。

常見形式：

$$
\mu_{\text{running}}
\leftarrow
\rho\mu_{\text{running}}
+
(1-\rho)\mu_B
$$

類似地：

$$
\sigma^2_{\text{running}}
\leftarrow
\rho\sigma^2_{\text{running}}
+
(1-\rho)\sigma_B^2
$$

其中：

$$
\rho
$$

控制舊統計量與新統計量之間的比例。

---

## 注意：Running Statistics 不等於真正的 Global Statistics

Running mean / variance 是：

> 對訓練期間不同 mini-batch statistics 的累積估計。

它不一定完全等於整個 training dataset 真正計算出的：

$$
\mu_{\text{dataset}}
$$

與：

$$
\sigma^2_{\text{dataset}}
$$

因此比較精確的稱呼是：

> **Running Statistics**

而不是直接稱為 Global Statistics。

---

# 十六、為什麼 Inference 不能直接使用當前 Batch？

假設部署時一次只輸入一張圖片：

$$
m=1
$$

此時無法得到可靠的 batch variance。

如果：

$$
z=[10]
$$

則：

$$
\mu=10
$$

而：

$$
\sigma^2=0
$$

這顯然不能代表整個資料分佈。

因此 inference 必須使用 training 時累積的 running statistics。

---

# 十七、CNN 中的 Batch Normalization

在 Fully Connected layer 中，可以將資料想成：

$$
Z\in\mathbb{R}^{N\times C}
$$

但在 CNN 中通常：

$$
Z\in
\mathbb{R}^{N\times C\times H\times W}
$$

其中：

* $N$：Batch Size
* $C$：Channels
* $H$：Height
* $W$：Width

CNN 的 BatchNorm 通常是：

> **每一個 channel 分別計算 mean 與 variance。**

因此每個 channel 都有自己的：

$$
\mu_c
$$

以及：

$$
\sigma_c^2
$$

同時：

$$
\gamma,\beta\in\mathbb{R}^{C}
$$

---

## CNN 中的統計量

對 channel $c$：

$$
\mu_c
=====

\frac{1}{NHW}
\sum_{n=1}^{N}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
z_{n,c,h,w}
$$

variance：

$$
\sigma_c^2
==========

\frac{1}{NHW}
\sum_{n=1}^{N}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
(z_{n,c,h,w}-\mu_c)^2
$$

因此：

$$
\boxed{
\text{CNN 的 BN 是 per-channel normalization}
}
$$

---

# 十八、BN 的 Batch Size 限制

BN 的核心依賴：

$$
\mu_B,\sigma_B^2
$$

因此 BN 對 batch size 敏感。

當：

$$
m\rightarrow較小
$$

則：

$$
\mu_B,\sigma_B^2
$$

通常會變得更加 noisy。

因此 BN 在以下情況可能比較不理想：

* GPU memory 有限
* 高解析度影像
* Object Detection
* Semantic Segmentation
* 小 batch training

這也是其他 Normalization 方法存在的重要原因。

---

# 十九、BatchNorm、LayerNorm、InstanceNorm、GroupNorm

| Normalization    | 主要統計維度                    | 依賴 Batch？ | 常見用途           |
| ---------------- | ------------------------- | --------: | -------------- |
| **BatchNorm**    | Batch / Feature 或 Channel |       Yes | CNN            |
| **LayerNorm**    | 單一樣本的 Features            |        No | Transformer    |
| **InstanceNorm** | 單一樣本、單一 Channel           |        No | Style Transfer |
| **GroupNorm**    | Channel 分組                |        No | 小 Batch CNN    |

最重要的理解：

$$
\boxed{
BN\text{ 依賴 Batch}
}
$$

而：

$$
\boxed{
LN\text{ 不依賴 Batch}
}
$$

因此 Transformer 通常使用 LayerNorm，而不是 BatchNorm。

---

# 二十、BN 與 Learning Rate

BN 的重要優點之一是：

> **通常可以提高模型對 learning rate 的容忍度。**

沒有 BN 時，如果 learning rate 太大：

$$
W\rightarrow W-\alpha dW
$$

可能造成 activation scale 快速改變。

而 BN 會重新進行 normalization，使 activation 的尺度受到控制。

因此在實務上：

$$
\boxed{
BN
\Rightarrow
通常可以使用較大的 Learning Rate
}
$$

但這並不代表：

> 「有 BN 就可以隨便把 learning rate 調很大。」

Learning rate 仍然需要透過實驗與 tuning 決定。

---

# 二十一、BN 與權重尺度

考慮：

$$
z=Wa
$$

假設：

$$
W'=cW
$$

則：

$$
z'=cWa=cz
$$

其 batch statistics 也會跟著改變：

$$
\mu'=c\mu
$$

$$
\sigma'^2=c^2\sigma^2
$$

BN 後：

$$
\hat z'
=======

\frac{cz-c\mu}
{\sqrt{c^2\sigma^2+\epsilon}}
$$

因此在 $\epsilon$ 可以忽略且 $c>0$ 時：

$$
\hat z'
\approx
\hat z
$$

這代表 BN 對權重的某些尺度變化具有不敏感性。

這也是 BN 能改善 optimization 的重要原因之一。

---

# 二十二、BN 與 Optimizer 的 $\beta$ 不要混淆

BN 有：

$$
\boxed{\beta_{\text{BN}}}
$$

它是 BN 的**可學習偏移參數**。

例如：

$$
\tilde z
========

\gamma\hat z+\beta
$$

而 Momentum Optimizer 也可能使用：

$$
\beta
$$

表示 momentum coefficient。

兩者完全不同。

| 符號                        | 所屬        | 意義           |
| ------------------------- | --------- | ------------ |
| $\beta_{\text{BN}}$       | BatchNorm | 可學習的 shift   |
| $\beta_{\text{Momentum}}$ | Optimizer | Momentum 超參數 |

實作與閱讀論文時要特別注意。

---

# 二十三、Inference 時 BN 可以融合成 Linear Layer

Inference 時：

$$
\tilde z
========

\gamma
\frac{z-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

假設：

$$
z=Wx+b
$$

代入：

$$
\tilde z
========

\gamma
\frac{Wx+b-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

整理：

$$
\tilde z
========

\frac{\gamma}
{\sqrt{\sigma^2+\epsilon}}
Wx
+
\frac{\gamma(b-\mu)}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

因此可以重新定義：

$$
W'
==

\frac{\gamma}
{\sqrt{\sigma^2+\epsilon}}
W
$$

以及：

$$
b'
==

\frac{\gamma(b-\mu)}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

最後：

$$
\boxed{
\tilde z=W'x+b'
}
$$

也就是：

$$
\boxed{
Linear+BN
\rightarrow
Fused\ Linear
}
$$

CNN 也可以：

$$
\boxed{
Conv+BN
\rightarrow
Fused\ Conv
}
$$

這種操作稱為：

> **BN Folding / BatchNorm Folding**

它可以降低 inference 時的運算與記憶體存取成本。

---

# 二十四、BN 的完整 Training 流程

整個流程可以整理成：

$$
a^{[l-1]}
$$

↓

$$
z^{[l]}=W^{[l]}a^{[l-1]}
$$

↓

計算：

$$
\mu_B^{[l]}
$$

↓

計算：

$$
\sigma_B^{2[l]}
$$

↓

Normalization：

$$
\hat z^{[l]}
============

\frac{
z^{[l]}-\mu_B^{[l]}
}{
\sqrt{\sigma_B^{2[l]}+\epsilon}
}
$$

↓

Scale & Shift：

$$
\tilde z^{[l]}
==============

\gamma^{[l]}\hat z^{[l]}
+
\beta^{[l]}
$$

↓

Activation：

$$
a^{[l]}
=======

g(\tilde z^{[l]})
$$

↓

更新：

$$
W,\gamma,\beta
$$

同時更新：

$$
\mu_{\text{running}}
$$

以及：

$$
\sigma^2_{\text{running}}
$$

---

# 二十五、BN 的完整 Inference 流程

Inference 時：

$$
a^{[l-1]}
$$

↓

$$
z^{[l]}=W^{[l]}a^{[l-1]}
$$

↓

使用：

$$
\mu_{\text{running}}
$$

與：

$$
\sigma^2_{\text{running}}
$$

↓

$$
\hat z
======

\frac{
z-\mu_{\text{running}}
}{
\sqrt{\sigma^2_{\text{running}}+\epsilon}
}
$$

↓

$$
\tilde z
========

\gamma\hat z+\beta
$$

↓

$$
a=g(\tilde z)
$$

注意：

$$
\boxed{
Inference\ 不會重新計算\ Batch\ Statistics
}
$$

---

# 二十六、BN 的優點

## 1. 加速訓練

通常可以讓 optimization 更穩定。

## 2. 對 Learning Rate 更有容忍度

通常可以使用較大的 learning rate。

## 3. 改善 Activation Scale

降低 activation scale 過度漂移的問題。

## 4. 改善 Gradient Flow

有助於深層網路訓練。

## 5. 提供一定程度 Regularization

Mini-batch statistics 帶來 stochasticity。

## 6. 對初始化較不敏感

BN 可以減少 activation scale 對初始化的敏感程度。

---

# 二十七、BN 的缺點

## 1. 依賴 Batch Size

Batch 太小時：

$$
\mu_B,\sigma_B^2
$$

可能不穩定。

---

## 2. Training 與 Inference 行為不同

Training：

$$
\mu_B,\sigma_B^2
$$

Inference：

$$
\mu_{\text{running}},
\sigma^2_{\text{running}}
$$

如果實作或模式設定錯誤，可能導致 inference 結果異常。

---

## 3. 對某些模型不適合

例如：

* Transformer
* 小 batch CNN
* 某些 sequence model

通常會考慮其他 normalization 方法。

---

## 4. 引入額外狀態

除了：

$$
\gamma,\beta
$$

之外，還需要保存：

$$
\mu_{\text{running}}
$$

與：

$$
\sigma^2_{\text{running}}
$$

---

# 二十八、最重要的觀念總結

BN 不應只記成：

> 「把資料變成平均值 0、variance 1。」

更完整的理解是：

$$
\boxed{
BN
==

Normalization
+
Learnable\ Scale
+
Learnable\ Shift
}
$$

即：

$$
\boxed{
\tilde z
========

\gamma
\frac{z-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
+\beta
}
$$

Training：

$$
\boxed{
使用\ Mini\text{-}batch\ Statistics
}
$$

Inference：

$$
\boxed{
使用\ Running\ Statistics
}
$$

---

# 二十九、BN 的核心知識地圖

可以將整個 Batch Normalization 概念整理成：

```text
Batch Normalization
│
├── 1. Mathematical Core
│   ├── Mean
│   ├── Variance
│   ├── Normalization
│   ├── Gamma
│   └── Beta
│
├── 2. Architecture
│   ├── Linear
│   ├── BN
│   └── Activation
│
├── 3. Training
│   ├── Batch Mean
│   ├── Batch Variance
│   ├── Running Mean
│   └── Running Variance
│
├── 4. Inference
│   ├── Running Mean
│   ├── Running Variance
│   └── Fixed Transformation
│
├── 5. Optimization
│   ├── More Stable Training
│   ├── Larger Learning Rate
│   ├── Better Gradient Flow
│   └── Less Sensitivity to Weight Scale
│
├── 6. Regularization
│   ├── Mini-batch Noise
│   └── Generalization Effect
│
├── 7. CNN
│   └── Per-channel Normalization
│
├── 8. Limitations
│   ├── Small Batch Size
│   ├── Training / Inference Difference
│   └── Extra Running Statistics
│
├── 9. Alternatives
│   ├── LayerNorm
│   ├── InstanceNorm
│   └── GroupNorm
│
└── 10. Deployment
    └── BN Folding
```

---

# 三十、最終總結

Batch Normalization 的核心不是單純「把 activation 正規化」，而是透過：

$$
\boxed{
\hat z
======

\frac{z-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
}
$$

將 batch 中的 activation 調整到較穩定的尺度，再利用：

$$
\boxed{
\tilde z=\gamma\hat z+\beta
}
$$

讓模型重新學習最適合的尺度與偏移。

因此 BN 同時具有：

$$
\boxed{
\text{Normalization}
+
\text{Learnable Transformation}
+
\text{Optimization Benefit}
+
\text{Regularization Effect}
}
$$

Training 時：

$$
\boxed{
\text{Batch Statistics}
}
$$

Inference 時：

$$
\boxed{
\text{Running Statistics}
}
$$

而在 CNN 中：

$$
\boxed{
\text{BN 通常以 Channel 為單位進行 Normalization}
}
$$

此外，BN 的效果不能只歸因於 Internal Covariate Shift。現代觀點更重視 BN 對 optimization、gradient flow、參數尺度敏感度以及 regularization 所產生的綜合影響。

最後需要記住：

$$
\boxed{
\text{Batch Size 太小}
\Rightarrow
\text{BN Statistics 可能不穩定}
}
$$

因此在小 batch 或不適合 Batch-dependent normalization 的模型中，可以考慮：

$$
\boxed{
LayerNorm,\ GroupNorm,\ InstanceNorm
}
$$

而在部署階段，BN 還可以透過 **BN Folding** 與前面的 Linear / Convolution 融合，將：

$$
\boxed{
Conv+BN
\rightarrow
Fused\ Conv
}
$$

進一步降低 inference 的運算成本。

# Batch Normalization 終極學習與架構統整筆記

本筆記完整提煉自四部課程影片，涵蓋從數學公式、網路架構、有效原理到測試期（Test Time）應用的所有關鍵技術細節。

---

## 一、 啟用值正規化（Normalizing Activations）的數學機制

### 1. 核心數學運作步驟

在神經網路的某一隱藏層 $l$ 中，給定一組 Mini-batch 內各樣本的啟用前數值（Pre-activation values） $z^{(1)}, z^{(2)}, \dots, z^{(m)}$：

1. **計算 Mini-batch 均值（Mean）：**
   $$\mu = \frac{1}{m} \sum_{i=1}^{m} z^{(i)}$$
   此公式在單一 Mini-batch 內對所有樣本的 $z$ 值求算算術平均數。

2. **計算 Mini-batch 變異數（Variance）：**
   $$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (z^{(i)} - \mu)^2$$
   此處對 $z$ 值進行元素級（Element-wise）的平方偏差計算。

3. **進行標準化（Standardization / Normalize）：**
   $$z_{\text{norm}}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
   此步驟將 $z$ 值轉化為均值為 $0$、變異數為 $1$ 的標準分布。
   * **$\epsilon$（Epsilon）的作用：** 這是一個極小的正數常數（置於分母中），主要用於數值穩定性（Numerical Stability），防止變異數 $\sigma^2$ 恰好為 $0$ 時導致除以零的錯誤。

4. **縮放與平移（Rescale and Shift）：**
   $$\tilde{z}^{(i)} = \gamma z_{\text{norm}}^{(i)} + \beta$$
   為了不限制隱藏層只能使用「均值 0、變異數 1」的分布，我們引入了兩個**可學習參數** $\gamma$ 和 $\beta$。

---

### 2. 引入 $\gamma$ 與 $\beta$ 的直覺與「保留非線性表達能力」

如果我們強制隱藏層的輸出分布永遠是均值為 $0$、變異數為 $1$，在遇到如 Sigmoid 或 Tanh 等啟用函數時，會產生嚴重的問題：

* **非線性能力喪失的危機：** Sigmoid 函數在 $0$ 附近的區間幾乎是線性的。如果 $z$ 被強制限縮在均值 $0$、變異數 $1$ 的標準分布內，所有的啟用值都會被擠壓在該函數的中央線性區域，導致神經網路失去多層非線性擬合的能力。
* **恆等映射（Identity Function）的保留：** 當模型學習出 $\gamma = \sqrt{\sigma^2 + \epsilon}$ 且 $\beta = \mu$ 時，帶入上述公式會發現 $\tilde{z}^{(i)}$ 會完全還原為原始的 $z^{(i)}$。這代表這四個正規化公式允許模型在必要時主動撤銷正規化，將主導權交還給梯度更新，從而確保網路能靈活選擇最適合的非線性分布範圍。

---

## 二、 將 Batch Norm 融入深層神經網路的架構與細節

### 1. 運作位置：置於 $z$ 與 $a$ 之間

* **常規神經網路計算：** 
  $$x \to z = W a + b \to a = g(z)$$
* **引入 Batch Norm 後計算：** 
  $$z \to \text{Batch Norm (BN)} \to \tilde{z} \to a = g(\tilde{z})$$

亦即，Batch Norm 是對啟用前的 $z$ 值進行正規化，再將得到的 $\tilde{z}$ 送入啟用函數 $g(\cdot)$ 得到 $a$

---

### 2. 偏置參數 $b$ 的消除原理

在進行 Batch Norm 的層中，常規的偏置參數（Bias）$b$ 會變得完全多餘並被直接消除（或設為 $0$）：

1. **數學原理：** 因為 $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$，而在 Batch Norm 的第一步中，我們會減去該 Mini-batch 的均值 $\mu$。
2. **抵消效應：** 不論我們在 $z$ 上加上任何常數 $b^{[l]}$，在計算均值並執行減法（$z - \mu$）時，這個常數都會被完全減去而抵消。
3. **替代機制：** 原先偏置項 $b^{[l]}$ 的「控制平移/偏置」功能，已完全由 Batch Norm 運算中的可學習平移參數 $\beta^{[l]}$ 所取代。
4. **注意事項：** 上述「Bias 冗餘」僅適用於 `Linear / Conv -> BN` 直接相連的結構。若是 `Linear -> ReLU -> BN` 等非直接相連順序，則不能用同樣理由說明 bias 可以直接消除。

---

### 3. 卷積神經網路（CNN）中的 Batch Normalization

在全連接層（FC）中，BN 是針對各特徵單元獨立求 Mini-batch 均值；而在 CNN 中，BN 的運算邏輯隨 spatial 結構進行了調整：

* **特徵張量維度：** 給定輸入特徵圖張量 $Z \in \mathbb{R}^{N \times C \times H \times W}$（$N$: Batch size, $C$: Channels, $H$: Height, $W$: Width）。
* **Channel-wise 統計：** CNN 中的 BN 會對**每一個 Channel 分別計算** mean 與 variance。亦即，將同一 Channel 內跨 Batch ($N$)、跨空間高度 ($H$) 與寬度 ($W$) 的所有數值視為同一母體：
  $$\mu_c = \frac{1}{N H W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} z_{n,c,h,w}$$
  $$\sigma_c^2 = \frac{1}{N H W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} (z_{n,c,h,w} - \mu_c)^2$$
* **參數維度：** 因為統計是跨 $(N, H, W)$ 進行的，所以每一個 Channel 共享同一組縮放與平移參數，即：
  $$\gamma, \beta \in \mathbb{R}^C$$

---

### 4. 參數維度與 Mini-batch 更新

* **參數維度：** 若層 $l$ 的隱藏單元數為 $n^{[l]}$，則該層的 $z^{[l]}$ 維度為 $(n^{[l]}, 1)$。因此，該層的 $\gamma^{[l]}$ 與 $\beta^{[l]}$ 的維度同樣為 $(n^{[l]}, 1)$，它們與每一隱藏單元一一對應。
* **訓練更新步驟：** 在每個 Mini-batch 的迭代中，執行前向傳播（Forward Prop），在各隱藏層將 $z^{[l]}$ 替換為經由該 Mini-batch 計算出的 $\tilde{z}^{[l]}$：
  1. 計算前向傳播，得到 $z^{[l]}$。
  2. 利用當前 Mini-batch 計算 $\mu$ 與 $\sigma^2$，求得 $	ilde{z}^{[l]}$。
  3. 利用反向傳播（Backprop）計算梯度 $dW^{[l]}, d\gamma^{[l]}, d\beta^{[l]}$（此時無 $db^{[l]}$ 梯度）。
  4. 使用梯度下降（或結合 Momentum、RMSprop、Adam 等優化器）更新參數：
     $$W^{[l]} \leftarrow W^{[l]} -  lpha \cdot dW^{[l]}$$
     $$\beta^{[l]} \leftarrow \beta^{[l]} - \alpha \cdot d\beta^{[l]}$$
     $$\gamma^{[l]} \leftarrow \gamma^{[l]} -  lpha \cdot d\gamma^{[l]}$$

---

## 三、 Batch Norm 為什麼有效？深度原理解析

### 1. 對比輸入正規化（Input Normalization）

我們已知將輸入特徵 $X$ 正規化（均值 $0$，變異數 $1$）能將拉長、不均勻的損失函數等高線（Contours）轉化為圓形，使梯度下降能更直接、快速地收斂。Batch Norm 的直覺即是將此概念推廣到網路的深層內部——不僅正規化輸入層，也持續正規化深層網路中每一個隱藏層的輸入值（即前一層的輸出）。

---

### 2. 解決「協變量偏移」（Covariate Shift）的物理意義

* **什麼是協變量偏移？** 若訓練集的輸入分布（例如：只有黑貓的圖像）與測試集的分布（例如：彩色的貓）不同，即使真實對應關係（$X 	o Y$ 是貓或非貓）完全一致，模型也可能因輸入分布的改變而失效，必須重新訓練。
* **深層網路中的內部協變量偏移：** 對於第 3 隱藏層而言，它的輸入來自於前兩層的啟用值 $a^{[2]}$。然而，前兩層的參數 $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$ 在訓練過程中是不斷變化的，這導致第 3 層所看到的「輸入特徵分布」一直在劇烈漂移。
* **Batch Norm 的物理防護作用：** Batch Norm 限制了這些隱藏層數值分布漂移的程度。它強制確保不論前幾層如何更新，其輸出的均值與變異數都維持穩定（由該層的 $\beta$ 與 $\gamma$ 錨定）。
* **減弱層間耦合性（Weakening Coupling）：** 這使得每一層都可以更獨立地進行學習，而不需要無止境地去適應前幾層數值分布的劇烈波動，從而大幅加速了整個網路的聯合訓練與收斂速度。

---

### 3. BN 與 Learning Rate（學習率容忍度）

BN 對優化（Optimization）最實質的貢獻之一，在於控制了激活值的尺度（Activation Scale），使梯度更新更加平穩：

* **穩定優化軌跡：** BN 往往可以提高模型對 Learning Rate 的容忍度，使訓練過程可以使用相對較大的 Learning Rate。
* **直覺邏輯：**
  $$\text{BN} \to \text{更穩定的 Optimization} \to \text{可以使用較積極的 Learning Rate}$$
* **重要觀念澄清：** 這並不代表有了 BN 就可以無限制地提高 Learning Rate；而是 BN 降低了因為過大學習率導致梯度爆炸或數值不穩定的風險。

---

### 4. Mini-batch 帶來的輕微「正則化（Regularization）噪聲效應」

* **噪聲的來源：** 因為均值 $\mu$ 和變異數 $\sigma^2$ 是在單個 Mini-batch（如 64、128 或 256 個樣本）上估算，而非整個數據集，因此這兩個估算值本身帶有採樣噪聲（Noise）。
* **噪聲的傳播：** 在將 $z$ 轉換為 $\tilde{z}$ 的過程中，這種微小的噪聲會伴隨縮放與平移注入到隱藏層的啟用值中。
* **正則化效果：** 類似於 Dropout 隨機將啟用值乘以 0 或 1 來注入噪聲，Batch Norm 透過這種乘法（除以標準差）與加法（減去均值）的噪聲，迫使後續的隱藏單元不能過度依賴任何單一隱藏單元的精確數值，進而產生了輕微的正則化效果。
* **與 Batch Size 的關係：** Mini-batch 越小，採樣噪聲越劇烈，正則化效果越明顯；反之，若 Batch Size 設得很大（例如 512），估算值越接近母體，噪聲減小，正則化效應也會隨之減弱。
* **備註：** 正則化只是 Batch Norm 的副作用，不應將其視為主要的 Regularizer，必要時仍應搭配 Dropout 使用。

---

## 四、 測試時（Test Time）的 Batch Norm 運作方式

### 1. 單一測試樣本的挑戰

在訓練期，我們有 Mini-batch（如 $m=64$）可以據此計算均值與變異數。但在測試/預測期，我們往往需要一次只處理一個樣本（Single Example）。對單一數值求均值與變異數在數學上是沒有意義的（變異數會直接為 0）。

---

### 2. 解決方案：指數加權平均（Exponentially Weighted Average）

為了解決這個問題，我們必須在訓練過程中，額外估算並記錄一個不依賴單一測試 Batch 的全局均值與變異數：

* **估算方式：** 在訓練時，我們針對每一隱藏層 $l$，記錄每一個 Mini-batch 所計算出來的 $\mu^{\{1\}[l]}, \mu^{\{2\}[l]}, \dots$ 以及 ${\sigma^2}^{\{1\}[l]}, {\sigma^2}^{\{2\}[l]}, \dots$。
* **運行平均值（Running Average）：** 利用**指數加權平均（Exponentially Weighted Average）**的方式，隨著訓練進行，持續更新該層全局的均值與變異數估算值。
* 在實作中，這通常由深度學習框架（如 TensorFlow/PyTorch）在後台自動維護（通常設有特定的動量參數來控制運行平均的更新速度）。

---

### 3. 測試時的具體計算步驟

當神經網路接收到一個測試樣本時，在該隱藏層 $l$：

1. 直接使用訓練期間估算出的全局指數加權平均均值 $\mu_{	ext{running}}$ 與變異數 $\sigma^2_{\text{running}}$。
2. **計算測試標準化值：**
   $$z_{	ext{norm}} = \frac{z - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}}$$
3. 利用訓練好的學習參數 $\gamma$ 與 $\beta$，**計算最終的啟用前輸入值：**
   $$\tilde{z} = \gamma z_{\text{norm}} + \beta$$
4. 將 $\tilde{z}$ 送入啟用函數 $g(\tilde{z})$ 得到啟用值 $a$。

## 五、 各種 Normalization 技術比較（BN vs. LN vs. IN vs. GN）

不同的正規化方法主要差在**沿著張量的哪些維度計算 mean 與 variance**。以下為常見的正規化架構對比：

| 方法 | 統計計算方式 | 是否依賴 Batch Size ($N$) | 常見應用場景 |
| :--- | :--- | :---: | :--- |
| **BatchNorm (BN)** | 對 Batch ($N$) / Feature（或 Channel+Spatial $H,W$）求統計值 | **是 ($\checkmark$)** | CNN、一般視覺模型 |
| **LayerNorm (LN)** | 對單一樣本的所有 Feature（或 Channels）求統計值 | **否 ($\times$)** | Transformer、NLP、RNN |
| **InstanceNorm (IN)** | 對單一樣本、單一 Channel 的 Spatial ($H,W$) 求統計值 | **否 ($\times$)** | 圖像風格轉換 (Style Transfer) |
| **GroupNorm (GN)** | 將 Channels 分組，對單一樣本組內的 Channel+Spatial 求統計值 | **否 ($\times$)** | 小 Batch Size 下的 CNN 訓練/檢測 |

### 關鍵對比總結：
* **BatchNorm 依賴 Batch：** 若 Mini-batch 太小（例如 $N=2$ 或 $1$），估算出的均值與變異數極不準確，BN 效能會急劇下降。
* **LayerNorm 不依賴 Batch：** Transformer 等序列模型（輸入長度變動大且 Batch size 有限）中，LayerNorm 表現遠比 BatchNorm 穩定，因此 Transformer 普遍選擇使用 LayerNorm。

---


## 六、 進階部署優化：BatchNorm Folding（算子融合）

在模型訓練完成並準備部署進行推理（Inference / Deployment）時，推導可知 BN 運算可以完全融合（Fuse）到前一層的線性層（Linear 或 Conv）中，從而實現零額外計算開銷的推理加速。

### 1. 數學推導

在 Inference 階段，給定線性層輸出 $z = Wx + b$ 與 BN 運算：
$$\tilde{z} = \gamma \left( \frac{Wx + b - \mu}{\sqrt{\sigma^2 + \epsilon}}  \right) + \beta$$

展開並重組項次：
$$	ilde{z} = \left( \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} W 
ight) x + \left( \frac{\gamma (b - \mu)}{\sqrt{\sigma^2 + \epsilon}} + \beta 
ight)$$

### 2. 融合後的參數公式

定義融合後的新權重 $W'$ 與新偏置 $b'$：
$$W' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} W$$
$$b' = \frac{\gamma (b - \mu)}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

則原本兩層的計算直接縮減為單一線性運算：
$$\tilde{z} = W' x + b'$$

### 3. 實務部署意義
* **FC + BN：** $\text{Linear} + \text{BN} \to \text{Fused Linear}$
* **CNN + BN：** $\text{Conv} + \text{BN} \to \text{Fused Conv}$
* **結論：** 經過 BatchNorm Folding 後，推理階段**完全不需要執行 BN 運算**，不僅節省了計算時間，還減少了推論時的記憶體存取（Memory Bandwidth），是模型量化與部署（如 TensorRT、ONNX Runtime、CoreML）中極為關鍵的優化技術。

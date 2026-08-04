# 深度學習中的正則化技術（Regularization in Deep Learning）

---

## 一、 正規化的核心目的（Core Purpose）

1. **解決過擬合（Overfitting）**：
   - 當神經網路出現**高方差（High Variance）**問題（即訓練集表現極佳，但測試集/驗證集表現較差）時，正則化是首選且最有效的解決方法之一。
2. **與數據擴充的比較**：
   - 獲取更多訓練數據（Data Augmentation / Data Collection）雖能直接降低方差，但通常成本高昂且在許多場景下不可行。
   - 加入正則化能以極低的計算成本有效約束模型複雜度，減少方差並防止過擬合。

---

## 二、 L2 正則化（L2 Regularization）

L2 正則化是深度學習中最常用、最經典的正則化類型。

### 1. 邏輯回歸中的應用
在原始成本函數 $J(w, b)$ 中加入權重 $w$ 的歐幾里德範數（Euclidean Norm）平方作為懲罰項：

$$J(w, b) = rac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \|w\|_2^2$$

其中：
$$\|w\|_2^2 = w^T w = \sum_{j=1}^{n_x} w_j^2$$

### 2. 為何只針對 $W$ 而非 $b$？
- **$W$ 承載絕大部分參數**：權重向量 $W$ 通常是高維度參數矩陣，決定了模型的複雜度與擬合能力。
- **$b$ 的影響微乎其微**：偏置（Bias） $b$ 僅為單一純量或低維向量，對模型過擬合的影響極小。因此在實踐中，省略對 $b$ 的正則化可簡化計算而不影響效果。

### 3. 超參數 $\lambda$（Regularization Parameter）
- **作用**：控制懲罰強度。$\lambda$ 越大，模型越傾向於保持較小的權重。
- **設定方式**：屬於超參數，通常透過開發集（Dev Set）或交叉驗證（Cross-Validation）進行調優。
- **程式實作提醒**：在 Python 中，`lambda` 為系統保留關鍵字（用於匿名函數），因此在程式編寫時（例如 PyTorch、NumPy 或自定義函數參數），通常寫作 `lambd` 以避免語法衝突。

---

## 三、 L1 正則化（L1 Regularization）

### 1. 定義與公式
在成本函數中加入參數絕對值的和：

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{m} \sum_{j=1}^{n_x} |w_j| = J_0 + \frac{\lambda}{m} \|w\|_1$$

### 2. 特性：權重稀疏化（Sparsity）
- 使用 L1 正則化會導致權重矩陣變得**稀疏（Sparse）**，即 $W$ 中會有許多元素精確地變為 0。

### 3. 優點與實務侷限
- **理論優點**：有助於特徵選擇（Feature Selection）與模型壓縮（Model Compression）。
- **實務狀況**：如 Andrew Ng 所指出，在訓練深度神經網路時，L1 正則化對於縮小模型的實際效益有限，且稀疏矩陣可能增加計算複雜度。因此在深度學習中，**L2 正則化更為常用**。

---

## 四、 神經網路中的 L2 正則化：Frobenius 範數

在多層神經網路中，需對每一層的權重矩陣 $W^{[l]}$ 進行正則化。

### 1. 成本函數修改
$$J(W^{[1]}, b^{[1]}, \dots, W^{[L]}, b^{[L]}) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2$$

### 2. Frobenius 範數（Frobenius Norm）
在矩陣層面上，L2 範數被稱為 **Frobenius 範數**（標記為 $\|W\|_F^2$），定義為矩陣中所有元素平方的總和：

$$\|W^{[l]}\|_F^2 = \sum_{i=1}^{n^{[l-1]}} \sum_{j=1}^{n^{[l]}} (W_{i, j}^{[l]})^2$$

---

## 五、 權重衰減（Weight Decay）

L2 正則化在梯度下降更新時呈現出的動態特性，被稱為「**權重衰減**」。

### 1. 梯度與更新邏輯
1. 計算成本函數對權重矩陣 $W^{[l]}$ 的偏微分：
   $$dW^{[l]} = (	ext{from backprop}) + \frac{\lambda}{m} W^{[l]}$$

2. 帶入梯度下降更新公式：
   $$W^{[l]} \leftarrow W^{[l]} -  lpha dW^{[l]}$$
   $$W^{[l]} \leftarrow W^{[l]} -  lpha \left[ (	ext{from backprop}) + \frac{\lambda}{m} W^{[l]} 
ight]$$

3. 重組公式：
   $$W^{[l]} \leftarrow \left(1 - \frac{ lpha \lambda}{m}
ight) W^{[l]} -  lpha (	ext{from backprop})$$

### 2. 直觀理解
- 由於 $\left(1 - \frac{ lpha \lambda}{m}
ight)$ 是一個**略小於 1 的正數**，因此在每一次迭代更新前，權重矩陣都會先被「縮小（Decay）」一部分，再減去反向傳播得到的梯度。這就是 L2 正則化又被稱為權重衰減的原因。

---

## 六、 為什麼 L2 正則化能減少過擬合？（直覺解釋）

### 1. 直覺一：簡化神經網路結構（Suppressing Hidden Units）
- 當 $\lambda$ 設定很大時，為了最小化總成本函數 $J$，學習演算法被迫將權重矩陣 $W$ 壓縮至接近於 0 的數值。
- 當 $W  pprox 0$ 時，許多隱藏單元（Hidden Units）對最終輸出的影響會大幅減弱。
- **結果**：複雜且深層的神經網路在行為上會變得像是一個較簡單的小型網路（甚至接近線性/邏輯回歸），從而降低模型的變異數（Variance），將模型從過擬合狀態推向「剛剛好（Just Right）」的平衡狀態。
- *註：隱藏單元並未真正被刪除，而是其權重極小，影響力微乎其微。*

### 2. 直覺二：利用激活函數的線性區域（Linear Region of Activation）
- 以雙曲正切激活函數 $g(z) = 	anh(z)$ 為例：
  - 當正則化迫使權重 $W$ 變得很小時，計算出的 $z = W a + b$ 值也會落在接近 0 的較小區間內。
  - 在 $z  pprox 0$ 的區域，$	anh(z)$ 的函數圖形近乎一條斜率為 1 的**直線（Linear）**。
  - **結果**：如果網路中每一層的激活函數都處於接近線性的區間運作，那麼整個深層網路實質上只相當於一個巨大的**線性函數**。線性模型無法擬合極端複雜且曲折的非線性決策邊界，因此能有效防止過擬合。

---

## 七、 實作與調試注意事項（Debugging & Best Practices）

### 1. 正確定義調試目標（Cost Function Definition）
- **繪製 $J$ 曲線**：在調試梯度下降（Gradient Descent）時，繪製「成本函數 $J$ 隨迭代次數（Iterations）變化」的圖表，必須確保使用的是**包含正則化懲罰項的新成本函數 $J$**：
  $$J_{	ext{total}} = J_{	ext{original}} + \frac{\lambda}{2m} \sum \|W\|_F^2$$
- **常見錯誤**：若程式碼中只繪製了原始成本函數 $J_{	ext{original}}$，圖表可能會出現波動或無法保證單調遞減（Monotonically Decrease），從而誤判梯度下降過程有 Bug。

### 2. 實作檢查清單
- [x] 是否在所有 $dW^{[l]}$ 計算中加上了 $\frac{\lambda}{m} W^{[l]}$？
- [x] 是否在計算總成本 $J$ 時加上了 $\frac{\lambda}{2m} \sum_l \|W^{[l]}\|_F^2$？
- [x] Python 程式碼中變數名稱是否使用了 `lambd` 避免與關鍵字 `lambda` 衝突？

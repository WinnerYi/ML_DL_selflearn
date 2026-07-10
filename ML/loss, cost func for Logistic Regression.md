# 邏輯回歸：損失函數與代價函數（Logistic Regression Loss & Cost Function）

在邏輯回歸中，模型的預測輸出為非線性的 S 型函數（Sigmoid），形式為 $f_{\mathbf{w},b}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}\cdot\mathbf{x} + b)}}$，其值介於 $0$ 到 $1$ 之間，代表分類為正樣本（$y=1$）的機率。

---

## 一、 為什麼不能使用平方誤差（Squared Error）？

在線性回歸中，我們習慣使用**均方誤差（MSE）**。但在邏輯回歸中，這並不可行：
* **線性回歸的代價函數：** 是**凸函數（Convex Function）**，圖形呈現碗狀，只有一個全域最小值，梯度下降法能保證收斂。
* **邏輯回歸若用平方誤差：** 由於 $f_{\mathbf{w},b}(\mathbf{x})$ 的非線性特性，會導致代價函數變成**非凸函數（Non-convex Function）**。
* **非凸函數的風險：** 函數表面凹凸不平，存在大量**局部最小值（Local Minima）**。梯度下降法極可能卡在局部最優解，而無法找到真正的全域最小值。

---

## 二、 概念區分：損失（Loss）與代價（Cost）

為了構建一個凸函數，我們必須明確區分這兩個概念：
* **損失函數（Loss Function）：** 衡量模型在**單一訓練樣本**上的預測誤差，記作 $L(f_{\mathbf{w},b}(\mathbf{x}), y)$。
* **代價函數（Cost Function）：** 衡量模型在**整個訓練集（共 $m$ 個樣本）**上的平均誤差，記作 $J(\mathbf{w},b)$。

---

## 三、 邏輯回歸的損失函數（分段與簡化）

為了確保整體函數具備凸性，邏輯回歸採用了基於**對數（Logarithm）**的損失函數。

### 1. 分段定義與運作直覺
當真實標籤 $y$ 的取值不同時，處罰機制如下：

* **當 $y = 1$ 時（例如：惡性腫瘤）：**
  $$L(f_{\mathbf{w},b}(\mathbf{x}), 1) = -\log(f_{\mathbf{w},b}(\mathbf{x}))$$
  * *直覺：* 若模型預測 $f(\mathbf{x}) \to 1$（預測正確），損失值 $\to 0$；若預測 $f(\mathbf{x}) \to 0$（嚴重錯誤），損失值 $\to \infty$。
* **當 $y = 0$ 時（例如：良性腫瘤）：**
  $$L(f_{\mathbf{w},b}(\mathbf{x}), 0) = -\log(1 - f_{\mathbf{w},b}(\mathbf{x}))$$
  * *直覺：* 若模型預測 $f(\mathbf{x}) \to 0$（預測正確），損失值 $\to 0$；若預測 $f(\mathbf{x}) \to 1$（嚴重錯誤），損失值 $\to \infty$。

這種設計能強力激勵演算法避開錯誤的預測。

### 2. 簡化後的單一公式
由於二元分類的 $y$ 只能是 $0$ 或 $1$，我們可以運用代數技巧將上述分段函數合併為一個等式，以便於程式實作：

$$L(f_{\mathbf{w},b}(\mathbf{x}), y) = -y \log(f_{\mathbf{w},b}(\mathbf{x})) - (1 - y) \log(1 - f_{\mathbf{w},b}(\mathbf{x}))$$

> **等效性證明：**
> * 當 $y=1$ 時：後項 $(1-1)=0$ 消失，剩下 $-1 \cdot \log(f(\mathbf{x}))$。
> * 當 $y=0$ 時：前項 $0 \cdot \log(f(\mathbf{x}))=0$ 消失，剩下 $-1 \cdot \log(1-f(\mathbf{x}))$。

---

## 四、 邏輯回歸的代價函數（Cost Function）

將所有 $m$ 個樣本的簡化損失函數相加並取平均值，並將負號提取到加總符號 $\sum$ 之外，即得到廣泛應用的邏輯回歸代價函數：

$$J(\mathbf{w},b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(f_{\mathbf{w},b}\left(\mathbf{x}^{(i)}\right)\right) + \left(1 - y^{(i)}\right) \log\left(1 - f_{\mathbf{w},b}\left(\mathbf{x}^{(i)}\right)\right) \right]$$

---

## 五、 背後原理與核心性質

1. **統計學基礎：**
   這個特定的對數代價函數並非憑空想像，而是基於統計學中的**最大概似估計（Maximum Likelihood Estimation, MLE）**原理推導而來。它是一種在統計學上被證明能有效、穩定尋找模型參數的標準方法。
2. **凸性（Convex Property）：**
   透過這種對數轉換，原本非凸的代價函數重新轉化為**凸函數**。這意味著函數圖形沒有局部最小值，我們能 100% 可靠地使用**梯度下降法（Gradient Descent）**來尋找全局最優參數 $\mathbf{w}$ 和 $b$。
3. **直觀實作表現：**
   * **擬合度好**的模型（如劃分正確的決策邊界）：對應較低的 $J(\mathbf{w},b)$ 成本值。
   * **擬合度差**的模型：對應較高的 $J(\mathbf{w},b)$ 成本值。

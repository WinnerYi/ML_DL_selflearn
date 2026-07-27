# 深度學習筆記：深層神經網路的前向傳播 (Forward Propagation in Deep Neural Networks)

---

## 1. 深層神經網路的基本概念 (Basic Concepts)

* **深度的定義**：深度是一個程度問題。
  * **邏輯回歸 (Logistic Regression)**：被視為非常淺的模型（僅 1 層）。
  * **深層模型 (Deep Models)**：擁有更多隱藏層 (Hidden Layers) 的神經網路。
* **演進歷程**：深層神經網路是結合了邏輯回歸、單隱藏層神經網路、向量化 (Vectorization) 以及權重隨機初始化 (Random Initialization) 等概念的產物。
* **模型效能**：機器學習社群近年發現，深層神經網路能學習到許多傳統淺層模型無法有效處理的複雜非線性函數。

---

## 2. 層數與符號表示法 (Notation & Layer Counting)

### 層數計算規則
* **不計入輸入層**：在計算神經網路的層數時，**輸入層 (Input Layer) 不算入**，僅計算隱藏層與輸出層。
  * **邏輯回歸**：技術上被視為 1 層神經網路。
  * **2 層神經網路**：1 個隱藏層 + 1 個輸出層。
  * **4 層神經網路 ($L=4$)**：3 個隱藏層 + 1 個輸出層。

### 關鍵符號總覽
| 符號 | 說明 / 意義 |
| :--- | :--- |
| $L$ | 代表網路中的總層數（例如 $L = 4$） |
| $l$ | 代表目前的層級索引 ($l = 1, 2, \dots, L$) |
| $n^{[l]}$ | 第 $l$ 層的節點 (單元) 數量 |
| $n^{[0]} = n_x$ | 輸入層的特徵數量 |
| $n^{[L]}$ | 最後輸出層的單元數量 |
| $m$ | 訓練集中的樣本總數 |

---

## 3. 單一樣本的前向傳播 (Forward Propagation for Single Example)

對於任意層 $l$，單一樣本 $x$ 的前向傳播計算遵循以下兩步邏輯：

1. **線性計算 (Linear Part)**：
   $$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
2. **激活函數轉換 (Activation Part)**：
   $$a^{[l]} = g^{[l]}(z^{[l]})$$

### 邊界條件與符號說明：
* **邊界條件 (Boundary Conditions)**：
  * **第 0 層 (輸入層)**：$a^{[0]} = x$
  * **第 $L$ 層 (輸出層)**：$a^{[L]} = \hat{y}$ (最終預測值)
* **參數說明**：
  * $W^{[l]}$：第 $l$ 層的權重矩陣 (Weight Matrix)
  * $b^{[l]}$：第 $l$ 層的偏差向量 (Bias Vector)
  * $g^{[l]}(\cdot)$：第 $l$ 層使用的激活函數 (如 ReLU, Sigmoid, Tanh 等)

---

## 4. 向量化實作 (Vectorized Implementation for $m$ Examples)

為了同時處理整個訓練集（包含 $m$ 個樣本），我們將個別樣本的列向量**橫向堆疊 (Horizontally Stack)** 成大寫矩陣，實行全矩陣化的高效計算：

### 矩陣構建
* **輸入矩陣 $X$**：$X = A^{[0]} = \begin{bmatrix} \vert{} & \vert{} & & \vert{} \\ x^{(1)} & x^{(2)} & \dots & x^{(m)} \\ \vert{} & \vert{} & & \vert{} \end{bmatrix}$ （維度：$n^{[0]} \times m$）

### 向量化傳播公式
對第 $l$ 層 ($l = 1, 2, \dots, L$)：

1. **線性矩陣計算**：
   $$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
   *(註：$b^{[l]}$ 在程式碼中會透過 Broadcasting 自動擴展為與 $Z^{[l]}$ 相同維度)*

2. **逐元素激活轉換**：
   $$A^{[l]} = g^{[l]}(Z^{[l]})$$

### 最終輸出
* **第 $L$ 層輸出矩陣**：$A^{[L]} = \hat{Y} = \begin{bmatrix} \hat{y}^{(1)} & \hat{y}^{(2)} & \dots & \hat{y}^{(m)} \end{bmatrix}$ （維度：$n^{[L]} \times m$）

---

## 5. 跨層計算與 For 迴圈 (Cross-Layer Iteration)

* **顯式迴圈的必要性**：在向量化計算中，我們成功消除了對訓練樣本 $m$ 的迴圈，但在**跨層傳播 ($l=1 \to L$)** 時，顯式 (Explicit) 的 `for` 迴圈是不可避免且完全正確的。
* **偽程式碼結構 (Pseudocode)**：
  ```python
  A = X  # A[0] = X

  for l in range(1, L + 1):
      A_prev = A
      Z = np.dot(W[l], A_prev) + b[l]
      A = g[l](Z)

  Y_hat = A  # A[L] = Y_hat

# 邏輯回歸與梯度下降（Logistic Regression & Gradient Descent）筆記

> **核心學習目標**：理解計算圖（Computation Graph）如何透過鏈式法則進行反向傳播，並掌握從單一樣本擴展至 $m$ 個訓練樣本的梯度下降實作與優化方向。

---

## 一、 單一樣本的計算圖與推導 (Single Sample Computation)

### 1. 前向傳播 (Forward Propagation)
建立計算圖以逐步求出預測值與損失值：
* **輸入**：特徵向量 $x = [x_1, x_2]^T$、權重 $w = [w_1, w_2]^T$、偏置 $b$
* **步驟 1 (線性組合)**：$z = w_1 x_1 + w_2 x_2 + b = w^T x + b$
* **步驟 2 (激活函數)**：$a = \sigma(z)$（其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 為 Sigmoid 函數）
* **步驟 3 (損失計算)**：計算單一樣本損失 $L(a, y)$

---

### 2. 反向傳播求導 (Backward Propagation)
利用**鏈式法則 (Chain Rule)** 沿計算圖反向推導梯度（程式碼習慣以 `d` 開頭代表導數）：

1. **損失對激活值的偏導 ($da$)**：
   $$da = \frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}$$
2. **損失對線性組合的偏導 ($dz$)**：
   $$dz = \frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} = a - y$$
3. **損失對各參數的偏導 ($dw, db$)**：
   * $dw_1 = \frac{\partial L}{\partial w_1} = x_1 \cdot dz$
   * $dw_2 = \frac{\partial L}{\partial w_2} = x_2 \cdot dz$
   * $db = \frac{\partial L}{\partial b} = dz$

---

### 3. 單一樣本參數更新 (Parameter Update)
利用求出的梯度執行一步梯度下降（$ lpha$ 為學習率 Learning Rate）：
$$w_1 := w_1 - \alpha \cdot dw_1$$
$$w_2 := w_2 - \alpha \cdot dw_2$$
$$b := b - \alpha \cdot db$$

---

## 二、 $m$ 個訓練樣本的梯度下降實作 (m-Training Samples)

### 1. 成本函數 (Cost Function $J$)
當資料擴展至 $m$ 個樣本時，整體成本 $J(w, b)$ 為所有單一樣本損失 $L(a^{(i)}, y^{(i)})$ 的**平均值**：
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(a^{(i)}, y^{(i)})$$
其中 $z^{(i)} = w^T x^{(i)} + b$、$a^{(i)} = \sigma(z^{(i)})$。

---

### 2. 演算法流程與梯度累加

每一輪梯度下降（Single Epoch/Iteration）的執行步驟如下：

1. **初始化**：
   * 成本累加器 $J = 0$
   * 梯度累加器 $dw_1 = 0, dw_2 = 0, \dots, dw_n = 0, db = 0$
2. **樣本遍歷 (For Loop i = 1 to m)**：
   * **前向傳播**：$z^{(i)} = w^T x^{(i)} + b \implies a^{(i)} = \sigma(z^{(i)})$
   * **累加成本**：$J += L(a^{(i)}, y^{(i)})$
   * **計算誤差**：$dz^{(i)} = a^{(i)} - y^{(i)}$
   * **累加梯度**：
     * $dw_1 += x_1^{(i)} dz^{(i)}$
     * $dw_2 += x_2^{(i)} dz^{(i)}$
     * $db += dz^{(i)}$
3. **計算平均值 (Divide by m)**：
   * $J = J / m$
   * $dw_1 = dw_1 / m$
   * $dw_2 = dw_2 / m$
   * $db = db / m$

---

### 3. 整體參數更新 (Parameter Update)
使用所有樣本的**平均梯度**更新一次參數：
$$w_1 := w_1 - \alpha \cdot dw_1$$
$$w_2 := w_2 - \alpha \cdot dw_2$$
$$b := b - \alpha \cdot db$$

> **注意**：上述完整流程僅代表梯度下降的「**一步（One Step）**」，需重複執行多次直到成本函數 $J$ 收斂。

---

## 三、 現有實作的瓶頸與未來優化 (Vectorization)

###  效能瓶頸
目前的傳統實作包含兩層**顯式迴圈 (Explicit For-Loops)**：
1. **外層迴圈**：遍歷 $m$ 個訓練樣本（$i = 1 \dots m$）。
2. **內層迴圈**：遍歷 $n$ 個特徵（$j = 1 \dots n$），用於計算各個 $dw_j$。

在大數據集或高維度特徵（深度學習）場景下，顯式迴圈會造成嚴重的運算效率低下。

###  解決方案：向量化 (Vectorization)
利用矩陣/向量運算（如 NumPy、PyTorch），消除顯式 for 迴圈，改以一次性矩陣乘法並行處理所有 $m$ 個樣本與 $n$ 個特徵，能大幅提升運算速度與擴展能力。

---

> ** 筆記小結**
> * **計算圖**：幫助我們透過直觀的節點拆解鏈式法則，理解反向傳播求導機制。
> * **單樣本 ➔ 全體樣本**：從單一損失 $L$ 推導至全體成本 $J$，梯度即為個體梯度的算術平均。
> * **下一階段重點**：將這套邏輯改寫為**向量化矩陣運算 (Matrix Vectorization)**。

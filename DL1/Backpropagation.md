# 類神經網路反向傳播（Backpropagation）核心筆記

## 1. 基礎觀念：從邏輯迴歸（Logistic Regression）出發
單層的邏輯迴歸是神經網路反向傳播的基石。

* **前向傳播 (Forward)**：
$$z = Wx + b \longrightarrow a = \sigma(z) \longrightarrow \mathcal{L}(a, y)$$

* **反向傳播 (Backward)**：
  * $da = \frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}$
  * $dz = \frac{\partial \mathcal{L}}{\partial z} = da \cdot \sigma'(z) = a - y$ *(核心極簡結果)*
  * $dW = dz \cdot x^T$
  * $db = dz$

---

## 2. 雙層神經網路推導（Single Sample）

反向傳播的本質就是**將連鎖律（Chain Rule）從輸出層逆向執行回輸入層**。

### 輸出層（Layer 2）
計算邏輯與邏輯迴歸完全一致，僅將輸入 $x$ 替換為上一層的輸出 $a^{[1]}$：
* $dz^{[2]} = a^{[2]} - y$
* $dW^{[2]} = dz^{[2]} \cdot (a^{[1]})^T$
* $db^{[2]} = dz^{[2]}$

### 隱藏層（Layer 1）
誤差從第二層反向傳回第一層，需要乘以權重並結合激活函數的導數：
* $dz^{[1]} = (W^{[2]})^T dz^{[2]} \odot g'^{[1]}(z^{[1]})$ 
  *(註： ⊙ 或 * 代表 Element-wise 元素對應相乘)*
* $dW^{[1]} = dz^{[1]} \cdot x^T$  ( **註：** 此處 $x$ 即為 $a^{[0]})
* $db^{[1]} = dz^{[1]}$

---

## 3. 向量化（Vectorization）與批次處理（$m$ 個樣本）

為了利用矩陣運算同時處理 $m$ 個訓練樣本，將向量按欄（Column）堆疊成大寫矩陣，且梯度需要對 $m$ 取平均值：

* **矩陣維度定義**：
  * $X, A^{[1]}, A^{[2]} \in \mathbb{R}^{n \times m}$
  * $Z^{[1]}, Z^{[2]} \in \mathbb{R}^{n \times m}$

* **向量化後的 6 大核心公式**：
  1. $dZ^{[2]} = A^{[2]} - Y$
  2. $dW^{[2]} = \frac{1}{m} dZ^{[2]} (A^{[1]})^T$
  3. $db^{[2]} = \frac{1}{m} \text{np.sum}(dZ^{[2]}, \text{axis}=1, \text{keepdims}=\text{True})$
  4. $dZ^{[1]} = (W^{[2]})^T dZ^{[2]} \odot g'^{[1]}(Z^{[1]})$
  5. $dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$
  6. $db^{[1]} = \frac{1}{m} \text{np.sum}(dZ^{[1]}, \text{axis}=1, \text{keepdims}=\text{True})$

---

## 4. 除錯心法：維度檢查（Dimensions Check）

實作神經網路時，**90% 的 Bug 來自矩陣維度不符**。請確保以下恆等關係：

> * $W^{[l]}$ 與 $dW^{[l]}$ 維度恆相等，均為 $(n^{[l]}, n^{[l-1]})$
> * $b^{[l]}$ 與 $db^{[l]}$ 維度恆相等，均為 $(n^{[l]}, 1)$
> * $Z^{[l]}, dZ^{[l]}, A^{[l]}$ 維度恆相等，均為 $(n^{[l]}, m)$

---

## 5. 總結與後續關鍵

1. **實作重點**：不需要每次都從頭推導微積分，只要掌握上述 **6 個核心公式** 並嚴格維持維度一致，即可順利寫出反向傳播。
2. **下一步（初始化）**：權重矩陣 $W$ **絕不能初始化為全 0**（會導致所有神經元學到完全一樣的特徵，即「對稱性問題 Symmetric Breaking Problem」），必須進行**隨機初始化**（如 He Initialization 或 Xavier Initialization）。

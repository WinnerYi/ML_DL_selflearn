# 深度學習基礎：NumPy 向量化與邏輯迴歸全景筆記

## 一、 向量化的核心觀念與硬體原理

### 1. 什麼是向量化（Vectorization）？
* **基本定義**：向量化是指在程式碼中**消除顯式 `for` 迴圈（explicit for loops）**的技術與風格。
* **技術地位**：在深度學習處理大規模資料集時，向量化是讓程式碼達到可接受執行速度的**必備核心技能**。

---

### 2. 效能對比（以 1,000,000 維度向量運算為例）
在 Jupyter Notebook 的基準測試中，計算 $Z = \mathbf{w}^T \mathbf{x} + b$：
* **非向量化（顯式 `for` 迴圈）**：約需 **400 ~ 500 毫秒 (ms)**。
* **向量化版本（`np.dot`）**：僅需約 **1.5 毫秒 (ms)**。
* **結論**：向量化提升了約 **300 倍** 的運算速度。這意味著原本需要運行 **5 小時** 的模型訓練，透過向量化可以在 **1 分鐘** 內完成。

---

### 3. 底層技術原理：SIMD 與平行運算
* **SIMD（Single Instruction Multiple Data，單指令多數據）**：不論是 CPU 還是 GPU，都擁有硬體層級的平行運算指令。GPU 在處理 SIMD 上極為優異，而現代 CPU 亦具備強大的 SIMD 能力。
* **NumPy 內建函數**：當呼叫 `np.dot` 或 `np.exp` 等內建函數時，NumPy 會底層調用高優化的 C / Fortran 函式庫（如 BLAS, LAPACK），直接調用硬體的 **平行運算（Parallelism）** 特性，而非逐一執行指令。

---

### 4. 向量化實作金律
> **深度學習第一準則**：只要有可能，就盡量避免使用顯式的 `for` 迴圈。

---

## 二、 常見運算範例與 NumPy 內建函數對照

### 1. 矩陣與向量乘法
* **運算內容**：計算 $u_i = \sum_{j} A_{ij} v_j$
* **非向量化**：需要兩層嵌套的 `for` 迴圈遍歷索引 $i$ 與 $j$。
* **向量化**：直接寫 `u = np.dot(A, v)`。

### 2. 元素級運算（Element-wise Operations）
對向量 $\mathbf{v}$ 中的每個元素套用運算：

| 運算類型 | 非向量化做法 | 向量化做法 |
| :--- | :--- | :--- |
| **指數運算** ($e^{v_i}$) | 初始化全零向量 + `for` 迴圈逐一計算 | `u = np.exp(v)` |
| **自然對數** ($\\ln(v_i)$) | `for` 迴圈呼叫 `math.log` | `u = np.log(v)` |
| **絕對值** ($\\mid v_i \\mid$) | `for` 迴圈呼叫 `abs()` | `u = np.abs(v)` |
| **ReLU/最大值** ($\\max(v_i, 0)$) | `if-else` 判斷式 + 迴圈 | `u = np.maximum(v, 0)` |
| **平方** ($v_i^2$) | `for` 迴圈累乘 | `u = v ** 2` |
| **倒數** ($1 / v_i$) | `for` 迴圈逐項相除 | `u = 1 / v` |

---

## 三、 邏輯迴歸（Logistic Regression）向量化演進

### 1. 資料結構與矩陣維度定義
設定資料集包含 $n_x$ 個特徵，以及 $m$ 個訓練樣本：

* **輸入矩陣 $\mathbf{X}$**：維度為 $(n_x, m)$。將所有訓練樣本成列堆疊：<br>
$$\mathbf{X} = $$
$$\begin{bmatrix} \mid & \mid & & \mid \\ \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \dots & \mathbf{x}^{(m)} \\ \mid & \mid & & \mid \end{bmatrix}$$
* **權重向量 $\mathbf{w}$**：維度為 $(n_x, 1)$。
* **偏差項 $b$**：實數（Scalar）。
* **真實標籤 $\mathbf{Y}$**：維度為 $(1, m)$ 的列向量：
  $$\mathbf{Y} = \begin{bmatrix} y^{(1)} & y^{(2)} & \dots & y^{(m)} \end{bmatrix}$$

---

### 2. 步驟一：消除「特徵維度 $n_x$」的 `for` 迴圈
在未完全向量化前，梯度下降需要遍歷 $m$ 個樣本與 $n_x$ 個特徵：

* **傳統做法**：分別初始化 $dw_1, dw_2, \dots$ 並用 `for` 迴圈累加。
* **優化做法**：
  1. 將梯度初始化為向量：`dw = np.zeros((nx, 1))`
  2. 針對特徵進行向量運算：`dw += x_i * dz_i`
  3. 取平均：`dw /= m`
* **成果**：將兩層迴圈減少為一層（僅剩遍歷 $m$ 個樣本的迴圈）。

---

### 3. 步驟二：完全消除「樣本數 $m$」的 `for` 迴圈

#### (A) 向量化前向傳播（Forward Propagation）
一次性計算所有 $m$ 個樣本的線性結果與激活值：

1. **計算線性輸出 $\mathbf{Z}$**：
   $$\mathbf{Z} = \mathbf{w}^T \mathbf{X} + b$$
   * **維度**：$(1, n_x) \times (n_x, m) + \text{scalar} \rightarrow (1, m)$
   * **關鍵技術（廣播機制 Broadcasting）**：Python 會自動將純量 $b$ 複製擴展成 $(1, m)$ 的向量，直接與 $\mathbf{w}^T \mathbf{X}$ 進行元素級加法。
   * **代碼**：`Z = np.dot(w.T, X) + b`

2. **計算激活值 $\mathbf{A}$**：
   $$\mathbf{A} = \sigma(\mathbf{Z}) = \begin{bmatrix} a^{(1)} & a^{(2)} & \dots & a^{(m)} \end{bmatrix}$$
   * **維度**：$(1, m)$
   * **代碼**：`A = sigmoid(Z)`

#### (B) 向量化反向傳播（Backward Propagation）

1. **誤差矩陣 $\mathbf{dZ}$**：
   $$\mathbf{dZ} = \mathbf{A} - \mathbf{Y} = \begin{bmatrix} a^{(1)}-y^{(1)} & a^{(2)}-y^{(2)} & \dots & a^{(m)}-y^{(m)} \end{bmatrix}$$
   * **維度**：$(1, m)$

2. **偏差導數 $db$**：
   $$db = \frac{1}{m} \sum_{i=1}^{m} dz^{(i)}$$
   * **代碼**：`db = (1 / m) * np.sum(dZ)`

3. **權重導數 $\mathbf{dW}$**：
   $$\mathbf{dW} = \frac{1}{m} \mathbf{X} \mathbf{dZ}^T$$
   * **維度推導**：$(n_x, m) \times (m, 1) \rightarrow (n_x, 1)$，完美符合權重 $\mathbf{w}$ 的維度。
   * **原理**：這行矩陣乘法一口氣完成了所有訓練樣本 $\mathbf{x}^{(i)} dz^{(i)}$ 的乘積與加總累加。
   * **代碼**：`dW = (1 / m) * np.dot(X, dZ.T)`

---

## 四、 邏輯迴歸單次梯度下降完整演算法

綜合上述步驟，單次迭代的向量化高效實作流程如下：

```python
import numpy as np

# 1. 前向傳播 (Forward Propagation)
Z = np.dot(w.T, X) + b        # Shape: (1, m)
A = sigmoid(Z)                # Shape: (1, m)

# 2. 反向傳播 (Backward Propagation)
dZ = A - Y                    # Shape: (1, m)
dW = (1 / m) * np.dot(X, dZ.T)# Shape: (nx, 1)
db = (1 / m) * np.sum(dZ)     # Scalar

# 3. 參數更新 (Parameter Update)
w = w - alpha * dW
b = b - alpha * db
```

---

## 五、 最終總結與注意事項

1. **全雙重迴圈消除**：透過向量化與廣播機制，我們成功消除了**特徵維度 ($n_x$)** 與 **訓練樣本維度 ($m$)** 的顯式 `for` 迴圈。
2. **廣播機制（Broadcasting）**：NumPy 處理不同形狀陣列相加（例如向量加純量 $b$）的核心優化，是實現簡潔代碼的功臣。
3. **迭代次數限制**：向量化消除的是**單次迭代內部**對資料和特徵的迴圈。若要進行多輪梯度下降（例如執行 1,000 次），最外層仍然需要一個 `for` 迴圈來控制總迭代次數。

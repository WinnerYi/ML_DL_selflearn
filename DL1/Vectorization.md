# 深度學習基礎：NumPy 向量化與邏輯迴歸全景筆記

# 一、向量化的核心觀念與硬體原理

## 1. 什麼是向量化（Vectorization）

- **基本定義**：向量化是指在程式碼中**消除顯式 `for` 迴圈（Explicit For Loops）**的技術與程式設計風格。
- **技術地位**：在深度學習處理大規模資料集時，向量化是讓程式碼達到可接受執行速度的**必備核心技能**。

---

## 2. 效能對比（以 1,000,000 維向量運算為例）

在 Jupyter Notebook 的基準測試中，計算

$$
Z = \mathbf{w}^T \mathbf{x} + b
$$

- **非向量化（顯式 `for` 迴圈）**：約 **400～500 ms**
- **向量化（`np.dot`）**：約 **1.5 ms**

### 結論

向量化可提升約 **300 倍** 的運算速度。

也就是說，原本需要 **5 小時** 的模型訓練，透過向量化後可能只需要 **1 分鐘**。

---

## 3. 底層技術原理：SIMD 與平行運算

### SIMD（Single Instruction Multiple Data）

CPU 與 GPU 都支援 SIMD（單指令多資料）平行運算。

其中：

- GPU 特別擅長大量 SIMD 運算。
- 現代 CPU 也具有 AVX、SSE 等 SIMD 指令集。

### NumPy 的優勢

當呼叫

```python
np.dot()
np.exp()
np.log()
```

等函式時，NumPy 並不是用 Python 的 `for` 迴圈。

它會呼叫底層高度最佳化的 C / Fortran 函式庫，例如：

- BLAS
- LAPACK

並利用 CPU/GPU 的平行運算能力，因此速度非常快。

---

## 4. 向量化實作金律

> **深度學習第一準則：**
>
> **只要有可能，就不要使用顯式 `for` 迴圈。**

---

# 二、常見運算範例與 NumPy 內建函式

## 1. 矩陣與向量乘法

計算

$$
u_i=\sum_j A_{ij}v_j
$$

### 非向量化

需要兩層 `for` 迴圈：

```python
for i:
    for j:
        ...
```

### 向量化

```python
u = np.dot(A, v)
```

---

## 2. 元素級運算（Element-wise Operations）

對向量 $\mathbf{v}$ 的每一個元素做相同運算。

| 運算 | 非向量化 | 向量化 |
|------|----------|---------|
| 指數 $e^{v_i}$ | `for` + `math.exp()` | `np.exp(v)` |
| 自然對數 $\ln(v_i)$ | `for` + `math.log()` | `np.log(v)` |
| 絕對值 $\lvert v_i \rvert$ | `for` + `abs()` | `np.abs(v)` |
| ReLU $\max(v_i,0)$ | `if-else` | `np.maximum(v,0)` |
| 平方 $v_i^2$ | `for` | `v ** 2` |
| 倒數 $\frac{1}{v_i}$ | `for` | `1 / v` |

---

# 三、邏輯迴歸（Logistic Regression）向量化演進

## 1. 資料結構與矩陣維度

假設：

- 特徵數：$n_x$
- 訓練樣本數：$m$

### 輸入矩陣

$$
\mathbf{X}=
\begin{bmatrix}
| & | & & |\\
\mathbf{x}^{(1)} &
\mathbf{x}^{(2)} &
\cdots &
\mathbf{x}^{(m)}\\
| & | & & |
\end{bmatrix}
$$

維度：

$$
(n_x,m)
$$

---

### 權重向量

$$\mathbf{w}$$

維度：

$$(n_x,1)$$

---

### 偏差

$$
b
$$

為一個 Scalar。

---

### 真實標籤

$$ \mathbf{Y} = \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix} $$

維度：

$$
(1,m)
$$

---

## 2. 第一步：消除特徵維度 $n_x$ 的迴圈

原始做法：

需要同時遍歷

- 每個樣本
- 每個特徵

因此共有兩層 `for`。

改善方式：

```python
dw = np.zeros((nx,1))

dw += x_i * dz_i

dw /= m
```

結果：

成功消除一層迴圈，只剩樣本數 $m$ 的迴圈。

---

## 3. 第二步：消除樣本數 $m$ 的迴圈

### (A) Forward Propagation

#### Step 1：計算線性輸出

$$ \mathbf{Z} = \mathbf{w}^T\mathbf{X}+b $$

維度：

$$
(1,n_x)
\times
(n_x,m)
\rightarrow
(1,m)
$$

由於 Broadcasting，

Python 會自動把 Scalar $b$

擴展成

$$
(1,m)
$$

因此可以直接相加。

程式：

```python
Z = np.dot(w.T, X) + b
```

---

#### Step 2：Sigmoid

$$
\mathbf{A}
=
\sigma(\mathbf{Z})
=
\begin{bmatrix}
a^{(1)}
&
a^{(2)}
&
\cdots
&
a^{(m)}
\end{bmatrix}
$$

維度：

$$
(1,m)
$$

程式：

```python
A = sigmoid(Z)
```

---

### (B) Backward Propagation

#### Step 1：誤差

$$
dZ
=
A-Y
$$

即

$$
dZ=
\begin{bmatrix}
a^{(1)}-y^{(1)}
&
a^{(2)}-y^{(2)}
&
\cdots
&
a^{(m)}-y^{(m)}
\end{bmatrix}
$$

維度：

$$
(1,m)
$$

---

#### Step 2：偏差梯度

$$ db = \frac{1}{m} \sum_{i=1}^{m} dz^{(i)} $$

程式：

```python
db = (1/m) * np.sum(dZ)
```

---

#### Step 3：權重梯度

$$ dW = \frac{1}{m} X dZ^T $$

維度推導：

$$ (n_x,m) \times (m,1) = (n_x,1) $$

因此與權重

$$
w
$$

的維度完全一致。

程式：

```python
dW = (1/m) * np.dot(X, dZ.T)
```

矩陣乘法一次完成所有樣本

$$
x^{(i)}dz^{(i)}
$$

的乘積與累加。

---

# 四、完整一次梯度下降（Gradient Descent）

```python
import numpy as np

# Forward Propagation
Z = np.dot(w.T, X) + b
A = sigmoid(Z)

# Backward Propagation
dZ = A - Y
dW = (1 / m) * np.dot(X, dZ.T)
db = (1 / m) * np.sum(dZ)

# Parameter Update
w = w - alpha * dW
b = b - alpha * db
```

---

# 五、重點整理

1. 利用向量化，同時消除了：
   - 特徵維度 $n_x$ 的 `for`
   - 樣本維度 $m$ 的 `for`

2. Broadcasting 讓 Scalar（例如 $b$）能自動擴展，直接與矩陣做運算。

3. 向量化消除的是**單次梯度下降內部**的迴圈。

4. 若要訓練 1000 次 Gradient Descent，最外層仍然需要：

```python
for i in range(1000):
    ...
```

來控制迭代次數。

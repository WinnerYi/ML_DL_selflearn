# 邏輯迴歸（Logistic Regression）參數最佳化學習筆記

## 1. 核心目標
邏輯迴歸的最佳化目標是尋找一組參數 $w$ 和 $b$，以最小化代價函數（Cost Function） $J(w,b)$。一旦找到理想的參數，模型就能針對新輸入的數據（如患者的腫瘤大小與年齡）來預測標籤 $y=1$ 的機率。

---

## 2. 梯度下降法（Gradient Descent）的應用
為了最小化代價函數，我們同樣使用梯度下降演算法。其核心步驟是不斷重複更新參數，直到收斂：

### 更新公式
$$w_j = w_j -  alpha \frac{\partial J(w,b)}{\partial w_j}$$

$$b = b -  alpha \frac{\partial J(w,b)}{\partial b}$$

其中 $alpha$ 為學習率（Learning Rate）。

### 微分項（Derivative terms）內容
* **對於 $w_j$ 的偏微分：**
  $$\frac{\partial J(w,b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} 
ight) x_j^{(i)}$$

* **對於 $b$ 的偏微分：**
  $$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} 
ight)$$

### 更新原則
* **同時更新（Simultaneous updates）：** 必須先計算出所有參數的右側更新值，再同時覆蓋掉左側的舊參數值。

---

## 3. 與線性迴歸的異同
雖然邏輯迴歸的梯度下降公式在形式上與線性迴歸完全相同，但兩者本質上是完全不同的演算法，關鍵在於 $f(x)$ 的定義不同：

* **線性迴歸（Linear Regression）：**
  $$f_{w,b}(x) =  ec{w} \cdot  ec{x} + b$$
* **邏輯迴歸（Logistic Regression）：**
  $$f_{w,b}(x) = g( ec{w} \cdot  ec{x} + b) = \frac{1}{1 + e^{-( ec{w} \cdot  ec{x} + b)}}$$
  即將 $ ec{w} \cdot  ec{x} + b$ 套用 **Sigmoid 函數** 後的結果。

---

## 4. 提升效率與效能的方法
1. **向量化（Vectorization）：** 使用向量化實作（如矩陣運算）可以顯著提升梯度下降在邏輯迴歸中的運算速度。
2. **特徵縮放（Feature Scaling）：** 將不同特徵的數值範圍縮放到相似的大小（例如 $-1$ 到 $+1$ 之間），可以幫助梯度下降更平穩且快速地收斂。

---

## 5. 實作工具
在實際應用中，許多機器學習從業者會使用 **scikit-learn** 這類熱門的 Python 函式庫來訓練分類模型，這比手寫底層程式碼更為高效且常用。

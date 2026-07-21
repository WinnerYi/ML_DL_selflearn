# 計算圖（Computation Graph）與導數計算筆記

## 1. 計算圖的核心目的與方向

計算圖是用來結構化與組織神經網路計算的一種工具，將複雜的函數拆解為節點與路徑，主要包含兩個運作方向：

* **正向傳遞（Forward Propagation / Forward Path）：**
  * **方向：** 從左至右（Left-to-Right）。
  * **目的：** 計算並得出神經網路的最終輸出值（如成本函數 $J$）。
* **反向傳遞（Backward Pass / Back Propagation）：**
  * **方向：** 從右至左（Right-to-Left）。
  * **目的：** 計算最終輸出對各中間變數與輸入變數的**梯度（Gradients）與導數（Derivatives）**。

---

## 2. 計算圖的組成步驟與正向計算

以函數 $J = 3(a + bc)$ 為例，可將計算拆解為三個連續步驟：

1. **計算 $u$：** $u = b \times c$（輸入為 $b, c$）
2. **計算 $v$：** $v = a + u$（輸入為 $a, u$）
3. **計算 $J$：** $J = 3 \times v$（輸出最終目標值 $J$）

### 實際數值運算範例
假設輸入變數為 $a = 5, b = 3, c = 2$：
* $u = 3 \times 2 = 6$
* $v = 5 + 6 = 11$
* $J = 3 \times 11 = 33$

---

## 3. 連鎖律（Chain Rule）與反向傳播

反向傳播的核心原理是微積分中的**連鎖律**。透過從右向左回推，後續計算出的導數結果可以被前面的節點重複使用，這是極具效率的求導方式。

### 連鎖律公式與直觀理解
若變數 $a$ 影響 $v$，而 $v$ 又影響 $J$，則 $a$ 對 $J$ 的總影響可拆解為：

$$\frac{dJ}{da} = \frac{dJ}{dv} \cdot \frac{dv}{da}$$

* **直觀理解：** 當微調 $a$ 時，$v$ 會隨之改變（變化率由 $\frac{dv}{da}$ 決定）；而 $v$ 的改變又會進一步導致 $J$ 的改變（變化率由 $\frac{dJ}{dv}$ 決定）。兩者相乘即為 $a$ 對 $J$ 的最終影響。

---

## 4. 具體導數計算推導

基於上述範例（$J = 3v, v = a + u, u = bc$，且 $a=5, b=3, c=2$），反向推導過程如下：

| 變數 | 導數符號 | 數學表示式 | 具體推導與計算過程 | 計算結果 |
| :--- | :--- | :--- | :--- | :--- |
| **$v$** | `dv` | $\frac{dJ}{dv}$ | 若 $v$ 增加 $0.001$，$J$ 增加 $0.003$（3倍），故 $\frac{dJ}{dv} = 3$ | **3** |
| **$a$** | `da` | $\frac{dJ}{da}$ | 因 $v = a + u$，$\frac{dv}{da} = 1$。<br>由連鎖律：$\text{da} = \text{dv} \times \frac{dv}{da} = 3 \times 1$ | **3** |
| **$u$** | `du` | $\frac{dJ}{du}$ | 因 $v = a + u$，$\frac{dv}{du} = 1$。<br>由連鎖律：$\text{du} = \text{dv} \times \frac{dv}{du} = 3 \times 1$ | **3** |
| **$b$** | `db` | $\frac{dJ}{db}$ | 因 $u = bc$ 且 $c=2$，$\frac{du}{db} = c = 2$。<br>由連鎖律：$\text{db} = \text{du} \times \frac{du}{db} = 3 \times 2$ | **6** |
| **$c$** | `dc` | $\frac{dJ}{dc}$ | 因 $u = bc$ 且 $b=3$，$\frac{du}{dc} = b = 3$。<br>由連鎖律：$\text{dc} = \text{du} \times \frac{du}{dc} = 3 \times 3$ | **9** |

---

## 5. 程式實作命名慣例

在撰寫反向傳播的程式碼（如 Python）時，為了避免變數名稱過於冗長（例如 `final_output_derivative_with_respect_to_var`），會採用簡化命名標準：

* **慣例：** 統一將「最終輸出 $J$ 對變數 `var` 的導數 $\frac{dJ}{d(\text{var})}$」簡寫為 **`dvar`**。
* **常見範例：**
  * `dv` 代表 $\frac{dJ}{dv}$
  * `da` 代表 $rac{dJ}{da}$
  * `db` 代表 $rac{dJ}{db}$
  * `dc` 代表 $rac{dJ}{dc}$

---

## 6. 深度學習中的應用與總結

1. **優化目標：** 計算圖特別適合處理具有「特定輸出變數（如 $J$）」的模型。在深度學習（如邏輯回歸、神經網路）中，$J$ 通常代表**成本函數（Cost Function）**，即模型優化時需要最小化的目標。
2. **效率關鍵：** 
   * **正向（左 $\rightarrow$ 右）：** 計算神經網路的預測值與成本 $J$。
   * **反向（右 $\rightarrow$ 左）：** 沿著反向路徑利用連鎖律計算各參數梯度，供梯度下降法（Gradient Descent）更新權重使用。

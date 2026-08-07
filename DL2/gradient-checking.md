gradient-checking-notes
Detailed study notes on Gradient Checking in deep learning, covering math formulation, implementation steps, evaluation metrics, and practical tips.

Instructions

梯度檢驗（Gradient Checking）學習筆記

**梯度檢驗（Gradient Checking，簡稱 Grad Check）**是一項能大幅節省時間，並協助找出反向傳播（Backpropagation）實作中隱藏 Bug 的高效技術。以下是實作步驟、數學原理、評估指標與實作建議的完整整理：

一、 核心實作步驟

1. 參數重塑與拼接 (Reshaping & Concatenation)

在神經網路中，參數通常分散在各層的矩陣與向量中（如 $W^{[1]}, b^{[1]}, \dots, W^{[L]}, b^{[L]}$）。為了方便進行梯度檢驗，必須先進行整合：

參數向量 $\theta$：將所有權重矩陣 $W$ 與偏置向量 $b$ 全部重塑（Reshape）為一維向量，並依序拼接成一個巨大的參數向量 $\theta$。此時，代價函數 $J$ 即可簡化表示為以 $\theta$ 為變數的函數 $J(\theta)$。
導數向量 $d\theta$：以完全相同的重塑與拼接順序，將反向傳播計算出的導數（$dW^{[1]}, db^{[1]}, \dots, dW^{[L]}, db^{[L]}$）整合為一個與 $\theta$ 維度完全相同的巨大導數向量 $d\theta$。
2. 近似梯度計算 ($d\theta_{\text{approx}}$)

梯度檢驗的核心是利用**雙邊差分（Two-sided difference）**來逼近每個參數的偏導數。針對 $\theta$ 中的每一個分量 $i$ 執行以下步驟：

加上微小值 $\epsilon$：保持其他所有參數不變，僅將第 $i$ 個元素加上 $\epsilon$（即 $\theta_i + \epsilon$），並計算此時的代價函數值 $J(\theta_1, \dots, \theta_i + \epsilon, \dots)$。
減去微小值 $\epsilon$：同樣保持其他參數不變，僅將第 $i$ 個元素減去 $\epsilon$（即 $\theta_i - \epsilon$），計算代價函數值 $J(\theta_1, \dots, \theta_i - \epsilon, \dots)$。
計算近似值 $d\theta_{\text{approx}}[i]$：將上述兩個代價函數值相減，並除以 $2\epsilon$： $$d\theta_{\text{approx}}[i] = \frac{J(\theta_1, \dots, \theta_i + \epsilon, \dots) - J(\theta_1, \dots, \theta_i - \epsilon, \dots)}{2\epsilon}$$ 這個計算結果 $d\theta_{\text{approx}}[i]$ 應該要非常接近反向傳播算出來的真實偏導數 $d\theta_i$。
在所有分量 $i$ 計算完畢後，你會得到一個與 $d\theta$ 維度相同的近似梯度向量 $d\theta_{\text{approx}}$。

二、 誤差評估指標 (Evaluation Metric)

為了量化評估 $d\theta_{\text{approx}}$ 與 $d\theta$ 是否足夠接近，我們會計算它們的相對距離（Relative Difference）：

$$\text{相對距離} = \frac{|d\theta_{\text{approx}} - d\theta|_2}{|d\theta_{\text{approx}}|_2 + |d\theta|_2}$$

分子：計算兩個向量差值的歐幾里得範數（$\mathcal{L}_2$ Norm / 距離，即差值平方和再開根號）。
分母：除以兩個向量的歐幾里得長度之和進行歸一化（Normalization），以避免向量數值極大或極小時影響標準評估。
三、 評估標準與除錯指引（以 $\epsilon = 10^{-7}$ 為例）

計算出相對距離後，可依據以下標準評估實作的正確性：

相對距離	評估結果	建議處理方式
$\le 10^{-7}$	極佳	導數近似值極可能完全正確，可以非常有信心。
$\approx 10^{-5}$	可能可接受	需保持謹慎。建議仔細檢查向量中的每個分量，確認沒有任何單一分量的誤差過大。
$\ge 10^{-3}$	極令人擔憂	程式碼中幾乎確定存在 Bug。
偵錯建議

如果相對距離大於 $10^{-3}$，你應該逐一對比 $d\theta_{\text{approx}}[i]$ 與 $d\theta_i$ 的個別分量。尋找是哪一個特定的索引 $i$ 出現了巨大的數值差異，這能幫助你精確定位是哪一個導數計算（例如某一層的 $W$ 或 $b$）實作錯誤，進而快速修復。重複「除錯 $\rightarrow$ 重新檢驗」的過程，直到誤差值降到安全範圍。

四、 關鍵實作建議與注意事項

不要在訓練期間運行梯度檢驗： 計算近似梯度 $d\theta_{\text{approx}}$ 的過程非常緩慢（需針對每個參數計算兩次前向傳播）。因此，梯度檢驗僅應在偵錯（Debugging）時使用。在實際訓練模型時，請務必關閉梯度檢驗，並只使用反向傳播（Backprop）來計算導數。
檢驗失敗時，分析個別元件以定位 Bug： 如果檢驗結果顯示 $d\theta_{\text{approx}}$ 與反向傳播計算出的 $d\theta$ 差異極大，可以進一步觀察個別參數元件。例如，若發現只有與偏置項（Parameter $b$）相對應的某些層出現極大偏差，而權重（$W$）的元件差異很小，那麼 Bug 很可能就藏在 $db$ 的計算邏輯中。
務必納入正則化項（Regularization Term）： 如果代價函數 $J$ 包含了正則化（例如 $L_2$ 正則化/Frobenius 範數），那麼在計算梯度 $d\theta$ 時，也必須將正則化項的導數納入計算（如 $dJ/dW = \text{Backprop gradient} + \frac{\lambda}{m}W$）。
梯度檢驗無法與 Dropout 同時運作： 由於 Dropout 在每次迭代中會隨機關閉不同的節點，這使得代價函數 $J$ 沒有一個簡單穩定的公式可以直接計算。實作建議：先關閉 Dropout（例如將 keep_prob 設為 1.0），在此狀態下運行梯度檢驗以確保基礎演算法的正確性，確認無誤後再重新開啟 Dropout。
在「隨機初始化」與「訓練一段時間後」分別進行檢驗： 有些反向傳播的實作在參數 $W$ 和 $b$ 接近於 0（隨機初始化時）是正確的，但當參數隨著訓練逐漸變大時，計算誤差可能就會變得明顯。因此，建議在隨機初始化時進行一次梯度檢驗，接著讓模型訓練一段時間（使參數遠離 0），然後再次運行梯度檢驗以確保模型的正確性。

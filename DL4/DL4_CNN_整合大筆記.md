# DL4 卷積神經網路整合大筆記

本筆記整合 DL4 資料夾中的 CNN、卷積體積、單層 CNN、padding、stride、pooling、CNN 優勢與架構範例。符號以 channels-last 的單一樣本表示為 `H x W x C`；若使用 batch，會另外標示 batch 維度。

## 1. CNN 解決了什麼問題

影像同時具有大量像素與空間結構。若把影像直接攤平後接全連接層，參數量會快速爆炸：

- `64 x 64 x 3 = 12,288` 個輸入值。
- `1000 x 1000 x 3 = 3,000,000` 個輸入值；若下一層有 1,000 個單元，光權重就有 `3,000,000 x 1,000 = 3,000,000,000` 個。

CNN 利用兩個先驗假設降低參數量並保留空間結構：

1. **局部連接 / 局部感受野（local receptive field）**：一個輸出位置只看輸入的一小塊區域。
2. **參數共享（parameter sharing）**：同一個 filter 在所有空間位置重複使用，因此同一種特徵可以在不同位置被偵測。

CNN 的計算量仍會隨影像大小增加；「參數量與輸入 H/W 無關」不等於「計算量與記憶體無關」。若最後接全連接層，輸入尺寸仍可能受限，因此現代網路常使用 global average pooling 或全卷積設計。

## 2. 電腦視覺任務與特徵階層

- **影像分類**：輸出整張影像的類別，例如是否有貓。
- **目標檢測**：找出物體類別與位置，通常輸出多個 bounding boxes。
- **神經風格遷移**：保留內容結構並改變視覺風格。

CNN 常形成階層式特徵：早期層學習邊緣、方向與簡單紋理；中間層組合成局部部位；後期層整合成物體或類別相關表示。這是常見的學習結果，不是每個模型都必然嚴格遵循的固定分工。

## 3. 二維與三維體積卷積

灰階影像可表示為 `H x W x 1`，RGB 影像可表示為 `H x W x 3`。當輸入有多個通道時，filter 必須涵蓋全部輸入通道：

- 輸入：`H x W x C_in`
- 單一 filter：`f_H x f_W x C_in`
- 一個 filter 在每個位置對所有空間位置與通道做逐元素乘法後求和，再加一個 bias，產生一個 scalar。
- 使用 `C_out` 個 filters，就會得到 `H_out x W_out x C_out`。

例如 `6 x 6 x 3` 輸入搭配 `3 x 3 x 3` filter、stride 1、padding 0，單一 filter 產生 `4 x 4 x 1`；兩個 filters 堆疊後為 `4 x 4 x 2`。輸出通道數不是輸入通道數，而是 filters 的數量。

filter 可以只對某通道敏感，也可以跨通道組合資訊。傳統電腦視覺會手工設計 Sobel、Scharr 等 filter；CNN 則由反向傳播學習權重，可能學到任意方向與更複雜的特徵。

### 邊緣偵測直覺

常見的垂直邊緣 filter 例如：

$$
\begin{bmatrix}
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1
\end{bmatrix}
$$

它會比較左右兩側的亮度。正負號代表過渡方向；若只關心邊緣強度，可觀察絕對值或使用後續非線性，但實際 CNN 會讓模型自行學習適合任務的表示。

## 4. 卷積層的前向傳播

對第 `l` 層，每個 filter 先在輸入 `a^[l-1]` 上滑動並做點積，再加上該 filter 對應的 bias：

$$z^{[l]} = \operatorname{conv}(a^{[l-1]}, W^{[l]}) + b^{[l]}$$

接著套用非線性函數，例如 ReLU：

$$a^{[l]} = g(z^{[l]})$$

每個 filter 產生一張 2D feature map；所有 feature maps 沿通道維度堆疊，形成輸出 activation volume。

- 輸入單一樣本：`n_H^[l-1] x n_W^[l-1] x n_C^[l-1]`
- filter：`f_H x f_W x n_C^[l-1]`
- filters 數量：`n_C^[l]`
- 輸出：`n_H^[l] x n_W^[l] x n_C^[l]`

偏差可寫成每個輸出通道一個 scalar；實作中常 reshape 成 `1 x 1 x 1 x n_C`，再利用 broadcasting 加到 batch 上。

## 5. 維度公式、padding 與 stride

若高度方向輸入大小為 `n`、filter 大小為 `f`、每側 padding 為 `p`、stride 為 `s`：

$$
 n_{out}=\left\lfloor\frac{n+2p-f}{s}+1\right\rfloor
$$

寬度方向同理；高度與寬度可以使用不同參數。下取整代表只保留 filter 完全位於 padding 後輸入範圍內的位置。若框架採用不同的 `ceil` 或 `same` 規則，邊界行為要以該框架定義為準。

### 輸出尺寸公式的推導

以一個空間方向為例。padding 後的輸入長度是 `n+2p`，filter 長度是 `f`，所以 filter 最左端的合法起點不能超過 `n+2p-f`。若每次移動 `s` 格，合法起點為：

$$
0,s,2s,\ldots,ks\quad\text{且}\quad ks\le n+2p-f
$$

因此最大的 `k` 為：

$$
k=\left\lfloor\frac{n+2p-f}{s}\right\rfloor
$$

起點從 `0` 算起，所以總位置數是 `k+1`：

$$
n_{out}=\left\lfloor\frac{n+2p-f}{s}\right\rfloor+1
$$

這也解釋了為什麼不能只計算 `(n+2p-f)/s`，以及為什麼不整除時要取 floor。

### Padding

無 padding 稱為 **valid**：stride 1 時輸出為 `n-f+1`，空間尺寸會縮小，邊界像素也比中心像素少被涵蓋。

padding 通常補 0，也可以使用反射、複製或其他策略。stride 1、奇數 filter 且希望輸出與輸入相同時：

$$p=\frac{f-1}{2}$$

因此 `f=3` 用 `p=1`，`f=5` 用 `p=2`。這個簡式不是所有 same convolution 都適用：stride 大於 1 時，same 通常把目標輸出定為 `ceil(n/s)`，所需 padding 可能依輸入大小而定，也可能左右不對稱；偶數 filter 也常需要非對稱 padding。奇數 filter 之所以常見，是因為有明確中心且 stride 1 時容易對稱 padding。

### Strided convolution

stride 是 filter 每次移動的格數。`s=2` 是每次移動兩格，不是「跳兩格」；口語上可說跳過中間一格。stride 增大會減少輸出空間尺寸與運算量，但也會丟失較細的空間資訊。

## 6. 卷積與互相關

嚴格的數學卷積在逐元素相乘前會將 kernel 在水平與垂直方向翻轉；深度學習框架通常實作的是不翻轉 kernel 的**互相關（cross-correlation）**，但慣例上仍稱為 convolution。因為 filter 是從資料學出的，固定翻轉可以被吸收到學到的權重中，所以模型表達能力不受影響。嚴格卷積的結合律是信號處理上的性質，不是深度學習省略翻轉的主要必要條件。

嚴格卷積的結合律可寫成 `(A * B) * C = A * (B * C)`；互相關一般不具有相同的結合律保證。這個差異影響數學定義與某些信號處理推導，但不改變 CNN 透過可學習 filter 建立特徵的核心方式。

## 7. Pooling layers

Pooling 是固定運算，沒有可由梯度下降學習的權重。它通常對每個通道獨立處理，因此輸出通道數等於輸入通道數：

- **Max pooling**：區域內取最大值，保留最強 activation。
- **Average pooling**：區域內取平均值，保留平均訊號；global average pooling 可把 `7 x 7 x 1000` 沿空間平均成 `1 x 1 x 1000`。

其尺寸公式與卷積相同：

$$
H_{out}=\left\lfloor\frac{H+2p-f}{s}+1\right\rfloor,\qquad
W_{out}=\left\lfloor\frac{W+2p-f}{s}+1\right\rfloor
$$

pooling 常用 `f=2, s=2, p=0`，大約把 H/W 減半。padding 在 pooling 中較少見，但並非不可能。pooling 可降低後續計算並增加對小幅位置變化的容忍度，也可能造成資訊遺失；它不是保證防止 overfitting 的方法。max pooling 的「區域內只要有強特徵就保留」是有用直覺，但不應視為已完全證實的唯一根本原因。

Pooling 的尺寸公式沿用「合法起點數量」的推導；差別在於 pooling 沒有跨 channel 的 filter。每個 channel 各自產生一張 pooled map，因此 H/W 會變化，但 `n_C` 保持不變。Max pooling 的反向梯度通常只回到該區域中取得最大值的位置；average pooling 則會依平均運算規則分配梯度。

## 8. 參數量與計算量

卷積層參數量為：

$$
(f_H f_W C_{in}+1)C_{out}
$$

其中 `+1` 是每個 filter 的 bias。參數量與輸入 H/W 無關，但每次 forward 的乘加量大致會隨輸出空間大小增加，不能把「參數少」直接等同於「計算量低」。

### 參數量公式的推導

單一 filter 的空間大小是 `f_H x f_W`，每個空間位置都要看全部 `C_in` 個輸入通道，因此單一 filter 有 `f_H x f_W x C_in` 個 weights。每個 filter 再加 1 個 bias；若共有 `C_out` 個 filters：

$$
\begin{aligned}
\text{parameters per filter}&=f_Hf_WC_{in}+1\\
\text{parameters per layer}&=(f_Hf_WC_{in}+1)C_{out}
\end{aligned}
$$

輸入的 H/W 不出現在這個式子中，因為同一組 filter weights 會在所有合法位置共享。

若只估算乘加運算（MACs），每一個 output value 需要 `f_H x f_W x C_in` 次乘法與加總；整層大致為：

$$
H_{out}W_{out}C_{out}(f_Hf_WC_{in})
$$

bias、activation function 與記憶體搬移通常另行計算，所以 MACs 不等於完整執行時間。

以 `32 x 32 x 3` 輸入、6 個 `5 x 5 x 3` filters、valid、stride 1 為例：

- 輸出：`28 x 28 x 6`，共 4,704 個 activation。
- 卷積參數：`(5 x 5 x 3 + 1) x 6 = 456`。
- 若錯把 RGB filter 當成 `5 x 5`，會得到 156，這是不完整的計算。
- 若把輸入與輸出完全以全連接方式相連，權重為 `3072 x 4704 = 14,450,688`，約 1,445 萬，不是精確的 1,400 萬；若連 bias 也計算，還要再加上 4,704。

卷積的另一個優勢是**稀疏連接**：每個輸出只連到局部輸入，而非整張影像。全連接層的參數是否佔模型大多數取決於架構，不能當作普遍固定比例；而在現代 CNN 中，計算量也不必然由卷積層或全連接層單獨主導。

## 9. 平移：等變性與不變性

卷積共享同一 filter，使特徵出現在不同位置時仍能被相同 detector 偵測。這首先帶來的是**平移等變性（translation equivariance）**：輸入平移，feature map 通常也會相應平移（邊界、stride、padding 會造成例外）。

整個分類器對平移完全不變並不會自動成立。pooling、stride、global pooling、資料增強與模型訓練共同增加對小幅位移的容忍度，但過大的位移仍可能改變預測。

## 10. 維度順序與實作慣例

- channels-last：`[batch, height, width, channels]`，TensorFlow/Keras 常見預設。
- channels-first：`[batch, channels, height, width]`，PyTorch 常見格式，Caffe 也常見。

框架可提供其他設定；重點是 filter shape、activation shape 與所有操作前後一致。

## 11. 兩個架構範例

### 範例 A：三層卷積

輸入 `39 x 39 x 3`，每層 valid：

| 層 | filter / stride / padding | filters | 輸出 |
|---|---|---:|---|
| Conv 1 | `3 x 3 / 1 / 0` | 10 | `37 x 37 x 10` |
| Conv 2 | `5 x 5 / 2 / 0` | 20 | `17 x 17 x 20` |
| Conv 3 | `5 x 5 / 2 / 0` | 40 | `7 x 7 x 40` |

最後 flatten：`7 x 7 x 40 = 1,960` 個值，再接二分類 sigmoid 或 K 類 softmax。卷積層、pooling、FC 都可組合，但「一定要 flatten 後接 FC」不是唯一設計。

### 範例 B：LeNet 風格分類器

若輸入是 RGB，應寫成 `32 x 32 x 3`；原始 MNIST 則是灰階，通常是 `28 x 28 x 1`，LeNet-5 經典示例常將輸入補成 `32 x 32 x 1`，不能把 `32 x 32 x 3` 同時稱為原始 MNIST 的輸入。

`32 x 32 x 3` 經 `5 x 5` valid、stride 1、6 filters 後為 `28 x 28 x 6`；再經 `2 x 2`、stride 2 pooling 為 `14 x 14 x 6`。第二個 `5 x 5` valid 卷積、16 filters 得 `10 x 10 x 16`，再 pooling 得 `5 x 5 x 16`，flatten 後為 400。若 FC 依序為 120、84，最後用 10 類 softmax：

- `W_FC3` 可用「輸出 x 輸入」表示為 `120 x 400`。
- `W_FC4` 為 `84 x 120`。
- 輸出層權重為 `10 x 84`。
- pooling 沒有 trainable parameters；FC 與卷積層有 weights/biases。

「Conv + Pool 合稱一個 stage」是方便描述的架構分組，不是所有論文計算 layer 數的標準。通常 layer count 會依作者定義，需明確說明。

## 12. CNN 的訓練流程

1. 準備影像 `X` 與標籤 `Y`。
2. 初始化卷積與全連接層的 weights、biases；實務上會使用適合 activation 的初始化方法。
3. forward propagation 得到 logits 或預測值。
4. 以二元交叉熵、softmax cross-entropy 等 loss 計算成本 `J`。
5. 反向傳播計算所有參數梯度。
6. 用 SGD、Momentum、RMSProp、Adam 等 optimizer 更新參數。
7. 在 validation/dev set 調整 filter 大小、channels、stride、padding、pooling、learning rate、正則化等超參數，最後以 test set 做一次客觀評估。

## 13. 常見錯誤速查

- RGB filter 不能只算 `f x f`，必須算 `f x f x C_in`。
- `p=(f-1)/2` 只直接適用於 stride 1、奇數 filter 的對稱 same padding。
- stride 是移動幾格，不是跳過幾格。
- pooling 維持通道數；卷積輸出通道數由 filter 數量決定。
- CNN 通常先有平移等變性，不能宣稱自動產生完全平移不變性。
- 卷積參數少不代表卷積計算量一定比 FC 少。
- activation size 沒有「必須每層下降」的硬規則；卷積可能增加通道數，暫時增加 activation 數量。
- layer、stage、block 是不同層級的命名，合併描述時要先說明定義。

## 14. 從灰階影像到多通道影像

早期介紹卷積時，常先使用灰階影像，因為灰階影像只有一個 channel，可以把它畫成單一的二維矩陣。例如 `6 x 6` 灰階影像可寫成 `6 x 6 x 1`。

RGB 影像則有紅、綠、藍三個 channels，因此形狀是 `6 x 6 x 3`。這個第三維不是神經網路的層數，而是同一張影像的不同資料通道。

當輸入變成 `6 x 6 x 3` 時，`3 x 3` filter 不能只是一個 `3 x 3` 矩陣；它必須是 `3 x 3 x 3` 的 volume。filter 在三個 channel 都有權重，並在一次運算中將三個 channel 的結果全部加總。

對一個位置而言，三個 channel 各有 9 個值，所以總共有 27 個 input values 與 27 個 filter weights 參與點積。加總後只產生一個數值，而不是產生三個數值。

這也是為什麼一個 `3 x 3 x 3` filter 只會產生一張 2D feature map。若要得到多個 feature maps，就要使用多個 filters。

## 15. 多個 filters 如何形成輸出 volume

假設輸入是 `6 x 6 x 3`，每個 filter 是 `3 x 3 x 3`，使用 valid convolution 與 stride 1：

1. Filter 1 在輸入上滑動，得到 `4 x 4` 的 feature map。
2. Filter 2 用另一組 weights 在相同輸入上滑動，也得到另一張 `4 x 4` feature map。
3. 將兩張 map 沿第三維堆疊，得到 `4 x 4 x 2`。

因此 output channel 數量等於 filters 的數量。它不等於輸入 channel 數量，也不是由 filter 的空間尺寸決定。

若第一個 filter 的紅色 channel 使用垂直邊緣矩陣，而綠色與藍色 channel 的 weights 全為 0，這個 filter 主要偵測紅色通道的垂直變化。

若三個 channel 都使用相同的垂直邊緣矩陣，filter 會對不同顏色共同存在的垂直變化產生反應。實際 CNN 不必手動指定這些 weights，而是由訓練資料學習。

## 16. 卷積實際做的是什麼

在每一個空間位置，卷積層會執行以下程序：

1. 取出輸入的一個局部區域，大小由 filter 與 channels 決定。
2. 將 filter 與該局部區域做 element-wise product。
3. 將所有乘積相加，形成一個 scalar。
4. 加上該 filter 對應的 bias。
5. 將結果送入 activation function，例如 ReLU。
6. 將此 filter 在所有允許的位置重複使用。

對每一個 filter 重複完整流程，最後把所有 feature maps 疊成輸出 volume。這個流程就是卷積層對一般 fully connected layer 的空間化版本。

一般神經網路寫成：

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]},\qquad a^{[l]}=g(z^{[l]})
$$

卷積層則以 filter 取代一般矩陣 `W`，以局部滑動點積取代整體矩陣乘法。每個 output channel 有一個 filter，也有一個 bias。

## 17. 邊緣 filter 的方向與數值

垂直邊緣 filter 常使用左右相減的形式：

$$
\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1
\end{bmatrix}
$$

左側亮、右側暗時，輸出可能是正值；左側暗、右側亮時，輸出可能是負值。正負號表示亮暗轉換方向，絕對值則表示邊緣強度的直覺。

水平邊緣可以使用上下相減的形式：

$$
\begin{bmatrix}
1&1&1\\
0&0&0\\
-1&-1&-1
\end{bmatrix}
$$

將垂直 filter 旋轉 90 度，也能取得水平 filter。不同的權重配置會讓 filter 偏好不同方向、不同平滑程度與不同強度的影像變化。

## 18. Sobel 與 Scharr 的意義

Sobel filter 在中心位置使用較大的權重：

$$
S_{vertical}=\begin{bmatrix}
1&0&-1\\
2&0&-2\\
1&0&-1
\end{bmatrix}
$$

Scharr filter 使用更大的中心權重：

$$
C_{vertical}=\begin{bmatrix}
3&0&-3\\
10&0&-10\\
3&0&-3
\end{bmatrix}
$$

這些是傳統 computer vision 中由人設計的固定 filter。深度 CNN 的 filter weights 則是參數，會根據 loss 的梯度更新，因此不只會學到水平與垂直邊緣，也可能學到斜向邊緣、紋理與資料集特有的複雜模式。

## 19. 單層 CNN 的參數計算

假設輸入有 3 個 channels，本層使用 10 個 `3 x 3 x 3` filters：

- 單一 filter 的 weights：`3 x 3 x 3 = 27`。
- 單一 filter 的 bias：1 個 scalar。
- 單一 filter 總參數：`27 + 1 = 28`。
- 全層總參數：`28 x 10 = 280`。

這個 280 與輸入影像的 H/W 無關。輸入是 `6 x 6 x 3`、`1000 x 1000 x 3` 或 `5000 x 5000 x 3`，只要 filter size、輸入 channels 與 filters 數量相同，參數量都一樣。

但是輸入越大，需要滑動與計算的位置越多，所以 computation、activation memory 與 inference time 並不固定。參數量固定不代表整個模型的資源需求固定。

## 20. 局部連接與參數共享的差別

**局部連接**表示一個 output unit 只連到 input 的局部 grid。例如 `3 x 3` filter 的某個 output，只依賴對應的 `3 x 3` 區域，不會直接依賴遠處所有 pixels。

**參數共享**表示同一個 filter 的 weights 在不同空間位置重複使用。左上角使用的 edge detector 與右下角使用的是同一組 weights，不需要為每個位置重新學習一組 detector。

兩者共同使 CNN 比直接 FC 更有效率，也讓同一類局部特徵可以在不同位置被偵測。它們可以用於低階邊緣，也可以用於較高階的局部部位 detector。

## 21. 為什麼 CNN 適合影像

影像的鄰近 pixels 通常具有比任意遠距 pixels 更強的局部關係。卷積的 local receptive field 保留了這個結構，不必把每個 output 連到整張影像。

同一個物體特徵可能出現在影像的不同位置。parameter sharing 讓模型不需要知道特徵一定位於左上角或中央，便能在整張影像搜尋它。

隨網路加深，早期 layer 的 edge 或 texture 可以被後續 layer 組合成眼睛、輪廓、車輪、臉部或完整物體。這種 hierarchical feature extraction 是 CNN 的重要能力。

## 22. 維度計算的逐步規則

對單一空間方向，輸入大小為 `n`、filter 大小為 `f`、padding 每側為 `p`、stride 為 `s`：

$$
n_{out}=\left\lfloor\frac{n+2p-f}{s}+1\right\rfloor
$$

高度與寬度分別計算。如果高度與寬度使用相同的參數，才可以簡寫成相同的數字；一般情況下兩個方向可以有不同尺寸、padding 或 stride。

下取整表示最後一個 filter 必須完整落在 padding 後的輸入範圍內。若剩餘空間不足以放入完整 filter，該位置不會產生 output。

## 23. `6 x 6`、`3 x 3` 的尺寸例子

沒有 padding、stride 1 時：

$$
(6-3+1)\times(6-3+1)=4\times4
$$

輸入的第一個 `3 x 3` 區域位於左上角。filter 先向右移動，填滿第一列 output，再回到左側向下移動一格，繼續產生下一列。

如果輸入是 RGB，這個 `3 x 3` 區域實際上是 `3 x 3 x 3`，所以每個 output scalar 都是跨三個 channels 的總和。

## 24. Padding 的兩個主要理由

第一個理由是避免空間尺寸在每層都縮小。若每一層都使用 valid convolution，深度增加後 H/W 會很快變小，最後可能沒有足夠空間繼續套用 filter。

第二個理由是保留邊界資訊。沒有 padding 時，中心 pixel 會在許多重疊區域中被看見，但角落 pixel 可能只被看見一次，因此邊界資訊在輸出中的影響相對較小。

padding 讓 filter 能夠在靠近邊界的位置運算。補 0 是常見選擇，但不是唯一選擇；也可以使用 reflection、replication 等策略，實際效果要看任務與框架設定。

## 25. valid 與 same 的精確條件

**Valid** 表示 `p=0`。stride 1 時輸出為 `n-f+1`。

**Same** 的目的通常是讓輸出空間尺寸與輸入相同，但這句話必須連同 stride 一起解讀。當 stride 1、filter 是奇數且使用對稱 padding 時：

$$
n+2p-f+1=n\quad\Longrightarrow\quad p=\frac{f-1}{2}
$$

所以 `3 x 3` filter 需要 `p=1`，`5 x 5` filter 需要 `p=2`。

當 stride 大於 1，許多 framework 把 same output 定義為 `ceil(n/s)`，此時總 padding 可能依輸入大小而不同，也可能必須左右不對稱。偶數 filter 也常無法用相同的左右 padding 達成完全對稱。

## 26. Stride 的位置例子

`7 x 7` 輸入、`3 x 3` filter、stride 2、padding 0 時，filter 的起始位置可為 0、2、4，因此每個方向有三個位置：

$$
\left\lfloor\frac{7-3}{2}\right\rfloor+1=3
$$

輸出為 `3 x 3`。stride 2 代表起點增加兩格，也就是跳過中間的一格；它不是跳過兩格再前進。

stride 變大通常會降低輸出尺寸與部分運算量，但也會降低取樣密度，可能漏掉細小或精確位置資訊。

## 27. Convolution 與 cross-correlation 的用語

嚴格數學 convolution 在逐元素相乘前，會先將 kernel 做水平與垂直翻轉。深度學習常用的運算通常不翻轉 kernel，因此嚴格說是 cross-correlation。

神經網路文獻與框架仍普遍稱其為 convolution。因為 filter 是由資料學習的，若固定翻轉一次，模型可以學到相應翻轉後的 weights，因此表達能力不會因此改變。

## 28. Pooling 的完整尺寸與通道規則

Pooling 的 filter 只在 H/W 方向移動，通常對每個 channel 分開處理。若輸入是 `n_H x n_W x n_C`，輸出為：

$$
\left\lfloor\frac{n_H+2p-f}{s}+1\right\rfloor
	imes
\left\lfloor\frac{n_W+2p-f}{s}+1\right\rfloor
	imes n_C
$$

輸出 channel 數保持為 `n_C`。這點和 convolution 不同：convolution 的 output channels 是 filters 的數量。

Max pooling 在每個區域取最大值；average pooling 在每個區域取平均值。兩者都沒有 learnable weights 或 bias，但 pooling 運算本身仍在 forward graph 中，梯度可以依其規則傳回前面 layer。

## 29. Pooling 的使用直覺與限制

若 feature map 的值表示某個 feature detector 的 activation，較大的值通常可解讀為該 feature 在該位置反應較強。Max pooling 會保留區域內最強反應，因此對小幅位置移動比較有容忍度。

不過「這就是 max pooling 有效的真正原因」只是一個常用直覺，不能當成唯一已完全證實的底層解釋。max pooling 被廣泛使用，主要是大量實務結果支持它。

Average pooling 通常較少使用，但在網路後段可用來折疊空間維度。例如 `7 x 7 x 1000` 做 global average pooling 後成為 `1 x 1 x 1000`，再視需要接分類器。

`f=2, s=2, p=0` 是常見組合，通常使 H/W 約減半；padding 在 pooling 中比較少見，但特殊架構仍可能使用。

## 30. 邊緣偵測的完整例子

以下用一張簡化的 `6 x 6` 灰階影像說明垂直邊緣。左半部亮度為 10，右半部亮度為 0：

$$
X=\begin{bmatrix}
10&10&10&0&0&0\\
10&10&10&0&0&0\\
10&10&10&0&0&0\\
10&10&10&0&0&0\\
10&10&10&0&0&0\\
10&10&10&0&0&0
\end{bmatrix}
$$

搭配垂直 filter：

$$
K_v=\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1
\end{bmatrix}
$$

在左上角的 `3 x 3` 區域，左右兩側亮度相同，結果是 0。filter 向右移動後，若覆蓋兩欄亮區與一欄暗區，結果為 `10+10+10 = 30`。完整 valid 輸出為：

$$
Z=\begin{bmatrix}
0&30&30&0\\
0&30&30&0\\
0&30&30&0\\
0&30&30&0
\end{bmatrix}
$$

如果輸入是左暗右亮，對應結果會是負值；正負號表示邊緣方向，而不是「有無邊緣」的唯一表示。若只想檢查強度，可使用絕對值。

水平邊緣 filter 的一個例子是：

$$
K_h=\begin{bmatrix}
1&1&1\\
0&0&0\\
-1&-1&-1
\end{bmatrix}
$$

滑動時若同時覆蓋正邊緣和負邊緣，兩者可能抵銷，產生較小的過渡值。這是 filter 與影像局部區域相對位置造成的結果。

## 31. Sobel、Scharr 與可學習 filter

傳統電腦視覺常由專家手工設計 filter：

### Sobel 垂直邊緣 filter

$$
\begin{bmatrix}
1&0&-1\\
2&0&-2\\
1&0&-1
\end{bmatrix}
$$

中間列的權重較大，可讓邊緣估計具有不同的平滑與抗噪特性。

### Scharr 垂直邊緣 filter

$$
\begin{bmatrix}
3&0&-3\\
10&0&-10\\
3&0&-3
\end{bmatrix}
$$
$$
\\text{Sobel}_h=\\text{rotate90}(\\text{Sobel}_v),\qquad
\\text{Scharr}_h=\\text{rotate90}(\\text{Scharr}_v)
$$

CNN 不需要人工指定每個數值，而是把 filter 數值當作 weights，利用 loss、backpropagation 與 gradient descent 學習。模型可能學到類似 Sobel/Scharr 的 filter，也可能學到 45 度、70 度、73 度或更複雜的資料特徵。

## 32. 一個卷積層的完整符號表

第 `l` 層的超參數與張量如下：

| 符號 | 意義 | 形狀或數值 |
|---|---|---|
| `f^[l]` | filter 的空間尺寸 | `f x f` |
| `p^[l]` | 每側 padding 寬度 | 非負整數 |
| `s^[l]` | stride | 正整數 |
| `a^[l-1]` | 前一層單一樣本 activation | `n_H^[l-1] x n_W^[l-1] x n_C^[l-1]` |
| `W^[l]` | 本層所有 filters | `f x f x n_C^[l-1] x n_C^[l]` |
| `b^[l]` | 每個 filter 一個 bias | `n_C^[l]` 個 scalar |
| `a^[l]` | 本層單一樣本 activation | `n_H^[l] x n_W^[l] x n_C^[l]` |

其中 `n_C^[l]` 就是本層 filter 的數量。filter 的第三維一定等於上一層輸入通道數；filter 不是只包含空間尺寸。

若有 `m` 個樣本，channels-last 的 vectorized activation 可寫成：

$$A^{[l]}\in\mathbb{R}^{m\times n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}}$$

bias 常 reshape 成 `1 x 1 x 1 x n_C^[l]`，透過 broadcasting 加到所有樣本與空間位置。

## 33. Padding 與 same 的補充條件

對 `n x n` 影像與 `f x f` filter：

- valid、stride 1：輸出為 `(n-f+1) x (n-f+1)`。
- padding `p`、stride 1：輸出為 `(n+2p-f+1) x (n+2p-f+1)`。
- 一般 stride：每個空間方向為 `floor((n+2p-f)/s)+1`。

`p=(f-1)/2` 來自令 stride 1 的輸出尺寸等於輸入尺寸。它要求對稱 padding，最自然地適用於奇數 `f`。當 `s>1` 時，許多框架的 same 會把輸出定為 `ceil(n/s)`，所需的總 padding 要依 `n`、`f`、`s` 計算，左右可能不同。

不使用 padding 時，邊緣像素被涵蓋的次數少於中心像素；padding 讓 filter 可以在邊界附近運算，減少空間尺寸縮小與邊界資訊損失。padding 不一定只能補 0，也可以採用反射、複製等策略。

## 34. Stride 的數值例子

`7 x 7` 輸入、`3 x 3` filter、`p=0`、`s=2` 時：

$$
n_{out}=\left\lfloor\frac{7+0-3}{2}\right\rfloor+1=3
$$

因此輸出是 `3 x 3`。filter 的水平起點為 0、2、4；垂直方向也為 0、2、4。`s=2` 的意思是起點每次增加 2，不是跳過兩格。

## 35. Pooling 的詳細行為

Pooling 不跨通道混合資料。假設輸入為 `n_H x n_W x n_C`，每個 channel 都各自套用相同的空間 pooling 規則，輸出為：

$$
\left\lfloor\frac{n_H+2p-f}{s}+1\right\rfloor
	imes
\left\lfloor\frac{n_W+2p-f}{s}+1\right\rfloor
	imes n_C
$$

`f=2, s=2, p=0` 通常約使 H/W 減半，但若輸入尺寸不是適合的偶數，仍要依 floor 計算。pooling 只有超參數，沒有 weights 或 bias，因此不需要反向傳播去更新其參數；但梯度仍會透過 pooling 運算回傳到前面的卷積層。

## 36. 參數量與運算量的精確比較

卷積層參數量：

$$
	ext{parameters}=(f_Hf_WC_{in}+1)C_{out}
$$

以 `32 x 32 x 3` 輸入、6 個 `5 x 5 x 3` filters 為例：

- 每個 filter：`5 x 5 x 3 = 75` 個 weights，加 1 個 bias，共 76。
- 全層：`76 x 6 = 456` 個參數。
- 卷積輸出 valid、stride 1：`28 x 28 x 6`，共 4,704 個 activation。

如果輸入是灰階 `32 x 32 x 1`，才是每個 filter 25 個 weights、每層 `(25+1)x6=156` 個參數。

全連接比較若把 3,072 個輸入直接連到 4,704 個輸出：

$$3072\times4704=14,450,688$$

這裡只算 weights；若包含輸出 bias，還要加 4,704。卷積通常大幅減少參數，但其 feature map 仍可能帶來大量乘加運算，因此參數量與計算量必須分開分析。

## 37. CNN 架構設計與 activation

常見模式是：

$$
(\text{Conv}-\text{Pool})\times N+\text{FC}\times M+\text{Output}
$$

常見趨勢是越深入，H/W 逐漸下降而 channel 數增加，將空間資訊逐步轉為特徵資訊。但這不是硬性規則：卷積可能讓 activation 總數暫時上升，例如 `32x32x3=3072` 到 `28x28x6=4704`。若空間尺寸或通道數過早大幅下降，才可能造成資訊損失。

卷積與 pooling 可以視為一個 stage，但 layer count 的算法依論文定義而異。pooling 不含 trainable parameters，不代表它不算一個運算層。

## 38. 實作與訓練補充

實作可使用課程中的 `conv_forward`，或框架 API：

- TensorFlow：`tf.nn.conv2d`。
- Keras：`Conv2D`。

訓練一個 CNN 的一個 iteration 通常包含：輸入 batch `X,Y`、forward、計算 loss、backpropagation、optimizer 更新。二分類可使用 sigmoid 與 binary cross-entropy；K 類分類可使用 softmax 與 categorical/sparse cross-entropy。

卷積層與 FC 層的 weights/biases 都要更新；pooling 的固定規則不更新。learning rate、filters 數量、filter size、stride、padding、pooling 類型、正則化與網路深度是 hyperparameters，應利用 validation/dev set 選擇，最後使用 test set 評估。

## 39. 維度格式提醒

channels-last：

$$[\text{batch},\text{height},\text{width},\text{channels}]$$

常見於 TensorFlow/Keras。channels-first：

$$[\text{batch},\text{channels},\text{height},\text{width}]$$

常見於 PyTorch，也常見於 Caffe。框架可能允許切換 data format；只要輸入、filters、activation 與每一層的格式前後一致即可。

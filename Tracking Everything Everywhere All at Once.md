**Tracking Everything Everywhere All at Once**

![image-20231017162604354](https://github.com/CatherineYun/YUN/blob/main/image/image-20231017162604354.png)

1. Related Work

​	Optical flow：光流传统上被表述为一个优化问题，最近的进展使利用神经网络直接预测光流成为可能，提高了质量和效率。虽然光流方法允许在连续帧之间进行精确的运动估计，但它们不适合进行远程运动估计：	

2. OmniMotion representation

   OmniMotion 是一个新的测试时间优化方法，用于从视频序列中估计密集和长距离运动。；OmniMotion 将视频用准三维规范体表示，并通过局部和规范空间之间的双射（满足单射和满射）映射到每一帧的局部 volume。local-canonical bijection 被参数化为神经网络。并捕捉摄像机和场景的运动，不解开两者。可以被认为是从一个固定的，静态的照相机产生的local volume 的渲染

   ![image-20231017163013035](G:\GitHub\YUN\image\image-20231017163013035.png)

   ![image-20231019132228545](G:\GitHub\YUN\image\image-20231019132228545.png)

   **Canonical 3D volume** *G*作为观测场景的三维图集，在*G*上定义一个基于坐标的网络  *F*~θ~将每个规范坐标**u** *∈* *G*映射到一个密度*σ*和颜色*c*。存储在*G*中的密度是告诉曲面在规范空间中的位置。结合3D bijections，允许我们在多个帧上跟踪 surfaces，以及关于遮挡关系的原因；存储在*G*中的颜色允许在优化过程中计算photometric loss。

​	**3D bijections** 连续的双射映射 ${{\mathcal{T}}_i}$，将来自每个局部坐标系*L*~i~的三维点*x*~i~映射到规范的三维坐标系**u** =${{\mathcal{T}}_i}(\boldsymbol{x}_i)$，其中*i*是帧索引。可以从一个局部的三维坐标中映射一个frame （*L*~i~) to another (*L*~i~ ):

$$
\boldsymbol{x}_j={\mathcal{T}}^{-1}_j◦{\mathcal{T}}_{i}(\boldsymbol{x}_i)\tag 1
$$


为了允许能够捕捉真实世界运动的表达性地图，将双射参数化为可逆神经网络（INNs）。使用Real-NVP（Real-NVP组合被称为仿射耦合层的简单双射变换来构建双射映射。仿射耦合层将输入分割成两部分；第一部分保持不变，但用于参数化，应用于第二部分的仿射变换。）。修改这个架构，condition on a per-frame latent code $ψ_i$,然后所有的可逆映射$T_i$都由相同的可逆网络$M_θ$参数化，but with different latent codes: $T_i(·) = M_θ(·; \boldsymbol{ψ}_i)$.

 **Computing frame-to-frame motion** 通过在射线上的采样点将查询像素“提升”到3D，使用双射${{\mathcal{T}}_i}$和${{\mathcal{T}}_j}$“映射”这些3D points 到一个目标帧$j$，通过alpha compositing “渲染”这些来自不同样本的3D points，最后“project” 回2D以获得假定的对应关系。

​	假设相机运动包含了 local-canonical bijections  ${{\mathcal{T}}_i}$ 并且使用fixed, orthographic camera。在$\boldsymbol{p_i}$ (query pixel)处的射线定义为$ \boldsymbol{r_i}(z) =\boldsymbol{o_i}+z\boldsymbol{d}$，在射线上采集K个样本${\boldsymbol{x}^k_i}$，这相当于在$\boldsymbol{p_i}$ 上附加一组深度值${ \lbrace{z^k_i}\rbrace}^K_{k=1}$。将这些样本映射到规范空间，然后查询密度网络$F_θ$，来获得这些样本的密度和颜色。以第*k*个样本$x^k_i$为例，它的密度和颜色可以写成$(\sigma{_k},\boldsymbol{c}_k)=F_θ(M_θ(\boldsymbol{x}^k_i；\boldsymbol{ψ}_i))$ 。我们还可以沿着每个样本的射线映射到帧j中相应的三维位置$\boldsymbol{x}^k_j$(Eq. 1).

​	聚合所有样本的对应样本$\boldsymbol{x}^k_j$，以生成一个单一的对应$\hat{\boldsymbol{x}}_j$。这种聚合类似于NeRF中样本点的颜色的聚合方式：使用alpha合成，第k个样本的alpha值为$α_k = 1−exp(−σ_k)$。$\hat{x}_j$为：
$$
\hat{\boldsymbol{x}}_j =\sum_{k=1}^KT_k{\alpha}_k\boldsymbol{x}^k_j,\quad{where}\quad{T_k} =\prod_{l=1}^{k-1}(1-\alpha_l)
$$
用类似的过程合成$\boldsymbol{c}^k$，得到$\boldsymbol{p}_i$的图像空间颜色$\hat{\boldsymbol{C}}^i$。

3. Loss functions

$$
\mathcal{L}_{flo} =\sum_{f_{i→j}∈Ω_f}||\hat{f}_{i→j} − f_{i→j} ||_1 \tag 3
$$

$\hat{\boldsymbol{f}}_{i→j}=\hat{\boldsymbol{p}}_j-\boldsymbol{p}_i$($\hat{\boldsymbol{p}}_j$是使用固定正交相机模型对$\hat{\boldsymbol{x}}_j$进行投影，得到查询位置$\boldsymbol{p}_i$的预测的二维相应位置$\hat{\boldsymbol{p}}_j$);$Ω_f$ is the set of all the filtered pairwise flows
$$
\mathcal{L}_{pho}=\sum_{(i,\boldsymbol{p})∈Ω_p}||\hat{C}^i(\boldsymbol{p}) − C_i(\boldsymbol{p})||^2_2 \tag 4
$$
$Ω_p$是所有帧上的所有像素位置的集合。

为了保证$M_θ$估计的三维运动的时间平滑性，我们应用了一个正则化项来惩罚大的加速度。给定帧i中的采样三维位置$\boldsymbol{x}_i$，我们使用等式 1将其映射到帧i−1和帧i+1，分别产生3D点$\boldsymbol{x}_{i−1}$和$\boldsymbol{x}_{i+1}$，然后最小化3D加速度：
$$
\mathcal{L}_{reg} =\sum_{(i,\boldsymbol{x})∈Ω_x}
||\boldsymbol{x}_{i+1} + \boldsymbol{x}_{i−1} − 2\boldsymbol{x}_i
||_1 \tag 5
$$
$Ω_x$是所有帧的局部三维空间的并集

分别最小化公式3、4、5，最终loss function 为
$$
\mathcal{L} = \mathcal{L}_{flo} + λ_{pho}\mathcal{L}_{pho} + λ_{reg}\mathcal{L}_{reg}
$$
权重λ控制了每个项的相对重要性。

# T-SEA: Transfer-based Self-Ensemble Attack on Object Detection

![image-20231020194333132](image\image-20231020194333132.png)

![image-20231020194429177](image\image-20231020194429177.png)

1. Method

    **Problem Formulation**

   ​	只使用一个白盒探测器进行基于传输的黑盒攻击，以降低白盒和黑盒探测器的平均平均精度（mAP）。给定目标输入数据分布$D(\mathcal{X}，\mathcal{F})$，将单个预先训练过的检测器$f_w∈\mathcal{F}$作为白盒攻击模型，$x_{1，..，N}∈\mathcal{X}$作为输入图像，其中N为训练样本的数量。从对抗分布$\mathcal{T}$中制作一个通用的对抗补丁$τ$来破坏检测过程，

    **Overall Framework**

   ![image-20231020200657557](image\image-20231020200657557.png)
# 深度非负矩阵分解
> Chen, Wen-Sheng, Qianwen Zeng and Binbin Pan. “A survey of deep nonnegative matrix factorization.” Neurocomputing 491 (2022): 305-320.
> 
## 摘要
近年来,深度非负矩阵分解 (Deep NMF) 是特征提取的有效策略。`通过对NMF算法的递归分解矩阵,我们获得了分层神经网络结构,并探索数据的更可解释的表示。`本文主要关注一些关于 Deep NMF 的理论研究,其中系统地包括基本模型、优化方法、属性及其扩展和泛化。我们将深度 NMF 算法分为五类:深度 NMF、约束深度 NMF、广义深度 NMF、多视图深度矩阵分解 (MF)、深度神经网络 (DNN) 和 NMF 之间的关联。此外,我们研究了 Deep NMF 算法在某些人脸数据库上的聚类性能。然后综合分析了深度NMF方法的设计原则、主要步骤、关系、应用领域和演化。此外,还讨论了 Deep NMF 的一些开放性问题。
## 引言
在模式识别、信号处理、特征工程、神经网络和计算机视觉的研究中,需要处理大量的冗余数据。因此,需要有效的方法来减少数据的冗余。数据的降维可以减少冗余和计算成本。近年来提出了各种**降维方法**,如**主成分分析(PCA)**、**局部保持投影(LPP)** 和**线性判别分析(LDA)**。此外,最近提出了**非线性和局部特征提取技术**。然而,由于大量真实数据的非负性,上述方法有一个共同的局限性,即降维中元素的符号没有限制。出于这个原因,许多研究人员对非负特征提取的学习很感兴趣。`非负矩阵分解(NMF)`算法被广泛应用于特征表达。NMF 在非负约束下将非负矩阵分解为两个低秩矩阵 $\mathbf{W}$ 和 $\mathbf{H}$。$W$和$H$分别称为基矩阵和特征矩阵。目标函数是 $\mathbf{V}$ 和 $\mathbf{WH}$ 之间的欧几里得距离或 Kullback Leibler 散度。NMF 方法学习到的特征不仅具有很强的泛化能力,而且有明确的物理意义。

尽管 NMF 已应用于场景中,但它只考虑了数据的浅层信息。对于内容丰富的数据,单层结构无法从几个方向获得表示。为了解决这些问题,将浅层网络扩展到多层结构,提出了一种称为`多重 NMF 的算法`,该算法通过分解特征矩阵来构建多层神经网络。`它结合了NMF算法和网络前向传播来推导每一层的基本矩阵和特征矩阵的迭代公式。`此外,通过迭代地分解特征矩阵以构建层次结构模型,可以得到一种称为`多层 NMF 的算法`,并通过 NMF 方法获取每一层的两个矩阵。最近,深度学习 (DL) 引起了很多的关注。DL 将一个复杂的任务分解为几个部分,这些部分在整个深层结构中由一个部分解决。DL 建立了一个具有层次结构的深度神经网络,克服了多层结构优化的困难。一些研究人员提出了具有深度网络结构的非负矩阵分解算法,称为`深度非负矩阵分解(Deep NMF)`。深度 NMF 可以发现数据的潜在层次特征,从而利用更具辨别力的表示。

现有的深度NMF算法分为以下五类:
*   深度NMF(Basic DNMF)——只包含了非负这一个约束
*   加约束的深度NMF(Constrained DNMF)——深度NMF的基础上,加了一些额外的约束来作为正则项
    * Sparse DNMF
    * Orthogonal DNMF
    * Discriminant DNMF (判别式)
    * Deep NMF combining geometric structure information (结合几何结构信息的深度NMF)
*   广义深度NMF(Generalized DNMF)——延申了传统的分解模型
    * 深度半非负矩阵分解
    * 深度非光滑非负矩阵分解
*   多视角深度MF(Multi-View DNMF)——在多视角(multi-view)的数据中使用深度矩阵分解
*   可以体现DNN&NMF之间关系的模型(Deep Netural Network and NMF)
    * 深度自编码器与深度NMF结合
    * 结合DNN和NMF   

## 深度NMF算法
深度NMF算法基于多层网络结构。Deep NMF 模型集成了微调步骤来优化矩阵分解后的模型。也就是说,Deep NMF 可以被视为具有微调过程的多层 NMF。深度 NMF 算法与约束深度 NMF 和广义深度 NMF 有关系。在本节中,首先介绍 Deep NMF 的基本框架和目标函数。还提出了一些深度 NMF 优化方法。
### 深度NMF基本框架和目标函数
基本的NMF可以看作是一个单层学习过程,它从数据矩阵 $\mathbf{X}$ 同时学习基矩阵 $\mathbf{W}$ 和特征矩阵 $\mathbf{H}$。为了使用多层NMF开发数据集中特征层次结构的知识,在单层上获得的矩阵 $\mathbf{H}_1$ 可以分解为矩阵 $\mathbf{W}_2$ 和 $\mathbf{H}_2$。因此,我们将浅层 NMF 扩展为两层 NMF 结构。然后单层建立转化为多层结构。将矩阵 $\mathbf{X}$ 分解为 $L+1$ 个因子,用以下公式表示:

$$
\mathbf{X}=\mathbf{W}_1 \mathbf{W}_2 \cdots \mathbf{W}_L \mathbf{H}_L
\tag{eq1}
$$
其中 $\mathbf{W}_i \in \mathbf{R}^{\mathbf{M}_{\mathbf{i}-\mathbf{1}} \times \mathbf{M}_{\mathbf{i}}}, \mathbf{i}=\mathbf{1}, \mathbf{2}, \cdots, \mathbf{L}$ 。这个公式允许每一层的隐式表示,可以通过以下分解给出:
$$
\mathbf{H}_{\mathbf{1}} \approx \mathbf{W}_{\mathbf{2}} \cdots \mathbf{W}_{\mathbf{L}} \mathbf{H}_{\mathbf{L}}, \mathbf{H}_{\mathbf{2}} \approx \mathbf{W}_{\mathbf{3}} \cdots \mathbf{W}_{\mathbf{L}} \mathbf{H}_{\mathbf{L}}, \cdots, \mathbf{H}_{\mathbf{L}-\mathbf{1}} \approx \mathbf{W}_{\mathbf{L}} \mathbf{H}_{\mathbf{L}}
$$
这些隐藏的 $\mathbf{H 1} \cdots \mathbf{H}_{\mathbf{L}}$ 都满足非负约束。许多 Deep NMF 算法在分解矩阵后微调自己,以便它们能够减少模型的重建误差。因此,Basic Deep NMF 模型分为以下两个阶段:
1. 预训练阶段: 此阶段旨在通过单层 NMF 算法预训练每一层。具体来说,数据矩阵 $\mathbf{X}$ 被分解为第一层的 $\mathbf{W}_1$ 和 $\mathbf{H}_1$。然后矩阵 $\mathbf{W}_2$ 和 $\mathbf{H}_2$ 由 $\mathbf{H}_1$ 获得。重复该过程,直到达到最大层数 $L$。

2. 微调阶段:此步骤处理所提出模型的整个分层网络结构。我们根据以下目标函数微调前一阶段获得的所有矩阵:
$$
\mathbf{C}_{\text {deep }}=\frac{1}{2}\parallel X-W_1 W_2 \cdots W_L H_L\parallel_F^2
\tag{eq2}
$$
基矩阵 $\mathbf{W}$ 和特征矩阵 $\mathbf{H}$ 由 $\mathbf{W}=\mathbf{W}_1 \mathbf{W}_2 \cdots \mathbf{W}_{\mathbf{L}}$ 和 $\mathbf{H}=\mathbf{H}_{\mathbf{L}}$ 计算。值得注意的是,大多数 Deep NMF 算法都是基于上面提到的 Deep NMF 设计的,它分解特征矩阵而不是基矩阵。分层分解 $H$ 的 DNMF 算法可以推导出原始数据的多层特征表示。这种多层表示可能有利于监督学习。相比之下,一些 DNMF 算法迭代地分解 $W$。这种分解有很好的解释,即高级基由低级基组成。

## 基本深度NMF的经典优化方法

基本深度NMF算法的预训练阶段依赖于单层NMF方法。已经提出了许多浅层NMF优化方法。作为类似于 EM 算法的优化方法,NMF 的更新规则相对简单。乘法和GD算法是称为交替非负最小二乘(ANLS)的框架的特例。具有单独凸性的特点,将两个变量的优化问题转化为非负最小二乘(NLS)优化子问题。

受上述乘法更新规则的启发,将近似更新标准应用于微调阶段。我们将 $(eq2)$ 中给出的成本函数重写为
$$
\mathbf{C}_{\text {deep }}=\frac{1}{2}||X-\psi_{i-1} W_i \widetilde{H_i}||_F^2
\tag{eq3}
$$
其中 $\psi_{\mathbf{i}-\mathbf{1}}=\mathbf{W}_{\mathbf{1}} \mathbf{W}_{\mathbf{2}} \cdots \mathbf{W}_{\mathbf{i}-\mathbf{1}}, \widetilde{\mathbf{H}}_{\mathbf{i}}=\mathbf{W}_{\mathbf{i}-\mathbf{1}} \cdots \mathbf{W}_{\mathbf{L}} \mathbf{H}_{\mathbf{L}}, \mathbf{i}=\mathbf{1}, \mathbf{2}, \cdots, \mathbf{L}$ 其中 $\psi_{\mathbf{0}}$ 是单位矩阵,用负梯度更新的方式交替迭代更新 $\mathbf{W}_1, \mathbf{W}_2, \cdots, \mathbf{W}_L$ 和 $\mathbf{H}_L$ 首先,通过关于 $\mathbf{W}_{\mathbf{i}}$ 和 $\mathbf{H}_{\mathbf{i}}$ 微分 $(eq3)$ 来获得梯度:
$$
\nabla \mathbf{w}_{\mathbf{i}} \mathbf{C}_{\text {deep }}=-\psi_{\mathbf{i}-\mathbf{1}}^{\mathbf{T}} \mathbf{X} \widetilde{\mathbf{H}_{\mathbf{i}}} ^\mathbf{T}+\psi_{\mathbf{i}-\mathbf{1}}^{\mathbf{T}} \psi_{\mathbf{i}-\mathbf{1}} \mathbf{\mathbf { W } _ { \mathbf { i } }} \widetilde{\mathbf{H}_{\mathbf{i}}} \widetilde{\mathbf{H}}_{\mathbf{i}}^T
\tag{eq4}
$$
以及
$$
\nabla_{\mathbf{H}_{\mathbf{i}}} \mathbf{C}_{\text {deep }}=\widetilde{\psi}_{\mathbf{i}-\mathbf{1}}^{\mathbf{T}} \mathbf{X}+\widetilde{\psi}_{\mathbf{i}}^{\mathbf{T}} \widetilde{\psi_{\mathbf{i}}} \mathbf{H}_{\mathbf{i}}
\tag{eq5}
$$
其中 $\widetilde{\psi_{\mathbf{i}}}=\widetilde{\psi}_{\mathbf{i}-\mathbf{1}} \mathbf{W}_{\mathbf{i}}, \mathbf{i}=\mathbf{1}, \cdots, \mathbf{L}$
通过调整适当的步长,在微调阶段获得第 $i$ 层的更新规则如下:
$$
\mathbf{W}_{\mathbf{i}} \leftarrow W_i \otimes \frac{(\psi_{i-1}^{T} X \widetilde{H_i^{\top}})}{(\psi_{i-1}^{\top} \psi_{i-1} W_i \widetilde{H_i} \widetilde{H_i^{\top}})}
\tag{eq6}
$$
$$
\mathbf{H}_{\mathbf{i}} \leftarrow H_i \otimes \frac{(\widetilde{\psi_{i-1}^{\top}} X)}{(\widetilde{\psi_i^{\top} \widetilde{\psi_i} H_i)}}, i=1, \cdots, L
\tag{eq7}
$$
深度NMF的传统优化方法的计算复杂度相对较低。然而,它也有一些缺点,如陷入局部最优解和收敛速度慢。
## 约束深度NMF
在以前的研究中,基本NMF算法不能仅在非负约束下获得唯一解。因此,NMF算法通常在矩阵 $\mathbf{W}$ 和 $\mathbf{H}$ 上添加一些限制条件。通过扩展原始目标函数,各种约束的深度NMF可以统一表示如下:
$$
\mathbf{C}_{\text {deep }}=\frac{1}{2}\|X-W_1 W_2 \cdots W_L H_L\|_F^2+\sum_i \alpha_j J_1(W_i)+\sum_j \beta_j J_2(H_j)
\tag{eq8}
$$

其中 $\mathbf{J}_{\mathbf{1}}(\mathbf{W}_{\mathbf{i}})$ 和 $\mathbf{J}_{\mathbf{2}}(\mathbf{H}_{\mathbf{j}})$ 是惩罚项, $\alpha_{\mathbf{i}}$ 和 $\beta_{\mathbf{i}}$ 被称为正则化参数,用于平衡拟合优度和约束之间的关系。第2.3节表明,约束多层模型可以通过多种方式进行优化。由于 $\mathbf{J}_{\mathbf{1}}(\mathbf{W}_{\mathbf{i}})$ 和 $\mathbf{J}_{\mathbf{2}}(\mathbf{H}_{\mathbf{j}})$ 的多样性,约束深度NMF算法分为三种形式:
1. 稀疏深度NMF
2. 正交深度NMF
3. 半监督深度NMF
### 稀疏深度NMF
在模式识别、信号处理、文本聚类和计算机视觉等领域,研究人员总是希望得到对原始数据更稀疏的表达。稀疏约束在单层约束NMF问题中得到了广泛应用。这种约束条件有助于提高分解的唯一性。然而,如何施加稀疏性约束是一个值得关注的问题。这要看实际问题。预计矩阵分解的结果会更加稀疏。然后降低特征矩阵的数据冗余度,从而进一步提高分解效率。基于1范数的稀疏约束模型比基于2范数的稀疏约束模型更有效。此外。也可以在在矩阵分解上使用 $1/2$ 范数约束,这种技术有助于解决方案以及更好的无偏性。

鉴于稀疏的单层NMF,Deep NMF采取了类似的步骤。可以使用使用深度NMF进行具有稀疏性约束的高光谱解混合,并减少了高光谱混合像素的信息量。在预训练部分,将每个层的丰度矩阵(特征矩阵)应用于1-范数约束。此后,目标函数 $(eq8)$ 的正则化项是关于所有层特征矩阵的 1 范数的形式。微调阶段可以表示为 $\mathbf{F}=\frac{1}{2}\|\mathbf{X}-\mathbf{W}_1 \mathbf{W}_1 \cdots \mathbf{W}_{\mathbf{L}} \mathbf{H}_{\mathbf{L}}\|_{\mathbf{F}}^2+\sum_{\mathbf{i}} \alpha_{\mathbf{i}}\|\mathbf{H}_{\mathbf{i}}\|_{\mathbf{1}}$ 。在预训练和微调阶段,选择Nesterov的最优梯度法和GD法作为最优性标准。对美国地质调查局(USGS)光谱库的合成数据和在内华达州Cuprite收集的真实数据进行了几次实验。由于NMF模型的非凸性,他们提出了将1范数稀疏约束添加到基本深度NMF的方法。在每一层中,丰度矩阵将作为下一层的输入,稀疏性约束将应用于丰度矩阵。此外,在微调阶段,目标函数 $(eq8)$ 的正则化项被每个层上丰度矩阵的l1正则化项的和所代替。Nesterov的加速梯度算法、乘法和GD方法都作为模型的更新规则。实验结果表明,该算法优于其他解混方法。但该模型的收敛性和稳定性尚未得到证明。

然而,1范数的一个缺点是它不满足完全可加性。鉴于此,通过将 $1/2$ 范数稀疏性约束应用于深层NMF中各层的丰度矩阵,用于高光谱解混合。全变差(TV)正则化也被集成到该模型中,该模型可以充分利用高光谱图像的光谱和空间信息,并提高丰度图估计中的分段平滑度。目标函数的惩罚项由丰度矩阵和TV正则化器上的 $1/2$ 范数约束添加。目标函数为
$$
\mathbf{C}_{\text {deep }}=\frac{1}{2}\|X-\psi_{i-1} W_i \widetilde{H}_i\|_F^2+\alpha\|H_i\|_{1 / 2}
\tag{eq9}
$$
其中 $\psi_{\mathbf{i}-\mathbf{1}}=\mathbf{W}_{\mathbf{1}} \mathbf{W}_{\mathbf{2}} \cdots \mathbf{W}_{\mathbf{i}-\mathbf{1}}, \widetilde{\mathbf{H}_{\mathbf{i}}}=\mathbf{W}_{\mathbf{i}-\mathbf{1}} \cdots \mathbf{W}_{\mathbf{L}} \mathbf{H}_{\mathbf{L}}, \mathbf{i}=\mathbf{2}, \mathbf{3}, \cdots, \mathbf{L}$ 。实验结果表明,与其他高光谱解混合方法相比,该算法的性能显著提高。然而,没有提供模型的收敛性。

在分层NMF方法中,不仅要对特征矩阵应用稀疏性约束,还要满足基本矩阵的稀疏性要求。在基于深度NMF的高光谱解混合中,对端元矩阵(基本矩阵)和丰度矩阵使用了稀疏性约束。事实上,关于矩阵 $\mathbf{W}_{\mathbf{i}}$ 和 $\mathbf{H}_{\mathbf{i}}$ 的 $1/2$范数惩罚项被添加到每个层的目标函数中,乘法更新规则可用于计算这些矩阵。该算法通过建立多层结构,可以将观测矩阵分解为几个稀疏矩阵。每一层的分解可以表示为 $\mathbf{C}_{\text {deep }}=\frac{1}{2}\|X-W H\|_F^2+\alpha\|H\|_{1 / 2}+\beta\|W\|_{1 / 2}$ 。对比试验表明,该算法提高了高光谱解混合性能,并消除了陷入局部极小值的可能性。此外,在水下彩色图像分解中也采用了类似于的算法,该算法通过1范数和2范数的组合来测量稀疏性。此外,该方法获得了更好的视觉质量。稀疏深度NMF算法在高光谱分解和人脸图像聚类中起着关键作用。针对稀疏编码问题,强制基本矩阵为稀疏矩阵。构建了四个模型,分别称为SDNMF/L、SDNMF/R、SDNMF/RL1和SDNMF/RL2,以扩展稀疏深度NMF算法。在这四种算法中分别加入了对各层基本矩阵或特征矩阵的稀疏性约束。具体而言,SDNMF/L中的矩阵 $\mathbf{W}_{\mathbf{i}}$ 列和SDNMF/R中的矩阵 $\mathbf{W}_{\mathbf{i}}$ 列使用1范数约束,其可以表示为
$$
\begin{aligned}
\mathbf{F}_{\mathbf{S D N M F} / \mathbf{L}}= & \frac{1}{\mathbf{2}}\|\mathbf{X}-\mathbf{W}_{\mathbf{1}} \mathbf{W}_{\mathbf{2}} \cdots \mathbf{W}_{\mathbf{L}} \mathbf{H}_{\mathbf{L}}\|_{\mathbf{F}}^2 \\
& +\frac{1}{2} \sum_{\mathbf{i}} \alpha_{\mathbf{i}} \mathbf{t r}((\xi_{\mathbf{i}} \mathbf{W}_{\mathbf{i}}\right)^{\mathbf{T}}\left(\xi_i \mathbf{W}_{\mathbf{i}}))
\end{aligned}
\tag(eq10)
$$
以及
$$
\begin{aligned}
\mathbf{F}_{\text {SDNMF } / \mathbf{R}}= & \frac{1}{\mathbf{2}}\left\|\mathbf{X}-\mathbf{W}_{\mathbf{1}} \mathbf{W}_{\mathbf{2}} \cdots \mathbf{W}_{\mathbf{L}} \mathbf{H}_{\mathbf{L}}\right\|_{\mathbf{F}}^{\mathbf{2}} \\
& +\frac{1}{\mathbf{2}} \sum_{\mathbf{i}} \beta_{\mathbf{i}} \mathbf{t r}\left(\left(\xi_{\mathbf{i}} \mathbf{H}_{\mathbf{i}}\right)^{\mathbf{T}}\left(\xi_{\mathbf{i}} \mathbf{H}_{\mathbf{i}}\right)\right), \mathbf{i} \\
= & \mathbf{1}, \cdots, \mathbf{L}
\end{aligned}
$$


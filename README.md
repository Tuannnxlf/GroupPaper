# GroupPaper
2024年上半年组会论文汇总
- Contrastive Collaborative Filtering for Cold-Start Item Recommendation  [[pdf](https://arxiv.org/pdf/2302.02151.pdf)]  
  WWW 2023.2 [cold-start item recommedation]  
  通过内容特征和协同特征之间的对比学习，使得内容编码器学习到协同知识，为冷启动提供帮助
    
- Aligning Disstillation For Cold-start Item Recommendation [[pdf](https://dl.acm.org/doi/10.1145/3539618.3591732)]  
  SIGIR 2023.7 [cold-start item recommedation]  
  通过对齐冷启动的学生模型与热启动的教师模型，并使用教师模型和学生模型共同进行推荐  
- Prompt,Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners [[pdf](https://arxiv.org/pdf/2303.02151.pdf)]  
  CVPR 2023.3 [Few-Shot Action Recognition]  
  针对多个预训练模型知识的融合问题，结合了各种预训练范式的不同先验知识，以实现更好的小样本学习  
- Domain Aligned CLIP for Few-shot Classification [[pdf](https://arxiv.org/pdf/2311.09191.pdf)]  
  WACV 2023.11 [Few-Shot Action Recognition]  
  针对视觉与文本模态对齐问题，引入了一种样本效率高的域自适应策略，在不微调主模型的情况下改善了目标分布上的模态内(图像-图像)和模态间对齐  
- Multi-behavior Self-supervised Learning for Recommendation [[pdf](https://arxiv.org/pdf/2305.18238.pdf)]  
  SIGIR 2023.7 [Multi-behavior Recommendation]  
  针对目标行为的稀疏性，提出利用对比学习通过辅助行为缓解目标行为的稀疏性  
- Adaptive Graph Contrastive Learning for Recommendation [[pdf](https://arxiv.org/abs/2305.10837.pdf)]  
  KDD 2023.8 [Graph Contrastive Learning]  
  针对数据稀疏性和噪声，提出生成两个自适应图，进行对比学习  
- Graph Masked Autoencoder for Sequential Recommendation [[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591692)]  
  SIGIR 2023.7 [Masked Autoencoder Sequential Recommendation]  
  针对对比学习的数据增强方式需要人工设计，通过下游任务自适应调整路径屏蔽  
- A Preference Learning Decoupling Framework for User [[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591627)]  
  SIGIR 2023.7 [Cold-start user recommendation]  
  针对现有的元学习推荐没有区分新用户和新偏好以及任务不互斥导致的过拟合现象，通过解耦新用户和新偏好以及对元学习中的原始任务添加噪音扰动实现元任务增强实现冷启动用户偏好的学习  
- Cold-Start Recommendation [[pdf](https://arxiv.org/abs/2310.05736)]  
  EMNLP 2023 [RAG]  
  压缩prompt的方式  
- VPA:Fully Test-Time Visual Prompt Adaptation [[pdf](https://arxiv.org/pdf/2309.15251.pdf)]  
  MM 2023.8 [zero-shot image classification]  
  针对测试时间适应领域缺乏视觉prompt的研究，提出了第一个结合测试时间适应微调视觉prompt的框架  
- Exploring Visual Prompts for Adapting Large-Scale Models [[pdf](https://arxiv.org/pdf/2203.17274.pdf)]  
  arxiv 2022 [image classification]  
  针对现有的token形式的视觉prompt缺乏通用性，结合对抗性重编程的思想，设计了一种像素形式的视觉prompt  
- Text Is MASS: Modeling as Stochastic Embedding for Text-Video Retrieval [[pdf](https://arxiv.org/pdf/2403.17998.pdf)]  
  CVPR 2024.3 [Text-Video Retrieval]  
  针对单个文本嵌入很难完全表达视频的冗余语义问题，提出了一种新的随机文本建模方法 T-MASS，即文本被建模为随机嵌入，以灵活且具有弹性的语义范围丰富文本嵌入.  
- IMAGE2SENTENCE BASED ASYMMETRIC ZERO-SHOTCOMPOSED IMAGE RETRIEVAL [[pdf](https://arxiv.org/abs/2403.01431.pdf)]  
  ICLR 2024.3 [composed image retrieval (CIR)]  
  针对基于MLP的方法将图片映射成单一token不足以有效表示图像特征的问题，提出了一种新的自适应令牌学习器，它将图像映射到 VL 模型的词嵌入空间中的一个句子, 句子自适应地捕获有区别的视觉信息.  
- ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation [[pdf](https://arxiv.org/pdf/2308.11131.pdf)]  
  WWW 2023.8 [LLM RS]  
  提出了一种检索增强的方法增强用户交互序列，结合 Instruction Tuning 实现了对于 Few-shot/Zero-shot 场景下性能提升  
- RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation [[pdf](https://arxiv.org/pdf/2312.16018v2.pdf)]  
  WWW 2023.12 [LLM RS]  
  提出了一种用 Pointwise, pairwise 和 Listwise 结合的任务实现 LLM 的 instruction tuning 方法，并在 Prompt 中融合传统 RS 模型的信号。
- Aiming at the Target: Filter Collaborative Information for Cross-Domain Recommendation [[pdf](https://arxiv.org/abs/2403.20296)]  
  SIGIR 2024.3 [cross-domain recommendation]  
  针对负迁移问题，提出了直接过滤具有目标域用户相似性约束的源域用户协作信息来缓解负迁移
- Cross-domain Recommendation From Implicit Feedback [[pdf](https://openreview.net/forum?id=wi8wMFuO0H)]  
  ICLR 2023.9.12 [LLM RS]  
  提出隐式反馈的跨域推荐，并针对其提出隐式反馈采样负样本，校准因子函数，动态去噪
- Data-efficient Fine-tuning for LLM-based Recommendation [[pdf](http://arxiv.org/abs/2401.17197)]  
  SIGIR 2024.1 [LLM finetune for Rec]  
  针对LLM在推荐系统的finetune成本巨大问题，在数据修建任务下，提出了使用影响分数和努力分数识别高价值样本，构建小型数据集finetune LLM的工作。
- Disentangling ID and Modality Effects for Session-based Recommendation [[pdf](http://arxiv.org/abs/2404.12969)]  
  SIGIR 2024.4 [muitymodal Rec]  
  针对现有多模态推荐系统忽略了id信息和物品模态信息捕捉偏好原理差异（前者是共现，后者是用户的细粒度偏好）的问题，提出了解耦id embedding和模态信息embedding对模型训练影响的多任务训练方法

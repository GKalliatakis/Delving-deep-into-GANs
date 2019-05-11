# Delving deep into Generative Adversarial Networks (GANs) 	[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Delving%20deep%20into%20Generative%20Adversarial%20Networks.%20&url=https://github.com/GKalliatakis/Delving-deep-into-GANs&via=GKalliatakis&hashtags=GANs,paper,code,github)


## A curated, quasi-exhaustive list of state-of-the-art publications and resources about Generative Adversarial Networks (GANs) and their applications. 

### Background
Generative models are models that can learn to create data that is similar to data that we give them. One of the most promising approaches of those models are Generative Adversarial Networks (GANs), a branch of unsupervised machine learning implemented by a system of two neural networks competing against each other in a zero-sum game framework. They were first introduced by Ian Goodfellow et al. in 2014. This repository aims at presenting an elaborate list of the state-of-the-art works on the field of Generative Adversarial Networks since their introduction in 2014.

<p align="center">
  <img src="https://raw.githubusercontent.com/GKalliatakis/Delving-deep-into-GANs/master/GAN2.gif?raw=true"/><br>
  Image taken from http://multithreaded.stitchfix.com/blog/2016/02/02/a-fontastic-voyage/<br>
</p>

**_This is going to be an evolving repo and I will keep updating it so make sure you have starred and forked this repository before moving on !_**

---

### :link: Contents
* [Opening Publication](#pushpin-opening-publication)
* [Latest paper from Ian Goodfellow](#fire-latest-paper-from-ian-goodfellow)
* [Papers](#clipboard-papers-descending-order-based-on-google-scholar-citations)
* [Theory](#notebook_with_decorative_cover-theory)
* [Presentations](#nut_and_bolt-presentations)
* [Courses](#books-courses--tutorials--blogs-webpages-unless-other-is-stated)
* [Code / Resources / Models](#package-resources--models-descending-order-based-on-github-stars)
* [Frameworks & Libraries](#electric_plug-frameworks--libraries-descending-order-based-on-github-stars)
---

### :busts_in_silhouette: Contributing
Contributions are welcome !! If you have any suggestions (missing or new papers, missing repos or typos) you can pull a request or start a discussion.

---


### :pushpin: Opening Publication 	
Generative Adversarial Nets (GANs) (2014) [[pdf]](https://arxiv.org/pdf/1406.2661v1.pdf)  [[presentation]](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf) [[code]](https://github.com/goodfeli/adversarial) [[video]](https://www.youtube.com/watch?v=HN9NRhm9waY)


### :fire: Latest paper from Ian Goodfellow

Self-Attention Generative Adversarial Networks (SAGAN) (2018) [[pdf]](https://arxiv.org/pdf/1805.08318.pdf) [[PyTorch implementation]](https://github.com/heykeetae/Self-Attention-GAN) 

---

### :clipboard: Papers (Descending order based on Google Scholar Citations)

S/N|Paper|Year|Citations
:---:|:---:|:---:|:---:
|1|Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGANs)  [[pdf]](https://arxiv.org/pdf/1511.06434v2.pdf)|2015|1534
|2|Explaining and Harnessing Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)|2014|895
|3|Improved Techniques for Training GANs  [[pdf]](https://arxiv.org/pdf/1606.03498v1.pdf )|2016|748
|4| :chart_with_upwards_trend: Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)  [[pdf]](https://arxiv.org/pdf/1611.07004.pdf)|2016|706
|5| :chart_with_upwards_trend: Wasserstein GAN (WGAN)  [[pdf]](https://arxiv.org/pdf/1701.07875.pdf)|2017|587
|6| :chart_with_upwards_trend: Conditional Generative Adversarial Nets (CGAN)  [[pdf]](https://arxiv.org/pdf/1411.1784v1.pdf)|2014|566
|7|Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (LAPGAN)  [[pdf]](https://arxiv.org/pdf/1506.05751.pdf)|2015|564
|8| :chart_with_upwards_trend: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)  [[pdf]](https://arxiv.org/pdf/1609.04802.pdf)|2016|549
|9|Semi-Supervised Learning with Deep Generative Models  [[pdf]](https://arxiv.org/pdf/1406.5298v2.pdf )|2014|494
|10| :chart_with_upwards_trend: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)  [[pdf]](https://arxiv.org/pdf/1703.10593.pdf)|2017|438
|11|InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1606.03657)|2016|394
|12|Context Encoders: Feature Learning by Inpainting  [[pdf]](https://arxiv.org/pdf/1604.07379)|2016|390
|13|Generative Adversarial Text to Image Synthesis  [[pdf]](https://arxiv.org/pdf/1605.05396)|2016|368
|14| :chart_with_upwards_trend: Improved Training of Wasserstein GANs (WGAN-GP)  [[pdf]](https://arxiv.org/pdf/1704.00028.pdf)|2017|331
|15|Deep multi-scale video prediction beyond mean square error  [[pdf]](https://arxiv.org/pdf/1511.05440.pdf)|2015|301
|16|Adversarial Autoencoders  [[pdf]](https://arxiv.org/pdf/1511.05644.pdf)|2015|277
|17|Energy-based Generative Adversarial Network (EBGAN)  [[pdf]](https://arxiv.org/pdf/1609.03126.pdf)|2016|238
|18|Autoencoding beyond pixels using a learned similarity metric (VAE-GAN)  [[pdf]](https://arxiv.org/pdf/1512.09300.pdf)|2015|233
|19| :chart_with_upwards_trend: Conditional Image Generation with PixelCNN Decoders  [[pdf]](https://arxiv.org/pdf/1606.05328.pdf)|2015|231
|20|Towards Principled Methods for Training Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1701.04862.pdf)|2017|229
|21|Adversarial Feature Learning (BiGAN)  [[pdf]](https://arxiv.org/pdf/1605.09782v6.pdf)|2016|224
|22| :chart_with_upwards_trend: Stacked Generative Adversarial Networks (SGAN)  [[pdf]](https://arxiv.org/pdf/1612.04357.pdf)|2016|215
|23| :chart_with_upwards_trend: StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1710.10916.pdf)|2017|214
|24|Adversarially Learned Inference (ALI)  [[pdf]](https://arxiv.org/pdf/1606.00704.pdf)|2016|211
|25| :chart_with_upwards_trend: Conditional Image Synthesis with Auxiliary Classifier GANs (AC-GAN)  [[pdf]](https://arxiv.org/pdf/1610.09585.pdf)|2016|202
|26| :chart_with_upwards_trend: Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN) by Apple  [[pdf]](https://arxiv.org/pdf/1612.07828v1.pdf)|2016|192
|27|f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization  [[pdf]](https://arxiv.org/pdf/1606.00709.pdf)|2016|186
|28|Generating Videos with Scene Dynamics (VGAN)  [[pdf]](http://web.mit.edu/vondrick/tinyvideo/paper.pdf)|2016|179
|29|Generative Visual Manipulation on the Natural Image Manifold (iGAN)  [[pdf]](https://arxiv.org/pdf/1609.03552.pdf)|2016|172
|30| :chart_with_upwards_trend: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (3D-GAN)  [[pdf]](https://arxiv.org/pdf/1610.07584)|2016|171
|31|Generative Moment Matching Networks  [[pdf]](https://arxiv.org/pdf/1502.02761.pdf)|2015|167
|32| :chart_with_upwards_trend: Coupled Generative Adversarial Networks (CoGAN)  [[pdf]](https://arxiv.org/pdf/1606.07536)|2016|162
|33| :chart_with_upwards_trend: BEGAN: Boundary Equilibrium Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1703.10717.pdf)|2017|162
|34|Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1602.02697.pdf)|2016|161
|35|Generating Images with Perceptual Similarity Metrics based on Deep Networks   [[pdf]](https://arxiv.org/pdf/1602.02644)|2016|151
|36| :chart_with_upwards_trend: Improving Variational Inference with Inverse Autoregressive Flow  [[pdf]](https://arxiv.org/pdf/1606.04934)|2016|150
|37|Unsupervised Learning for Physical Interaction through Video Prediction   [[pdf]](https://arxiv.org/pdf/1605.07157)|2016|146
|38|Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks (CatGAN)  [[pdf]](https://arxiv.org/pdf/1511.06390.pdf)|2015|141
|39|Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (DiscoGAN) [[pdf]](https://arxiv.org/pdf/1703.05192.pdf)|2017|135
|40|Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks (MGAN)  [[pdf]](https://arxiv.org/pdf/1604.04382.pdf)|2016|130
|41|SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient    [[pdf]](https://arxiv.org/pdf/1609.05473.pdf)|2016|125
|42|Generative Adversarial Imitation Learning  [[pdf]](https://arxiv.org/pdf/1606.03476)|2016|123
|43| :chart_with_upwards_trend: Adversarial Discriminative Domain Adaptation  [[pdf]](https://arxiv.org/pdf/1702.05464)|2017|123
|44|Generative Image Modeling using Style and Structure Adversarial Networks (S^2GAN)  [[pdf]](https://arxiv.org/pdf/1603.05631.pdf)|2016|121
|45|Unsupervised Cross-Domain Image Generation (DTN)  [[pdf]](https://arxiv.org/pdf/1611.02200.pdf)|2016|116
|46|Synthesizing the preferred inputs for neurons in neural networks via deep generator networks   [[pdf]](https://arxiv.org/pdf/1605.09304)|2016|99
|47|Least Squares Generative Adversarial Networks (LSGAN)  [[pdf]](https://arxiv.org/pdf/1611.04076.pdf)|2016|98
|48|Semantic Image Inpainting with Perceptual and Contextual Losses   [[pdf]](https://arxiv.org/pdf/1607.07539.pdf)|2016|98
|49|Conditional generative adversarial nets for convolutional face generation [[pdf]](https://pdfs.semanticscholar.org/42f6/f5454dda99d8989f9814989efd50fe807ee8.pdf)|2014|95
|50|StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1612.03242.pdf)|2016|90
|51|Mode Regularized Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1612.02136)|2016|89
|52|Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro [[pdf]](https://arxiv.org/pdf/1701.07717)|2017|87
|53|Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space (PPGN)  [[pdf]](https://arxiv.org/pdf/1612.00005.pdf)|2016|85
|54|DualGAN: Unsupervised Dual Learning for Image-to-Image Translation    [[pdf]](https://arxiv.org/pdf/1704.02510.pdf)|2017|85
|55|Training generative neural networks via Maximum Mean Discrepancy optimization  [[pdf]](https://arxiv.org/pdf/1505.03906.pdf)|2015|82
|56|Generating images with recurrent adversarial networks  [[pdf]](https://arxiv.org/pdf/1602.05110.pdf)|2016|81
|57|Semantic Segmentation using Adversarial Networks   [[pdf]](https://arxiv.org/pdf/1611.08408.pdf)|2016|81
|58|Learning What and Where to Draw (GAWWN)  [[pdf]](https://arxiv.org/pdf/1610.02454v1.pdf)|2016|77
|59|Amortised MAP Inference for Image Super-resolution (AffGAN)  [[pdf]](https://arxiv.org/pdf/1610.04490.pdf)|2016|75
|60|Generalization and Equilibrium in Generative Adversarial Nets (GANs)  [[pdf]](https://arxiv.org/pdf/1703.00573)|2017|74
|61|VIME: Variational Information Maximizing Exploration  [[pdf]](https://arxiv.org/pdf/1605.09674)|2016|70
|62|Disentangled Representation Learning GAN for Pose-Invariant Face Recognition   [[pdf]](http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf)|2017|70
|63|Neural Photo Editing with Introspective Adversarial Networks (IAN)  [[pdf]](https://arxiv.org/pdf/1609.07093.pdf)|2016|63
|64|Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks  [[pdf]](https://arxiv.org/pdf/1704.01155)|2017|63
|65|Learning in Implicit Generative Models  [[pdf]](https://arxiv.org/pdf/1610.03483.pdf)|2016|62
|66|Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis (TP-GAN)  [[pdf]](https://arxiv.org/pdf/1704.04086.pdf)|2017|60
|67|On the Quantitative Analysis of Decoder-Based Generative Models  [[pdf]](https://arxiv.org/pdf/1611.04273.pdf)|2016|59
|68|Invertible Conditional GANs for image editing (IcGAN)  [[pdf]](https://arxiv.org/pdf/1611.06355.pdf)|2016|57
|69|Unrolled Generative Adversarial Networks (Unrolled GAN)  [[pdf]](https://arxiv.org/pdf/1611.02163.pdf)|2016|56
|70|Attend, infer, repeat: Fast scene understanding with generative models  [[pdf]](https://arxiv.org/pdf/1603.08575.pdf)|2016|55
|71|Pixel-Level Domain Transfer   [[pdf]](https://arxiv.org/pdf/1603.07442)|2016|54
|72|SEGAN: Speech Enhancement Generative Adversarial Network    [[pdf]](https://arxiv.org/pdf/1703.09452.pdf)|2017|45
|73|MMD GAN: Towards Deeper Understanding of Moment Matching Network [[pdf]](https://arxiv.org/pdf/1705.08584.pdf)|2017|42
|74|Learning a Driving Simulator [[pdf]](https://arxiv.org/pdf/1608.01230)|2016|41
|75|Image De-raining Using a Conditional Generative Adversarial Network (ID-CGAN)  [[pdf]](https://arxiv.org/pdf/1701.05957)|2017|40
|76|Face Aging With Conditional Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.01983.pdf)|2017|39
|77| :chart_with_upwards_trend: Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (AnoGAN)  [[pdf]](https://arxiv.org/pdf/1703.05921.pdf)|2017|39
|78|Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities (LS-GAN)  [[pdf]](https://arxiv.org/pdf/1701.06264.pdf)|2017|38
|79|Adversarial Attacks on Neural Network Policies [[pdf]](https://arxiv.org/pdf/1702.02284.pdf)|2017|37
|80|AdaGAN: Boosting Generative Models  [[pdf]](https://arxiv.org/pdf/1701.02386.pdf)|2017|37
|81|Triple Generative Adversarial Nets (Triple-GAN)  [[pdf]](https://arxiv.org/pdf/1703.02291.pdf)|2017|37
|82|Semantic Image Inpainting with Deep Generative Models [[pdf]](https://arxiv.org/pdf/1607.07539.pdf)|2017|37
|83|A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection   [[pdf]](https://arxiv.org/pdf/1704.03414.pdf)|2017|36
|84|Adversarial Examples for Semantic Segmentation and Object Detection  [[pdf]](https://arxiv.org/pdf/1703.08603.pdf)|2017|35
|85|Generative face completion   [[pdf]](https://arxiv.org/pdf/1704.05838)|2016|34
|86|Age Progression / Regression by Conditional Adversarial Autoencoder [[pdf]](https://arxiv.org/pdf/1702.08423)|2017|32
|87|The Space of Transferable Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1704.03453)|2017|32
|88|Learning to Protect Communications with Adversarial Neural Cryptography [[pdf]](https://arxiv.org/pdf/1610.06918)|2016|31
|89| Perceptual Generative Adversarial Networks for Small Object Detection [[pdf]](https://arxiv.org/abs/1706.05274)|2017|30
|90|Temporal Generative Adversarial Nets (TGAN)  [[pdf]](https://arxiv.org/pdf/1611.06624.pdf)|2016|29
|91|Towards Large-Pose Face Frontalization in the Wild   [[pdf]](https://arxiv.org/pdf/1704.06244)|2016|29
|92|Boundary-Seeking Generative Adversarial Networks (BS-GAN)  [[pdf]](https://arxiv.org/pdf/1702.08431.pdf)|2017|28
|93|McGan: Mean and Covariance Feature Matching GAN    [[pdf]](https://arxiv.org/pdf/1702.08398.pdf)|2017|28
|94|The Cramer Distance as a Solution to Biased Wasserstein Gradients [[pdf]](https://arxiv.org/pdf/1705.10743.pdf)|2017|28
|95|SalGAN: Visual Saliency Prediction with Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1701.01081.pdf)|2016|28
|96|MoCoGAN: Decomposing Motion and Content for Video Generation [[pdf]](https://arxiv.org/pdf/1707.04993.pdf)|2017|27
|97|Imitating Driver Behavior with Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1701.06699)|2017|27
|98|A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models  [[pdf]](https://arxiv.org/pdf/1611.03852.pdf)|2016|26
|99|Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN (MalGAN)  [[pdf]](https://arxiv.org/pdf/1702.05983.pdf)|2016|26
|100|Maximum-Likelihood Augmented Discrete Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.07983)|2017|26
|101|Neural Face Editing with Intrinsic Image Disentangling  [[pdf]](https://arxiv.org/abs/1704.04131)|2017|26
|102|Pose Guided Person Image Generation [[pdf]](https://arxiv.org/pdf/1705.09368.pdf)|2017|26
|103|Good Semi-supervised Learning that Requires a Bad GAN [[pdf]](https://arxiv.org/abs/1705.09783)|2017|25
|104|Connecting Generative Adversarial Networks and Actor-Critic Methods  [[pdf]](https://arxiv.org/pdf/1610.01945.pdf)|2016|24
|105|LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation   [[pdf]](https://arxiv.org/pdf/1703.01560.pdf)|2017|24
|106|On Convergence and Stability of GANs  [[pdf]](https://arxiv.org/pdf/1705.07215.pdf)|2017|23
|107|C-RNN-GAN: Continuous recurrent neural networks with adversarial training    [[pdf]](https://arxiv.org/pdf/1611.09904.pdf)|2016|22
|108|Towards Diverse and Natural Image Descriptions via a Conditional GAN [[pdf]](https://arxiv.org/pdf/1703.06029.pdf)|2017|22
|109|Full Resolution Image Compression with Recurrent Neural Networks [[pdf]](https://arxiv.org/pdf/1608.05148)|2016|21
|110|Recurrent Topic-Transition GAN for Visual Paragraph Generation (RTT-GAN)  [[pdf]](https://arxiv.org/pdf/1703.07022.pdf)|2017|21
|111|Multi-View Image Generation from a Single-View   [[pdf]](https://arxiv.org/pdf/1704.04886)|2016|21
|112|Adversarial Transformation Networks: Learning to Generate Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1703.09387)|2017|21
|113|Simple Black-Box Adversarial Perturbations for Deep Networks  [[pdf]](https://arxiv.org/pdf/1612.06299)|2016|21
|114|Deep Generative Adversarial Networks for Compressed Sensing (GANCS) Automates MRI  [[pdf]](https://arxiv.org/pdf/1706.00051.pdf)|2017|21
|115|RenderGAN: Generating Realistic Labeled Data    [[pdf]](https://arxiv.org/pdf/1611.01331.pdf)|2016|21
|116|Stabilizing Training of Generative Adversarial Networks through Regularization [[pdf]](https://arxiv.org/abs/1705.09367)|2017|21
|117|CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1705.02355)|2017|20
|118|Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1704.00849.pdf)|2017|19
|119| :chart_with_upwards_trend: Gradient descent GAN optimization is locally stable  [[pdf]](https://arxiv.org/abs/1706.04156)|2017|19
|120|Adversarial Training Methods for Semi-Supervised Text Classification  [[pdf]](https://arxiv.org/abs/1605.07725)|2016|18
|121|Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks (SSL-GAN)  [[pdf]](https://arxiv.org/pdf/1611.06430.pdf)|2016|18
|122|Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation [[pdf]](https://arxiv.org/pdf/1705.00389.pdf)|2017|18
|123|Multi-Agent Diverse Generative Adversarial Networks (MAD-GAN)  [[pdf]](https://arxiv.org/pdf/1704.02906.pdf)|2017|18
|124|ALICE: Towards Understanding Adversarial Learning for Joint Distribution Matching [[pdf]](http://papers.nips.cc/paper/7133-alice-towards-understanding-adversarial-learning-for-joint-distribution-matching.pdf)|2017|18
|125|Cooperative Training of Descriptor and Generator Networks [[pdf]](https://arxiv.org/pdf/1609.09408.pdf)|2016|17
|126|3D Shape Induction from 2D Views of Multiple Objects (PrGAN)  [[pdf]](https://arxiv.org/pdf/1612.05872.pdf)|2016|17
|127|Learning to Generate Images of Outdoor Scenes from Attributes and Semantic Layouts (AL-CGAN)  [[pdf]](https://arxiv.org/pdf/1612.00215.pdf)|2016|17
|128|Conditional CycleGAN for Attribute Guided Face Image Generation [[pdf]](https://arxiv.org/abs/1705.09966)|2017|17
|129|Objective-Reinforced Generative Adversarial Networks (ORGAN) [[pdf]](https://arxiv.org/pdf/1705.10843.pdf)|2017|17
|130|Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models [[pdf]](https://arxiv.org/pdf/1705.10843.pdf)|2017|17
|131|Generative Multi-Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1611.01673.pdf)|2016|16
|132|Learning Representations of Emotional Speech with Deep Convolutional Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1705.02394)|2017|16
|133|Dual Motion GAN for Future-Flow Embedded Video Prediction [[pdf]](https://arxiv.org/pdf/1708.00284.pdf)|2017|16
|134|VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning () [[pdf]](https://arxiv.org/abs/1705.07761)|2017|16
|135|Unsupervised Image-to-Image Translation with Generative Adversarial Networks   [[pdf]](https://arxiv.org/pdf/1701.02676.pdf)|2017|15
|136|Improved generator objectives for GANs  [[pdf]](https://arxiv.org/pdf/1612.02780.pdf)|2017|15
|137|DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data  [[pdf]](https://arxiv.org/abs/1706.02071)|2017|15
|138|CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training   [[pdf]](https://arxiv.org/pdf/1703.10155.pdf)|2017|15
|139|Improved Semi-supervised Learning with GANs using Manifold Invariances [[pdf]](https://arxiv.org/abs/1705.08850)|2017|14
|140|Inverting The Generator Of A Generative Adversarial Network  [[pdf]](https://arxiv.org/pdf/1611.05644)|2016|14
|141|Precise Recovery of Latent Vectors from Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.04782)|2016|14
|142|Comparison of Maximum Likelihood and GAN-based training of Real NVPs [[pdf]](https://arxiv.org/pdf/1705.05263.pdf)|2017|14
|143|Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.01705)|2017|14
|144|Semantically Decomposing the Latent Spaces of Generative Adversarial Networks (SD-GAN) [[pdf]](https://arxiv.org/abs/1705.07904)|2017|14
|145|Adversarial Generation of Natural Language [[pdf]](https://arxiv.org/abs/1705.10929)|2017|14
|146|MAGAN: Margin Adaptation for Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1704.03817.pdf)|2017|13
|147|Reconstruction of three-dimensional porous media using generative adversarial neural networks [[pdf]](https://arxiv.org/pdf/1704.03225)|2017|13
|148|SCAN: Structure Correcting Adversarial Network for Chest X-rays Organ Segmentation  [[pdf]](https://arxiv.org/pdf/1703.08770)|2017|13
|149|Generating Multi-label Discrete Electronic Health Records using Generative Adversarial Networks (MedGAN)  [[pdf]](https://arxiv.org/pdf/1703.06490.pdf)|2017|13
|150|SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation  [[pdf]](https://arxiv.org/abs/1706.01805)|2017|13
|151| :chart_with_upwards_trend: CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms [[pdf]](https://arxiv.org/pdf/1706.07068.pdf)|2017|13
|152|It Takes (Only) Two: Adversarial Generator-Encoder Networks [[pdf]](https://arxiv.org/pdf/1704.02304.pdf)|2017|12
|153|Generative Temporal Models with Memory  [[pdf]](https://arxiv.org/pdf/1702.04649.pdf)|2017|12
|154|Generative Adversarial Residual Pairwise Networks for One Shot Learning  [[pdf]](https://arxiv.org/pdf/1703.08033)|2017|12
|155|Crossing Nets: Combining GANs and VAEs with a Shared Latent Space for Hand Pose Estimation [[pdf]](https://arxiv.org/pdf/1702.03431.pdf)|2017|12
|156|Texture Synthesis with Spatial Generative Adversarial Networks (SGAN)  [[pdf]](https://arxiv.org/pdf/1611.08207.pdf)|2016|12
|157|Semi-Latent GAN: Learning to generate and modify facial images from attributes (SL-GAN)   [[pdf]](https://arxiv.org/pdf/1704.02166.pdf)|2017|12
|158|Adversarial Networks for the Detection of Aggressive Prostate Cancer [[pdf]](https://arxiv.org/pdf/1702.08014)|2017|12
|159|Auto-painter: Cartoon Image Generation from Sketch by Using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1705.01908.pdf)|2017|12
|160|Interactive 3D Modeling with a Generative Adversarial Network [[pdf]](https://arxiv.org/abs/1706.05170)|2017|11
|161|Language Generation with Recurrent Generative Adversarial Networks without Pre-training  [[pdf]](https://arxiv.org/abs/1706.01399)|2017|11
|162|GP-GAN: Towards Realistic High-Resolution Image Blending   [[pdf]](https://arxiv.org/pdf/1703.07195.pdf)|2017|11
|163|Universal Adversarial Perturbations Against Semantic Image Segmentation  [[pdf]](https://arxiv.org/pdf/1704.05712)|2017|10
|164|A General Retraining Framework for Scalable Adversarial Classification [[pdf]](https://arxiv.org/pdf/1604.02606.pdf)|2016|10
|165|Contextual RNN-GANs for Abstract Reasoning Diagram Generation (Context-RNN-GAN)  [[pdf]](https://arxiv.org/pdf/1609.09444.pdf)|2016|10
|166|Adversarial Image Perturbation for Privacy Protection--A Game Theory Perspective  [[pdf]](https://arxiv.org/pdf/1703.09471)|2017|10
|167|Multi-Generator Gernerative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1708.02556.pdf)|2017|10
|168|Optimizing the Latent Space of Generative Networks [[pdf]](https://arxiv.org/pdf/1707.05776.pdf)|2017|10
|169|GANs for Biological Image Synthesis [[pdf]](https://arxiv.org/pdf/1708.04692.pdf)|2017|10
|170|MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation using 1D and 2D Conditions   [[pdf]](https://arxiv.org/pdf/1703.10847.pdf)|2016|10
|171| :chart_with_upwards_trend: Deep Generative Adversarial Compression Artifact Removal  [[pdf]](https://arxiv.org/pdf/1704.02518)|2017|10
|172|Adversarial Deep Structural Networks for Mammographic Mass Segmentation  [[pdf]](https://arxiv.org/abs/1612.05970)|2017|9
|173|Adversarial Training For Sketch Retrieval (SketchGAN)  [[pdf]](https://arxiv.org/pdf/1607.02748.pdf)|2016|9
|174|Perceptual Adversarial Networks for Image-to-Image Transformation  [[pdf]](https://arxiv.org/pdf/1706.09138)|2017|9
|175|PixelGAN Autoencoders [[pdf]](https://arxiv.org/pdf/1706.00531.pdf)|2017|9
|176|Unsupervised Diverse Colorization via Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.06674.pdf)|2017|9
|177|TAC-GAN - Text Conditioned Auxiliary Classifier Generative Adversarial Network    [[pdf]](https://arxiv.org/pdf/1703.06412.pdf)|2017|9
|178|Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking (GoGAN)  [[pdf]](https://arxiv.org/pdf/1704.04865.pdf)|2017|8
|179|From source to target and back: symmetric bi-directional adaptive GAN [[pdf]](https://arxiv.org/abs/1705.08824)|2017|8
|180|Robust LSTM-Autoencoders for Face De-Occlusion in the Wild   [[pdf]](https://arxiv.org/pdf/1612.08534.pdf)|2016|8
|181|Ensembles of Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1612.00991.pdf)|2016|8
|182|Bayesian GAN [[pdf]](https://arxiv.org/abs/1705.09558)|2017|8
|183|Outline Colorization through Tandem Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.08834.pdf)|2017|8
|184|Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss  [[pdf]](https://arxiv.org/pdf/1708.00961.pdf)|2017|8
|185|ArtGAN: Artwork Synthesis with Conditional Categorial GANs   [[pdf]](https://arxiv.org/pdf/1702.03410.pdf)|2017|8
|186|Generative Semantic Manipulation with Contrasting GAN [[pdf]](https://arxiv.org/pdf/1708.00315.pdf)|2017|8
|187|Steganographic Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1703.05502)|2017|7
|188|GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data  [[pdf]](https://arxiv.org/pdf/1705.04932.pdf)|2017|7
|189|Generative Adversarial Parallelization  [[pdf]](https://arxiv.org/pdf/1612.04021)|2015|7
|190|CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training  [[pdf]](https://arxiv.org/pdf/1709.02023.pdf)|2017|7
|191|Automatic Liver Segmentation Using an Adversarial Image-to-Image Network [[pdf]](https://arxiv.org/pdf/1707.08037.pdf)|2017|7
|192|Megapixel Size Image Creation using Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1706.00082.pdf)|2017|6
|193|Style Transfer for Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN  [[pdf]](https://arxiv.org/abs/1706.03319)|2017|6
|194|Semantic Image Synthesis via Adversarial Learning [[pdf]](https://arxiv.org/pdf/1707.06873.pdf)|2017|6
|195|WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images    [[pdf]](https://arxiv.org/pdf/1702.07392.pdf)|2017|6
|196|Retinal Vessel Segmentation in Fundoscopic Images with Generative Adversarial Networks  [[pdf]](https://arxiv.org/abs/1706.09318)|2017|6
|197|Representation Learning and Adversarial Generation of 3D Point Clouds [[pdf]](https://arxiv.org/pdf/1707.02392.pdf)|2017|6
|198|3D Object Reconstruction from a Single Depth View with Adversarial Learning [[pdf]](https://arxiv.org/pdf/1708.07969.pdf)|2017|6
|199|Abnormal Event Detection in Videos using Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1708.09644.pdf)|2017|6
|200|Improved Adversarial Systems for 3D Object Generation and Reconstruction  [[pdf]](https://arxiv.org/pdf/1707.09557.pdf)|2017|6
|201|Relaxed Wasserstein with Applications to GANs (RWGAN ) [[pdf]](https://arxiv.org/abs/1705.07164)|2017|5
|202|Generative Adversarial Networks as Variational Training of Energy Based Models (VGAN)  [[pdf]](https://arxiv.org/pdf/1611.01799.pdf)|2017|5
|203|Flow-GAN: Bridging implicit and prescribed learning in generative models [[pdf]](https://arxiv.org/abs/1705.08868)|2017|5
|204|Weakly Supervised Generative Adversarial Networks for 3D Reconstruction  [[pdf]](https://arxiv.org/abs/1705.10904)|2017|5
|205|Auto-Encoder Guided GAN for Chinese Calligraphy Synthesis  [[pdf]](https://arxiv.org/abs/1706.08789)|2017|5
|206|ExprGAN: Facial Expression Editing with Controllable Expression Intensity [[pdf]](https://arxiv.org/pdf/1709.03842.pdf)|2017|5
|207| Learning Texture Manifolds with the Periodic Spatial GAN [[pdf]](https://arxiv.org/abs/1705.06566)|2017|5
|208|SAD-GAN: Synthetic Autonomous Driving using Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1611.08788.pdf)|2017|5
|209|Compressed Sensing MRI Reconstruction with Cyclic Loss in Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.00753.pdf)|2017|5
|210|Generative Adversarial Network-based Synthesis of Visible Faces from Polarimetric Thermal Faces [[pdf]](https://arxiv.org/pdf/1708.02681.pdf)|2017|5
|211|Guiding InfoGAN with Semi-Supervision [[pdf]](https://arxiv.org/pdf/1707.04487.pdf)|2017|5
|212|APE-GAN: Adversarial Perturbation Elimination with GAN [[pdf]](https://arxiv.org/pdf/1707.05474.pdf)|2017|5
|213|Binary Generative Adversarial Networks for Image Retrieval [[pdf]](https://arxiv.org/pdf/1708.04150.pdf)|2017|5
|214|Message Passing Multi-Agent GANs (MPM-GAN)  [[pdf]](https://arxiv.org/pdf/1612.01294.pdf)|2017|4
|215|Multi-view Generative Adversarial Networks (MV-BiGAN)  [[pdf]](https://arxiv.org/pdf/1611.02019.pdf)|2017|4
|216|MARTA GANs: Unsupervised Representation Learning for Remote Sensing Image Classification [[pdf]](https://arxiv.org/pdf/1612.08879.pdf)|2017|4
|217|Associative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1611.06953)|2017|4
|218|Generative Adversarial Networks for Multimodal Representation Learning in Video Hyperlinking [[pdf]](https://arxiv.org/abs/1705.05103)|2017|4
|219|TextureGAN: Controlling Deep Image Synthesis with Texture Patches  [[pdf]](https://arxiv.org/abs/1706.02823)|2017|4
|220|On the effect of Batch Normalization and Weight Normalization in Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.03971)|2017|4
|221|Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN  [[pdf]](https://arxiv.org/pdf/1705.03387)|2017|4
|222|SeGAN: Segmenting and Generating the Invisible   [[pdf]](https://arxiv.org/pdf/1703.10239.pdf)|2017|3
|223|Softmax GAN [[pdf]](https://arxiv.org/pdf/1704.06191)|2017|3
|224|Class-Splitting Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.07359.pdf)|2017|3
|225|High-Quality Facial Photo-Sketch Synthesis Using Multi-Adversarial Networks [[pdf]](https://arxiv.org/pdf/1710.10182.pdf)|2017|3
|226|Generative Adversarial Structured Networks  [[pdf]](https://sites.google.com/site/nips2016adversarial/WAT16_paper_14.pdf)|2017|3
|227|Geometric GAN  [[pdf]](https://arxiv.org/pdf/1705.02894.pdf)|2017|3
|228|Dualing GANs [[pdf]](https://arxiv.org/pdf/1706.06216.pdf)|2017|3
|229|Adversarial nets with perceptual losses for text-to-image synthesis [[pdf]](https://arxiv.org/pdf/1708.09321.pdf)|2017|3
|230|ARIGAN: Synthetic Arabidopsis Plants using Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1709.00938.pdf)|2017|3
|231|GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks [[pdf]](https://arxiv.org/pdf/1710.00962.pdf)|2017|3
|232|How to Fool Radiologists with Generative Adversarial Networks? A Visual Turing Test for Lung Cancer Diagnosis [[pdf]](https://arxiv.org/pdf/1710.09762.pdf)|2017|3
|233|Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1709.07592.pdf)|2017|3
|234|Sharpness-aware Low dose CT denoising using conditional generative adversarial network [[pdf]](https://arxiv.org/pdf/1708.06453.pdf)|2017|3
|235|AlignGAN: Learning to Align Cross-Domain Images with Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1707.01400.pdf)|2017|2
|236|Generate Identity-Preserving Faces by Generative Adversarial Networks  [[pdf]](https://arxiv.org/abs/1706.03227)|2017|2
|237|Creatism: A deep-learning photographer capable of creating professional work [[pdf]](https://arxiv.org/pdf/1707.03491.pdf%20)|2017|2
|238|Image Generation and Editing with Variational Info Generative Adversarial Networks (ViGAN)  [[pdf]](https://arxiv.org/pdf/1701.04568.pdf)|2016|2
|239|Supervised Adversarial Networks for Image Saliency Detection [[pdf]](https://arxiv.org/pdf/1704.07242)|2017|2
|240|Training Triplet Networks with GAN  [[pdf]](https://arxiv.org/pdf/1704.02227)|2017|2
|241|Training Triplet Networks with GAN  [[pdf]](https://arxiv.org/pdf/1704.02227)|2017|2
|242|Generative Mixture of Networks  [[pdf]](https://arxiv.org/pdf/1702.03307.pdf)|2017|2
|243|Stopping GAN Violence: Generative Unadversarial Networks  [[pdf]](https://arxiv.org/pdf/1703.02528.pdf)|2016|2
|244|A Classification-Based Perspective on GAN Distributions [[pdf]](https://arxiv.org/pdf/1711.00970.pdf)|2017|2
|245|CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning [[pdf]](https://arxiv.org/pdf/1710.05106.pdf)|2017|2
|246|Statistics of Deep Generated Images [[pdf]](https://arxiv.org/pdf/1708.02688.pdf)|2017|2
|247|Anti-Makeup: Learning A Bi-Level Adversarial Network for Makeup-Invariant Face Verification [[pdf]](https://arxiv.org/pdf/1709.03654.pdf)|2017|2
|248|Freehand Ultrasound Image Simulation with Spatially-Conditioned Generative Adversarial Networks [[pdf]](https://arxiv.org/ftp/arxiv/papers/1707/1707.05392.pdf)|2017|2
|249|Simultaneously Color-Depth Super-Resolution with Conditional Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1708.09105.pdf)|2017|2
|250|Socially-compliant Navigation through Raw Depth Inputs with Generative Adversarial Imitation Learning [[pdf]](https://arxiv.org/pdf/1710.02543.pdf)|2017|2
|251|Adversarial Generation of Training Examples for Vehicle License Plate Recognition [[pdf]](https://arxiv.org/pdf/1707.03124.pdf)|2017|1
|252|Aesthetic-Driven Image Enhancement by Adversarial Learning [[pdf]](https://arxiv.org/pdf/1707.05251.pdf)|2017|1
|253|Generative Adversarial Models for People Attribute Recognition in Surveillance [[pdf]](https://arxiv.org/pdf/1707.02240.pdf)|2017|1
|254|Improving Heterogeneous Face Recognition with Conditional Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.02848.pdf)|2017|1
|255|Intraoperative Organ Motion Models with an Ensemble of Conditional Generative Adversarial Networks  [[pdf]](https://arxiv.org/ftp/arxiv/papers/1709/1709.02255.pdf)|2017|1
|256|Label Denoising Adversarial Network (LDAN) for Inverse Lighting of Face Images [[pdf]](https://arxiv.org/pdf/1709.01993.pdf)|2017|1
|257|Face Super-Resolution Through Wasserstein GANs  [[pdf]](https://arxiv.org/pdf/1705.02438)|2017|1
|258|Continual Learning in Generative Adversarial Nets [[pdf]](https://arxiv.org/abs/1705.08395)|2017|1
|259|An Adversarial Regularisation for Semi-Supervised Training of Structured Output Neural Networks  [[pdf]](https://arxiv.org/pdf/1702.02382)|2017|1
|260|Generative Cooperative Net for Image Generation and Data Augmentation  [[pdf]](https://arxiv.org/pdf/1705.02887.pdf)|2017|1
|261|Learning Loss for Knowledge Distillation with Conditional Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.00513.pdf)|2017|1
|262|Parametrizing filters of a CNN with a GAN [[pdf]](https://arxiv.org/pdf/1710.11386.pdf)|2017|1
|263|A step towards procedural terrain generation with GANs [[pdf]](https://arxiv.org/pdf/1707.03383.pdf)|2017|1
|264|Adversarial Networks for Spatial Context-Aware Spectral Image Reconstruction from RGB [[pdf]](https://arxiv.org/pdf/1709.00265.pdf)|2017|1
|265|Controllable Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1708.00598.pdf)|2017|1
|266|Improved ArtGAN for Conditional Synthesis of Natural Image and Artwork [[pdf]](https://arxiv.org/pdf/1708.09533.pdf)|2017|1
|267|Microscopy Cell Segmentation via Adversarial Neural Networks  [[pdf]](https://arxiv.org/pdf/1709.05860.pdf)|2017|1
|268|Neural Stain-Style Transfer Learning using GAN for Histopathological Images [[pdf]](https://arxiv.org/pdf/1710.08543.pdf)|2017|1
|269|Deep and Hierarchical Implicit Models (Bayesian GAN)  [[pdf]](https://arxiv.org/pdf/1702.08896.pdf)|2017|0
|270|How to Train Your DRAGAN [[pdf]](https://arxiv.org/abs/1705.07215)|2017|0
|271|Activation Maximization Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1703.02000.pdf)|2017|0
|272|Generative Adversarial Nets with Labeled Data by Activation Maximization (AMGAN)  [[pdf]](https://arxiv.org/pdf/1703.02000.pdf)|2017|0
|273|Depth Structure Preserving Scene Image Generation  [[pdf]](https://arxiv.org/abs/1706.00212)|2017|0
|274|Synthesizing Filamentary Structured Images with GANs  [[pdf]](https://arxiv.org/abs/1706.02185)|2017|0
|275|Bayesian Conditional Generative Adverserial Networks [[pdf]](https://arxiv.org/pdf/1706.05477.pdf)|2017|0
|276|Generative Adversarial Networks with Inverse Transformation Unit [[pdf]](https://arxiv.org/pdf/1709.09354.pdf)|2017|0
|277|Image Quality Assessment Techniques Show Improved Training and Evaluation of Autoencoder Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1708.02237.pdf)|2017|0
|278|KGAN: How to Break The Minimax Game in GAN  [[pdf]](https://arxiv.org/pdf/1711.01744.pdf)|2017|0
|279|Linking Generative Adversarial Learning and Binary Classification [[pdf]](https://arxiv.org/pdf/1709.01509.pdf)|2017|0
|280|Structured Generative Adversarial Networks  [[pdf]](http://papers.nips.cc/paper/6979-structured-generative-adversarial-networks.pdf)|2017|0
|281|Tensorizing Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1710.10772.pdf)|2017|0
|282|A Novel Approach to Artistic Textual Visualization via GAN [[pdf]](https://arxiv.org/pdf/1710.10553.pdf)|2017|0
|283|Artificial Generation of Big Data for Improving Image Classification: A Generative Adversarial Network Approach on SAR Data [[pdf]](https://arxiv.org/pdf/1711.02010.pdf)|2017|0
|284|Conditional Adversarial Network for Semantic Segmentation of Brain Tumor [[pdf]](https://arxiv.org/pdf/1708.05227)|2017|0
|285|Data Augmentation in Classification using GAN [[pdf]](https://arxiv.org/pdf/1711.00648.pdf)|2017|0
|286|Deep Generative Adversarial Neural Networks for Realistic Prostate Lesion MRI Synthesis [[pdf]](https://arxiv.org/pdf/1708.00129.pdf)|2017|0
|287|Face Transfer with Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1710.06090.pdf)|2017|0
|288|Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1710.04835.pdf)|2017|0
|289|Generative Adversarial Network based on Resnet for Conditional Image Restoration  [[pdf]](https://arxiv.org/pdf/1707.04881.pdf)|2017|0
|290|Hierarchical Detail Enhancing Mesh-Based Shape Generation with 3D Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1709.07581.pdf)|2017|0
|291|High-Quality Face Image SR Using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1707.00737.pdf)|2017|0
|292|Improving image generative models with human interactions [[pdf]](https://arxiv.org/pdf/1709.10459.pdf)|2017|0
|293|Learning to Generate Chairs with Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1705.10413.pdf)|2017|0
|294|Retinal Vasculature Segmentation Using Local Saliency Maps and Generative Adversarial Networks For Image Super Resolution [[pdf]](https://arxiv.org/pdf/1710.04783.pdf)|2017|0



----------

### :notebook_with_decorative_cover: Theory
- Improved Techniques for Training GANs [[pdf]](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans.pdf) 	
- Energy-Based GANs & other Adversarial things by Yann Le Cun [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Mode RegularizedGenerative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1612.02136.pdf)

----------

### :nut_and_bolt: Presentations / Books
- Generative Adversarial Networks (GANs) by Ian Goodfellow [[pdf]](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf) 	
- Learning Deep Generative Models by Ruslan Salakhutdinov [[presentation]](https://slideplayer.com/slide/12912305/)

- GANs in Action: Deep learning with Generative Adversarial Networks [[link to book]](https://www.manning.com/books/gans-in-action) 	

----------

### :books: Courses / Tutorials / Blogs (Webpages unless other is stated)
- NIPS 2016 Tutorial: Generative Adversarial Networks (2016) [[pdf]](https://arxiv.org/pdf/1701.00160.pdf)
- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
- [Generative Models by OpenAI](https://openai.com/blog/generative-models/)
- [MNIST Generative Adversarial Model in Keras](https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/)
- [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/)
- [Attacking machine learning with adversarial examples by OpenAI](https://openai.com/blog/adversarial-example-research/)
- [On the intuition behind deep learning & GANs—towards a fundamental understanding](https://blog.waya.ai/introduction-to-gans-a-boxing-match-b-w-neural-nets-b4e5319cc935)
- [SimGANs - a game changer in unsupervised learning, self driving cars, and more](https://blog.waya.ai/simgans-applied-to-autonomous-driving-5a8c6676e36b)
----------


### :package: Resources / Models (Descending order based on GitHub stars)
S/N|Name|Repo|Stars
:---:|:---:|:---:|:---:
|1|Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)|https://github.com/junyanz/CycleGAN|6383
|2| :chart_with_upwards_trend: Image-to-image translation with conditional adversarial nets (pix2pix)|https://github.com/phillipi/pix2pix|4851
|3|Image super-resolution through deep learning|https://github.com/david-gpu/srez|4801
|4|Tensorflow implementation of Deep Convolutional Generative Adversarial Networks (DCGAN)|https://github.com/carpedm20/DCGAN-tensorflow|4223
|5| :chart_with_upwards_trend: Generative Models: Collection of generative models, e.g. GAN, VAE in Pytorch and Tensorflow|https://github.com/wiseodd/generative-models|3374
|6| :chart_with_upwards_trend: Generative Visual Manipulation on the Natural Image Manifold (iGAN)|https://github.com/junyanz/iGAN|2775
|7|Deep Convolutional Generative Adversarial Networks (DCGAN)|https://github.com/Newmu/dcgan_code|2750
|8| :chart_with_upwards_trend: cleverhans: A library for benchmarking vulnerability to adversarial examples|https://github.com/openai/cleverhans|1939"
|9| :chart_with_upwards_trend: Wasserstein GAN|https://github.com/martinarjovsky/WassersteinGAN|1784
|10|Neural Photo Editing with Introspective Adversarial Networks|https://github.com/ajbrock/Neural-Photo-Editor|1709
|11|Generative Adversarial Text to Image Synthesis |https://github.com/paarthneekhara/text-to-image|1648
|12|Improved Techniques for Training GANs|https://github.com/openai/improved-gan|1345
|13 :chart_with_upwards_trend: |Improved Training of Wasserstein GANs|https://github.com/igul222/improved_wgan_training|1114
|14|StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks|https://github.com/hanzhanggit/StackGAN|1091
|15|Semantic Image Inpainting with Perceptual and Contextual Losses (2016) |https://github.com/bamos/dcgan-completion.tensorflow|998
|16|HyperGAN|https://github.com/255bits/HyperGAN|759
|17| :chart_with_upwards_trend: Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)|https://github.com/devnag/pytorch-generative-adversarial-networks|758
|18| :chart_with_upwards_trend: Learning to Discover Cross-Domain Relations with Generative Adversarial Networks|https://github.com/carpedm20/DiscoGAN-pytorch|710
|19|Unsupervised Cross-Domain Image Generation|https://github.com/yunjey/domain-transfer-network|652
|20|Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (KERAS-DCGAN)|https://github.com/jacobgil/keras-dcgan|649
|21|Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (The Eyescream Project)|https://github.com/facebook/eyescream|566
|22| :chart_with_upwards_trend: Image-to-image translation using conditional adversarial nets|https://github.com/yenchenlin/pix2pix-tensorflow|548
|23|Generating Videos with Scene Dynamics|https://github.com/cvondrick/videogan|537
|24|Deep multi-scale video prediction beyond mean square error|https://github.com/dyelax/Adversarial_Video_Generation|454
|25|Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space|https://github.com/Evolving-AI-Lab/ppgn|450
|26|Learning from Simulated and Unsupervised Images through Adversarial Training|https://github.com/carpedm20/simulated-unsupervised-tensorflow|448
|27|Synthesizing the preferred inputs for neurons in neural networks via deep generator networks|https://github.com/Evolving-AI-Lab/synthesizing|421
|28| :chart_with_upwards_trend: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling|https://github.com/zck119/3dgan-release|383
|29|A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection|https://github.com/xiaolonw/adversarial-frcnn|362
|30|Conditional Image Synthesis With Auxiliary Classifier GANs|https://github.com/buriburisuri/ac-gan|312
|31|Generating images with recurrent adversarial networks (sequence_gan)|https://github.com/ofirnachum/sequence_gan|298
|32|Learning What and Where to Draw|https://github.com/reedscot/nips2016|294
|33|Adversarially Learned Inference (2016) (ALI)|https://github.com/IshmaelBelghazi/ALI|248
|34|Precomputed real-time texture synthesis with markovian generative adversarial networks|https://github.com/chuanli11/MGANs|235
|35|Autoencoding beyond pixels using a learned similarity metric|https://github.com/andersbll/autoencoding_beyond_pixels|235
|36|Unrolled Generative Adversarial Networks|https://github.com/poolio/unrolled_gan|223
|37|Sampling Generative Networks|https://github.com/dribnet/plat|198
|38|Energy-based generative adversarial network|https://github.com/buriburisuri/ebgan|194
|39|Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network|https://github.com/leehomyc/Photo-Realistic-Super-Resoluton|175
|40|Invertible Conditional GANs for image editing|https://github.com/Guim3/IcGAN|174
|41|Pixel-Level Domain Transfer|https://github.com/fxia22/PixelDTGAN|169
|42|SalGAN: Visual Saliency Prediction with Generative Adversarial Networks|https://github.com/imatge-upc/saliency-salgan-2017|164
|43|Generative face completion (2017)|https://github.com/Yijunmaverick/GenerativeFaceCompletion|163
|44| :chart_with_upwards_trend: C-RNN-GAN: Continuous recurrent neural networks with adversarial training|https://github.com/olofmogren/c-rnn-gan|151
|45|Adversarial Autoencoders |https://github.com/musyoku/adversarial-autoencoder|148
|46|Coupled Generative Adversarial Networks|https://github.com/mingyuliutw/CoGAN|148
|47|Context Encoders: Feature Learning by Inpainting (2016)|https://github.com/jazzsaxmafia/Inpainting|108
|48|Generative Image Modeling using Style and Structure Adversarial Networks (ss-gan)|https://github.com/xiaolonw/ss-gan|100
|49|Conditional Generative Adversarial Nets|https://github.com/zhangqianhui/Conditional-Gans|98
|50|InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets|https://github.com/buriburisuri/supervised_infogan|79
|51|Reconstruction of three-dimensional porous media using generative adversarial neural networks |https://github.com/LukasMosser/PorousMediaGan|29
|52|Improving Generative Adversarial Networks with Denoising Feature Matching|https://github.com/hvy/chainer-gan-denoising-feature-matching|15
|53|Least Squares Generative Adversarial Networks|https://github.com/pfnet-research/chainer-LSGAN|9





----------

### :electric_plug: Frameworks & Libraries (Descending order based on GitHub stars)
- Tensorflow by Google  [C++ and CUDA]: [[homepage]](https://www.tensorflow.org/) [[github]](https://github.com/tensorflow/tensorflow)
- Caffe by Berkeley Vision and Learning Center (BVLC)  [C++]: [[homepage]](http://caffe.berkeleyvision.org/) [[github]](https://github.com/BVLC/caffe) [[Installation Instructions]](Caffe_Installation/README.md)
- Keras by François Chollet  [Python]: [[homepage]](https://keras.io/) [[github]](https://github.com/fchollet/keras)
- Microsoft Cognitive Toolkit - CNTK  [C++]: [[homepage]](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/) [[github]](https://github.com/Microsoft/CNTK)
- MXNet adapted by Amazon  [C++]: [[homepage]](http://mxnet.io/) [[github]](https://github.com/dmlc/mxnet)
- Torch by Collobert, Kavukcuoglu & Clement Farabet, widely used by Facebook  [Lua]: [[homepage]](http://torch.ch/) [[github]](https://github.com/torch) 	
- Convnetjs by Andrej Karpathy [JavaScript]: [[homepage]](http://cs.stanford.edu/people/karpathy/convnetjs/) [[github]](https://github.com/karpathy/convnetjs)
- Theano by Université de Montréal  [Python]: [[homepage]](http://deeplearning.net/software/theano/) [[github]](https://github.com/Theano/Theano) 	
- Deeplearning4j by startup Skymind  [Java]: [[homepage]](https://deeplearning4j.org/) [[github]](https://github.com/deeplearning4j/deeplearning4j) 	
- Caffe2 by Facebook Open Source [C++ & Python]: [[github]](https://github.com/caffe2/caffe2) [[web]](https://caffe2.ai/)
- Paddle by Baidu  [C++]: [[homepage]](http://www.paddlepaddle.org/) [[github]](https://github.com/PaddlePaddle/Paddle)
- Deep Scalable Sparse Tensor Network Engine (DSSTNE) by Amazon  [C++]: [[github]](https://github.com/amznlabs/amazon-dsstne)
- Neon by Nervana Systems  [Python & Sass]: [[homepage]](http://neon.nervanasys.com/docs/latest/) [[github]](https://github.com/NervanaSystems/neon) 	
- Chainer  [Python]: [[homepage]](http://chainer.org/) [[github]](https://github.com/pfnet/chainer) 	
- h2o  [Java]: [[homepage]](http://www.h2o.ai/) [[github]](https://github.com/h2oai/h2o-3) 	
- Brainstorm by Istituto Dalle Molle di Studi sull’Intelligenza Artificiale (IDSIA)  [Python]: [[github]](https://github.com/IDSIA/brainstorm)
- Matconvnet by Andrea Vedaldi  [Matlab]: [[homepage]](http://www.vlfeat.org/matconvnet/) [[github]](https://github.com/vlfeat/matconvnet) 	


----

License

MIT


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [@thomasfuchs]: <http://twitter.com/thomasfuchs>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [keymaster.js]: <https://github.com/madrobby/keymaster>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]:  <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>

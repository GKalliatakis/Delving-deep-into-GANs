# Delving deep into Generative Adversarial Networks (GANs) 	


## A curated, quasi-exhaustive list of state-of-the-art publications and resources about Generative Adversarial Networks (GANs) and their applications.

### Background
Generative models are models that can learn to create data that is similar to data that we give them. One of the most promising approaches of those models are Generative Adversarial Networks (GANs), a branch of unsupervised machine learning implemented by a system of two neural networks competing against each other in a zero-sum game framework. They were first introduced by Ian Goodfellow et al. in 2014. This repository aims at presenting an elaborate list of the state-of-the-art works on the field of Generative Adversarial Networks since their introduction in 2014.

<p align="center">
  <img src="https://raw.githubusercontent.com/GKalliatakis/Delving-deep-into-GANs/master/GAN2.gif?raw=true"/><br>
  Image taken from http://multithreaded.stitchfix.com/blog/2016/02/02/a-fontastic-voyage/<br>
</p>

**_This is going to be an evolving repo and I will keep updating it (at least twice monthly) so make sure you have starred and forked this repository before moving on !_**

---

### :link: Contents
* [Opening Publication](#pushpin-opening-publication)
* [State-of-the-art papers](#clipboard-state-of-the-art-papers-descending-order-based-on-google-scholar-citations)
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

---

### :clipboard: State-of-the-art papers (Descending order based on Google Scholar Citations)

S/N|Paper|Year|Citations
:---:|:---:|:---:|:---:
|1|Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGANs)  [[pdf]](https://arxiv.org/pdf/1511.06434v2.pdf)|2015|901
|2|Explaining and Harnessing Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)|2014|536
|3|Improved Techniques for Training GANs  [[pdf]](https://arxiv.org/pdf/1606.03498v1.pdf )|2016|436
|4|Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (LAPGAN)  [[pdf]](https://arxiv.org/pdf/1506.05751.pdf)|2015|405
|5|Semi-Supervised Learning with Deep Generative Models  [[pdf]](https://arxiv.org/pdf/1406.5298v2.pdf )|2014|367
|6|Conditional Generative Adversarial Nets (CGAN)  [[pdf]](https://arxiv.org/pdf/1411.1784v1.pdf)|2014|321
|7|Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)  [[pdf]](https://arxiv.org/pdf/1611.07004.pdf)|2016|320
|8|Wasserstein GAN (WGAN)  [[pdf]](https://arxiv.org/pdf/1701.07875.pdf)|2017|314
|9|Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)  [[pdf]](https://arxiv.org/pdf/1609.04802.pdf)|2016|281
|10|InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1606.03657)|2016|227
|11|Generative Adversarial Text to Image Synthesis  [[pdf]](https://arxiv.org/pdf/1605.05396)|2016|225
|12|Context Encoders: Feature Learning by Inpainting  [[pdf]](https://arxiv.org/pdf/1604.07379)|2016|217
|13|Deep multi-scale video prediction beyond mean square error  [[pdf]](https://arxiv.org/pdf/1511.05440.pdf)|2015|200
|14|Adversarial Autoencoders  [[pdf]](https://arxiv.org/pdf/1511.05644.pdf)|2015|163
|15|Energy-based Generative Adversarial Network (EBGAN)  [[pdf]](https://arxiv.org/pdf/1609.03126.pdf)|2016|158
|16|Autoencoding beyond pixels using a learned similarity metric (VAE-GAN)  [[pdf]](https://arxiv.org/pdf/1512.09300.pdf)|2015|155
|17|Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)  [[pdf]](https://arxiv.org/pdf/1703.10593.pdf)|2017|153
|18|Adversarial Feature Learning (BiGAN)  [[pdf]](https://arxiv.org/pdf/1605.09782v6.pdf)|2016|142
|19|Conditional Image Generation with PixelCNN Decoders  [[pdf]](https://arxiv.org/pdf/1606.05328.pdf)|2015|140
|20|Improved Training of Wasserstein GANs (WGAN-GP)  [[pdf]](https://arxiv.org/pdf/1704.00028.pdf)|2017|134
|21|Adversarially Learned Inference (ALI)  [[pdf]](https://arxiv.org/pdf/1606.00704.pdf)|2016|129
|22|Towards Principled Methods for Training Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1701.04862.pdf)|2017|121
|23|Generative Moment Matching Networks  [[pdf]](https://arxiv.org/pdf/1502.02761.pdf)|2015|120
|24|f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization  [[pdf]](https://arxiv.org/pdf/1606.00709.pdf)|2016|114
|25|Generating Videos with Scene Dynamics (VGAN)  [[pdf]](http://web.mit.edu/vondrick/tinyvideo/paper.pdf)|2016|114
|26|Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1602.02697.pdf)|2016|103
|27|Generative Visual Manipulation on the Natural Image Manifold (iGAN)  [[pdf]](https://arxiv.org/pdf/1609.03552.pdf)|2016|102
|28|Unsupervised Learning for Physical Interaction through Video Prediction   [[pdf]](https://arxiv.org/pdf/1605.07157)|2016|99
|29|Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (3D-GAN)  [[pdf]](https://arxiv.org/pdf/1610.07584)|2016|99
|30|StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1612.03242.pdf)|2016|99
|31|Generating Images with Perceptual Similarity Metrics based on Deep Networks   [[pdf]](https://arxiv.org/pdf/1602.02644)|2016|96
|32|Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks (CatGAN)  [[pdf]](https://arxiv.org/pdf/1511.06390.pdf)|2015|92
|33|Coupled Generative Adversarial Networks (CoGAN)  [[pdf]](https://arxiv.org/pdf/1606.07536)|2016|90
|34|Conditional Image Synthesis with Auxiliary Classifier GANs (AC-GAN)  [[pdf]](https://arxiv.org/pdf/1610.09585.pdf)|2016|86
|35|Improving Variational Inference with Inverse Autoregressive Flow  [[pdf]](https://arxiv.org/pdf/1606.04934)|2016|82
|36|Generative Image Modeling using Style and Structure Adversarial Networks (S^2GAN)  [[pdf]](https://arxiv.org/pdf/1603.05631.pdf)|2016|82
|37|BEGAN: Boundary Equilibrium Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1703.10717.pdf)|2017|82
|38|Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN) by Apple  [[pdf]](https://arxiv.org/pdf/1612.07828v1.pdf)|2016|80
|39|Unrolled Generative Adversarial Networks (Unrolled GAN)  [[pdf]](https://arxiv.org/pdf/1611.02163.pdf)|2016|75
|40|Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks (MGAN)  [[pdf]](https://arxiv.org/pdf/1604.04382.pdf)|2016|74
|41|SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient    [[pdf]](https://arxiv.org/pdf/1609.05473.pdf)|2016|73
|42|Conditional generative adversarial nets for convolutional face generation [[pdf]](https://pdfs.semanticscholar.org/42f6/f5454dda99d8989f9814989efd50fe807ee8.pdf)|2014|67
|43|Synthesizing the preferred inputs for neurons in neural networks via deep generator networks   [[pdf]](https://arxiv.org/pdf/1605.09304)|2016|64
|44|Unsupervised Cross-Domain Image Generation (DTN)  [[pdf]](https://arxiv.org/pdf/1611.02200.pdf)|2016|62
|45|Training generative neural networks via Maximum Mean Discrepancy optimization  [[pdf]](https://arxiv.org/pdf/1505.03906.pdf)|2015|61
|46|Generative Adversarial Imitation Learning  [[pdf]](https://arxiv.org/pdf/1606.03476)|2016|61
|47|Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space (PPGN)  [[pdf]](https://arxiv.org/pdf/1612.00005.pdf)|2016|58
|48|Least Squares Generative Adversarial Networks (LSGAN)  [[pdf]](https://arxiv.org/pdf/1611.04076.pdf)|2016|58
|49|Generating images with recurrent adversarial networks  [[pdf]](https://arxiv.org/pdf/1602.05110.pdf)|2016|56
|50|Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (DiscoGAN) [[pdf]](https://arxiv.org/pdf/1703.05192.pdf)|2017|54
|51|Amortised MAP Inference for Image Super-resolution (AffGAN)  [[pdf]](https://arxiv.org/pdf/1610.04490.pdf)|2016|52
|52|Adversarial Discriminative Domain Adaptation  [[pdf]](https://arxiv.org/pdf/1702.05464)|2017|52
|53|Mode Regularized Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1612.02136)|2016|50
|54|Learning What and Where to Draw (GAWWN)  [[pdf]](https://arxiv.org/pdf/1610.02454v1.pdf)|2016|49
|55|Semantic Image Inpainting with Perceptual and Contextual Losses   [[pdf]](https://arxiv.org/pdf/1607.07539.pdf)|2016|48
|56|Learning in Implicit Generative Models  [[pdf]](https://arxiv.org/pdf/1610.03483.pdf)|2016|43
|57|Attend, infer, repeat: Fast scene understanding with generative models  [[pdf]](https://arxiv.org/pdf/1603.08575.pdf)|2016|42
|58|Semantic Segmentation using Adversarial Networks   [[pdf]](https://arxiv.org/pdf/1611.08408.pdf)|2016|40
|59|VIME: Variational Information Maximizing Exploration  [[pdf]](https://arxiv.org/pdf/1605.09674)|2016|37
|60|Neural Photo Editing with Introspective Adversarial Networks (IAN)  [[pdf]](https://arxiv.org/pdf/1609.07093.pdf)|2016|37
|61|On the Quantitative Analysis of Decoder-Based Generative Models  [[pdf]](https://arxiv.org/pdf/1611.04273.pdf)|2016|36
|62|Generalization and Equilibrium in Generative Adversarial Nets (GANs)  [[pdf]](https://arxiv.org/pdf/1703.00573)|2017|35
|63|Generalization and Equilibrium in Generative Adversarial Nets (MIX+GAN)  [[pdf]](https://arxiv.org/pdf/1703.00573.pdf)|2016|35
|64|Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro [[pdf]](https://arxiv.org/pdf/1701.07717)|2017|35
|65|Stacked Generative Adversarial Networks (SGAN)  [[pdf]](https://arxiv.org/pdf/1612.04357.pdf)|2016|33
|66|DualGAN: Unsupervised Dual Learning for Image-to-Image Translation    [[pdf]](https://arxiv.org/pdf/1704.02510.pdf)|2017|32
|67|Disentangled Representation Learning GAN for Pose-Invariant Face Recognition   [[pdf]](http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf)|2017|31
|68|Pixel-Level Domain Transfer   [[pdf]](https://arxiv.org/pdf/1603.07442)|2016|30
|69|Full Resolution Image Compression with Recurrent Neural Networks [[pdf]](https://arxiv.org/pdf/1608.05148)|2016|29
|70|Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis (TP-GAN)  [[pdf]](https://arxiv.org/pdf/1704.04086.pdf)|2017|26
|71|Learning a Driving Simulator [[pdf]](https://arxiv.org/pdf/1608.01230)|2016|25
|72|Invertible Conditional GANs for image editing (IcGAN)  [[pdf]](https://arxiv.org/pdf/1611.06355.pdf)|2016|25
|73|Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities (LS-GAN)  [[pdf]](https://arxiv.org/pdf/1701.06264.pdf)|2017|24
|74|Deep and Hierarchical Implicit Models (Bayesian GAN)  [[pdf]](https://arxiv.org/pdf/1702.08896.pdf)|2017|24
|75|Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks  [[pdf]](https://arxiv.org/pdf/1704.01155)|2017|22
|76|Learning to Protect Communications with Adversarial Neural Cryptography [[pdf]](https://arxiv.org/pdf/1610.06918)|2016|21
|77|Adversarial Attacks on Neural Network Policies [[pdf]](https://arxiv.org/pdf/1702.02284.pdf)|2017|21
|78|SEGAN: Speech Enhancement Generative Adversarial Network    [[pdf]](https://arxiv.org/pdf/1703.09452.pdf)|2017|20
|79|Generative Multi-Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1611.01673.pdf)|2016|19
|80|AdaGAN: Boosting Generative Models  [[pdf]](https://arxiv.org/pdf/1701.02386.pdf)|2017|19
|81|Triple Generative Adversarial Nets (Triple-GAN)  [[pdf]](https://arxiv.org/pdf/1703.02291.pdf)|2017|19
|82|MMD GAN: Towards Deeper Understanding of Moment Matching Network [[pdf]](https://arxiv.org/pdf/1705.08584.pdf)|2017|19
|83|Connecting Generative Adversarial Networks and Actor-Critic Methods  [[pdf]](https://arxiv.org/pdf/1610.01945.pdf)|2016|18
|84|Boundary-Seeking Generative Adversarial Networks (BS-GAN)  [[pdf]](https://arxiv.org/pdf/1702.08431.pdf)|2017|18
|85|Adversarial Examples for Semantic Segmentation and Object Detection  [[pdf]](https://arxiv.org/pdf/1703.08603.pdf)|2017|17
|86|A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models  [[pdf]](https://arxiv.org/pdf/1611.03852.pdf)|2016|16
|87|Face Aging With Conditional Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.01983.pdf)|2017|16
|88|Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN (MalGAN)  [[pdf]](https://arxiv.org/pdf/1702.05983.pdf)|2016|15
|89|Generative face completion   [[pdf]](https://arxiv.org/pdf/1704.05838)|2016|15
|90|McGan: Mean and Covariance Feature Matching GAN    [[pdf]](https://arxiv.org/pdf/1702.08398.pdf)|2017|14
|91|Recurrent Topic-Transition GAN for Visual Paragraph Generation (RTT-GAN)  [[pdf]](https://arxiv.org/pdf/1703.07022.pdf)|2017|14
|92|The Cramer Distance as a Solution to Biased Wasserstein Gradients [[pdf]](https://arxiv.org/pdf/1705.10743.pdf)|2017|13
|93|LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation   [[pdf]](https://arxiv.org/pdf/1703.01560.pdf)|2017|13
|94|Maximum-Likelihood Augmented Discrete Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.07983)|2017|13
|95|Temporal Generative Adversarial Nets (TGAN)  [[pdf]](https://arxiv.org/pdf/1611.06624.pdf)|2016|12
|96|C-RNN-GAN: Continuous recurrent neural networks with adversarial training    [[pdf]](https://arxiv.org/pdf/1611.09904.pdf)|2016|12
|97|A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection   [[pdf]](https://arxiv.org/pdf/1704.03414.pdf)|2017|12
|98|Age Progression / Regression by Conditional Adversarial Autoencoder [[pdf]](https://arxiv.org/pdf/1702.08423)|2017|12
|99|The Space of Transferable Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1704.03453)|2017|12
|100|MoCoGAN: Decomposing Motion and Content for Video Generation [[pdf]](https://arxiv.org/pdf/1707.04993.pdf)|2017|11
|101|Semantic Image Inpainting with Deep Generative Models [[pdf]](https://arxiv.org/pdf/1607.07539.pdf)|2017|11
|102|Imitating Driver Behavior with Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1701.06699)|2017|11
|103|Image De-raining Using a Conditional Generative Adversarial Network (ID-CGAN)  [[pdf]](https://arxiv.org/pdf/1701.05957)|2017|11
|104|SalGAN: Visual Saliency Prediction with Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1701.01081.pdf)|2016|11
|105|Multi-View Image Generation from a Single-View   [[pdf]](https://arxiv.org/pdf/1704.04886)|2016|11
|106|CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1705.02355)|2017|11
|107|Adversarial Transformation Networks: Learning to Generate Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1703.09387)|2017|11
|108|Improved Semi-supervised Learning with GANs using Manifold Invariances [[pdf]](https://arxiv.org/abs/1705.08850)|2017|11
|109|Neural Face Editing with Intrinsic Image Disentangling  [[pdf]](https://arxiv.org/abs/1704.04131)|2017|11
|110|Cooperative Training of Descriptor and Generator Networks [[pdf]](https://arxiv.org/pdf/1609.09408.pdf)|2016|10
|111|Cooperative Training of Descriptor and Generator Network  [[pdf]](https://arxiv.org/pdf/1609.09408)|2016|10
|112|Unsupervised Image-to-Image Translation with Generative Adversarial Networks   [[pdf]](https://arxiv.org/pdf/1701.02676.pdf)|2017|10
|113|Improved generator objectives for GANs  [[pdf]](https://arxiv.org/pdf/1612.02780.pdf)|2017|10
|114|Inverting The Generator Of A Generative Adversarial Network  [[pdf]](https://arxiv.org/pdf/1611.05644)|2016|10
|115|Simple Black-Box Adversarial Perturbations for Deep Networks  [[pdf]](https://arxiv.org/pdf/1612.06299)|2016|10
|116| Perceptual Generative Adversarial Networks for Small Object Detection [[pdf]](https://arxiv.org/abs/1706.05274)|2017|10
|117|It Takes (Only) Two: Adversarial Generator-Encoder Networks [[pdf]](https://arxiv.org/pdf/1704.02304.pdf)|2017|9
|118|On Convergence and Stability of GANs  [[pdf]](https://arxiv.org/pdf/1705.07215.pdf)|2017|9
|119|Deep Generative Adversarial Networks for Compressed Sensing (GANCS) Automates MRI  [[pdf]](https://arxiv.org/pdf/1706.00051.pdf)|2017|9
|120|3D Shape Induction from 2D Views of Multiple Objects (PrGAN)  [[pdf]](https://arxiv.org/pdf/1612.05872.pdf)|2016|9
|121|Precise Recovery of Latent Vectors from Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.04782)|2016|9
|122|Adversarial Training Methods for Semi-Supervised Text Classification  [[pdf]](https://arxiv.org/abs/1605.07725)|2016|9
|123|Towards Diverse and Natural Image Descriptions via a Conditional GAN [[pdf]](https://arxiv.org/pdf/1703.06029.pdf)|2017|9
|124|Adversarial Generator-Encoder Networks  [[pdf]](https://arxiv.org/pdf/1704.02304)|2017|9
|125|How to Train Your DRAGAN [[pdf]](https://arxiv.org/abs/1705.07215)|2017|9
|126|Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks (SSL-GAN)  [[pdf]](https://arxiv.org/pdf/1611.06430.pdf)|2016|8
|127|MAGAN: Margin Adaptation for Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1704.03817.pdf)|2017|8
|128|Towards Large-Pose Face Frontalization in the Wild   [[pdf]](https://arxiv.org/pdf/1704.06244)|2016|8
|129|Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1704.00849.pdf)|2017|8
|130|Comparison of Maximum Likelihood and GAN-based training of Real NVPs [[pdf]](https://arxiv.org/pdf/1705.05263.pdf)|2017|7
|131|Pose Guided Person Image Generation [[pdf]](https://arxiv.org/pdf/1705.09368.pdf)|2017|7
|132|Adversarial Deep Structural Networks for Mammographic Mass Segmentation  [[pdf]](https://arxiv.org/abs/1612.05970)|2017|7
|133|Learning to Generate Images of Outdoor Scenes from Attributes and Semantic Layouts (AL-CGAN)  [[pdf]](https://arxiv.org/pdf/1612.00215.pdf)|2016|7
|134|RenderGAN: Generating Realistic Labeled Data    [[pdf]](https://arxiv.org/pdf/1611.01331.pdf)|2016|7
|135|Generative Temporal Models with Memory  [[pdf]](https://arxiv.org/pdf/1702.04649.pdf)|2017|7
|136|Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation [[pdf]](https://arxiv.org/pdf/1705.00389.pdf)|2017|7
|137|Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.01705)|2017|7
|138|Universal Adversarial Perturbations Against Semantic Image Segmentation  [[pdf]](https://arxiv.org/pdf/1704.05712)|2017|7
|139|Good Semi-supervised Learning that Requires a Bad GAN [[pdf]](https://arxiv.org/abs/1705.09783)|2017|7
|140|DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data  [[pdf]](https://arxiv.org/abs/1706.02071)|2017|7
|141|A General Retraining Framework for Scalable Adversarial Classification [[pdf]](https://arxiv.org/pdf/1604.02606.pdf)|2016|6
|142|Contextual RNN-GANs for Abstract Reasoning Diagram Generation (Context-RNN-GAN)  [[pdf]](https://arxiv.org/pdf/1609.09444.pdf)|2016|6
|143|Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (AnoGAN)  [[pdf]](https://arxiv.org/pdf/1703.05921.pdf)|2017|6
|144|Multi-Agent Diverse Generative Adversarial Networks (MAD-GAN)  [[pdf]](https://arxiv.org/pdf/1704.02906.pdf)|2017|6
|145|Generative Adversarial Residual Pairwise Networks for One Shot Learning  [[pdf]](https://arxiv.org/pdf/1703.08033)|2017|6
|146|Reconstruction of three-dimensional porous media using generative adversarial neural networks [[pdf]](https://arxiv.org/pdf/1704.03225)|2017|6
|147|Learning Representations of Emotional Speech with Deep Convolutional Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1705.02394)|2017|6
|148|Adversarial Image Perturbation for Privacy Protection--A Game Theory Perspective  [[pdf]](https://arxiv.org/pdf/1703.09471)|2017|6
|149|SCAN: Structure Correcting Adversarial Network for Chest X-rays Organ Segmentation  [[pdf]](https://arxiv.org/pdf/1703.08770)|2017|6
|150|Conditional CycleGAN for Attribute Guided Face Image Generation [[pdf]](https://arxiv.org/abs/1705.09966)|2017|6
|151|Stabilizing Training of Generative Adversarial Networks through Regularization [[pdf]](https://arxiv.org/abs/1705.09367)|2017|6
|152|Interactive 3D Modeling with a Generative Adversarial Network [[pdf]](https://arxiv.org/abs/1706.05170)|2017|6
|153|ALICE: Towards Understanding Adversarial Learning for Joint Distribution Matching [[pdf]](http://papers.nips.cc/paper/7133-alice-towards-understanding-adversarial-learning-for-joint-distribution-matching.pdf)|2017|5
|154|Crossing Nets: Combining GANs and VAEs with a Shared Latent Space for Hand Pose Estimation [[pdf]](https://arxiv.org/pdf/1702.03431.pdf)|2017|5
|155|Dual Motion GAN for Future-Flow Embedded Video Prediction [[pdf]](https://arxiv.org/pdf/1708.00284.pdf)|2017|5
|156|Texture Synthesis with Spatial Generative Adversarial Networks (SGAN)  [[pdf]](https://arxiv.org/pdf/1611.08207.pdf)|2016|5
|157|Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking (GoGAN)  [[pdf]](https://arxiv.org/pdf/1704.04865.pdf)|2017|5
|158|Generating Multi-label Discrete Electronic Health Records using Generative Adversarial Networks (MedGAN)  [[pdf]](https://arxiv.org/pdf/1703.06490.pdf)|2017|5
|159|Semi-Latent GAN: Learning to generate and modify facial images from attributes (SL-GAN)   [[pdf]](https://arxiv.org/pdf/1704.02166.pdf)|2017|5
|160|From source to target and back: symmetric bi-directional adaptive GAN [[pdf]](https://arxiv.org/abs/1705.08824)|2017|5
|161|Semantically Decomposing the Latent Spaces of Generative Adversarial Networks (SD-GAN) [[pdf]](https://arxiv.org/abs/1705.07904)|2017|5
|162|Objective-Reinforced Generative Adversarial Networks (ORGAN) [[pdf]](https://arxiv.org/pdf/1705.10843.pdf)|2017|5
|163| From source to target and back: symmetric bi-directional adaptive GAN [[pdf]](https://arxiv.org/abs/1705.08824)|2017|5
|164|Multi-Generator Gernerative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1708.02556.pdf)|2017|4
|165|Optimizing the Latent Space of Generative Networks [[pdf]](https://arxiv.org/pdf/1707.05776.pdf)|2017|4
|166|Robust LSTM-Autoencoders for Face De-Occlusion in the Wild   [[pdf]](https://arxiv.org/pdf/1612.08534.pdf)|2016|4
|167|Ensembles of Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1612.00991.pdf)|2016|4
|168|Adversarial Training For Sketch Retrieval (SketchGAN)  [[pdf]](https://arxiv.org/pdf/1607.02748.pdf)|2016|4
|169|Message Passing Multi-Agent GANs (MPM-GAN)  [[pdf]](https://arxiv.org/pdf/1612.01294.pdf)|2017|4
|170|Steganographic Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1703.05502)|2017|4
|171|GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data  [[pdf]](https://arxiv.org/pdf/1705.04932.pdf)|2017|4
|172|Relaxed Wasserstein with Applications to GANs (RWGAN ) [[pdf]](https://arxiv.org/abs/1705.07164)|2017|4
|173|VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning () [[pdf]](https://arxiv.org/abs/1705.07761)|2017|4
|174|Adversarial Generation of Natural Language [[pdf]](https://arxiv.org/abs/1705.10929)|2017|4
|175|SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation  [[pdf]](https://arxiv.org/abs/1706.01805)|2017|4
|176|Language Generation with Recurrent Generative Adversarial Networks without Pre-training  [[pdf]](https://arxiv.org/abs/1706.01399)|2017|4
|177|Perceptual Adversarial Networks for Image-to-Image Transformation  [[pdf]](https://arxiv.org/pdf/1706.09138)|2017|4
|178|Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models [[pdf]](https://arxiv.org/pdf/1705.10843.pdf)|2017|3
|179|PixelGAN Autoencoders [[pdf]](https://arxiv.org/pdf/1706.00531.pdf)|2017|3
|180|GANs for Biological Image Synthesis [[pdf]](https://arxiv.org/pdf/1708.04692.pdf)|2017|3
|181|Megapixel Size Image Creation using Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1706.00082.pdf)|2017|3
|182|Unsupervised Diverse Colorization via Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.06674.pdf)|2017|3
|183|GP-GAN: Towards Realistic High-Resolution Image Blending   [[pdf]](https://arxiv.org/pdf/1703.07195.pdf)|2017|3
|184|CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training   [[pdf]](https://arxiv.org/pdf/1703.10155.pdf)|2017|3
|185|Generative Adversarial Networks as Variational Training of Energy Based Models (VGAN)  [[pdf]](https://arxiv.org/pdf/1611.01799.pdf)|2017|3
|186|Generative Adversarial Parallelization  [[pdf]](https://arxiv.org/pdf/1612.04021)|2015|3
|187|Adversarial Networks for the Detection of Aggressive Prostate Cancer [[pdf]](https://arxiv.org/pdf/1702.08014)|2017|3
|188|Flow-GAN: Bridging implicit and prescribed learning in generative models [[pdf]](https://arxiv.org/abs/1705.08868)|2017|3
|189|Bayesian GAN [[pdf]](https://arxiv.org/abs/1705.09558)|2017|3
|190|Weakly Supervised Generative Adversarial Networks for 3D Reconstruction  [[pdf]](https://arxiv.org/abs/1705.10904)|2017|3
|191|Style Transfer for Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN  [[pdf]](https://arxiv.org/abs/1706.03319)|2017|3
|192|Auto-Encoder Guided GAN for Chinese Calligraphy Synthesis  [[pdf]](https://arxiv.org/abs/1706.08789)|2017|3
|193|SeGAN: Segmenting and Generating the Invisible   [[pdf]](https://arxiv.org/pdf/1703.10239.pdf)|2017|2
|194|Activation Maximization Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1703.02000.pdf)|2017|2
|195|Bayesian GAN [[pdf]](https://arxiv.org/pdf/1705.09558.pdf)|2017|2
|196|AlignGAN: Learning to Align Cross-Domain Images with Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1707.01400.pdf)|2017|2
|197|ExprGAN: Facial Expression Editing with Controllable Expression Intensity [[pdf]](https://arxiv.org/pdf/1709.03842.pdf)|2017|2
|198|Semantic Image Synthesis via Adversarial Learning [[pdf]](https://arxiv.org/pdf/1707.06873.pdf)|2017|2
|199|StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1710.10916.pdf)|2017|2
|200|Generative Adversarial Nets with Labeled Data by Activation Maximization (AMGAN)  [[pdf]](https://arxiv.org/pdf/1703.02000.pdf)|2017|2
|201|WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images    [[pdf]](https://arxiv.org/pdf/1702.07392.pdf)|2017|2
|202|Multi-view Generative Adversarial Networks (MV-BiGAN)  [[pdf]](https://arxiv.org/pdf/1611.02019.pdf)|2017|2
|203|MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation using 1D and 2D Conditions   [[pdf]](https://arxiv.org/pdf/1703.10847.pdf)|2016|2
|204|Softmax GAN [[pdf]](https://arxiv.org/pdf/1704.06191)|2017|2
|205|Auto-painter: Cartoon Image Generation from Sketch by Using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1705.01908.pdf)|2017|2
|206|Outline Colorization through Tandem Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.08834.pdf)|2017|2
|207| Learning Texture Manifolds with the Periodic Spatial GAN [[pdf]](https://arxiv.org/abs/1705.06566)|2017|2
|208|Generate Identity-Preserving Faces by Generative Adversarial Networks  [[pdf]](https://arxiv.org/abs/1706.03227)|2017|2
|209|Retinal Vessel Segmentation in Fundoscopic Images with Generative Adversarial Networks  [[pdf]](https://arxiv.org/abs/1706.09318)|2017|2
|210|SAD-GAN: Synthetic Autonomous Driving using Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1611.08788.pdf)|2017|1
|211|CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training  [[pdf]](https://arxiv.org/pdf/1709.02023.pdf)|2017|1
|212|Class-Splitting Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.07359.pdf)|2017|1
|213|Adversarial Generation of Training Examples for Vehicle License Plate Recognition [[pdf]](https://arxiv.org/pdf/1707.03124.pdf)|2017|1
|214|Aesthetic-Driven Image Enhancement by Adversarial Learning [[pdf]](https://arxiv.org/pdf/1707.05251.pdf)|2017|1
|215|Automatic Liver Segmentation Using an Adversarial Image-to-Image Network [[pdf]](https://arxiv.org/pdf/1707.08037.pdf)|2017|1
|216|Compressed Sensing MRI Reconstruction with Cyclic Loss in Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.00753.pdf)|2017|1
|217|Creatism: A deep-learning photographer capable of creating professional work [[pdf]](https://arxiv.org/pdf/1707.03491.pdf%20)|2017|1
|218|Generative Adversarial Models for People Attribute Recognition in Surveillance [[pdf]](https://arxiv.org/pdf/1707.02240.pdf)|2017|1
|219|Generative Adversarial Network-based Synthesis of Visible Faces from Polarimetric Thermal Faces [[pdf]](https://arxiv.org/pdf/1708.02681.pdf)|2017|1
|220|Guiding InfoGAN with Semi-Supervision [[pdf]](https://arxiv.org/pdf/1707.04487.pdf)|2017|1
|221|High-Quality Facial Photo-Sketch Synthesis Using Multi-Adversarial Networks [[pdf]](https://arxiv.org/pdf/1710.10182.pdf)|2017|1
|222|Improving Heterogeneous Face Recognition with Conditional Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.02848.pdf)|2017|1
|223|Intraoperative Organ Motion Models with an Ensemble of Conditional Generative Adversarial Networks  [[pdf]](https://arxiv.org/ftp/arxiv/papers/1709/1709.02255.pdf)|2017|1
|224|Label Denoising Adversarial Network (LDAN) for Inverse Lighting of Face Images [[pdf]](https://arxiv.org/pdf/1709.01993.pdf)|2017|1
|225|Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss  [[pdf]](https://arxiv.org/pdf/1708.00961.pdf)|2017|1
|226|MARTA GANs: Unsupervised Representation Learning for Remote Sensing Image Classification [[pdf]](https://arxiv.org/pdf/1612.08879.pdf)|2017|1
|227|Representation Learning and Adversarial Generation of 3D Point Clouds [[pdf]](https://arxiv.org/pdf/1707.02392.pdf)|2017|1
|228|ArtGAN: Artwork Synthesis with Conditional Categorial GANs   [[pdf]](https://arxiv.org/pdf/1702.03410.pdf)|2017|1
|229|Deep Unsupervised Representation Learning for Remote Sensing Images (MARTA-GAN)    [[pdf]](https://arxiv.org/pdf/1612.08879.pdf)|2017|1
|230|TAC-GAN - Text Conditioned Auxiliary Classifier Generative Adversarial Network    [[pdf]](https://arxiv.org/pdf/1703.06412.pdf)|2017|1
|231|Image Generation and Editing with Variational Info Generative Adversarial Networks (ViGAN)  [[pdf]](https://arxiv.org/pdf/1701.04568.pdf)|2016|1
|232|Associative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1611.06953)|2017|1
|233|Generative Adversarial Structured Networks  [[pdf]](https://sites.google.com/site/nips2016adversarial/WAT16_paper_14.pdf)|2017|1
|234|Supervised Adversarial Networks for Image Saliency Detection [[pdf]](https://arxiv.org/pdf/1704.07242)|2017|1
|235|Deep Generative Adversarial Compression Artifact Removal  [[pdf]](https://arxiv.org/pdf/1704.02518)|2017|1
|236|Training Triplet Networks with GAN  [[pdf]](https://arxiv.org/pdf/1704.02227)|2017|1
|237|Geometric GAN  [[pdf]](https://arxiv.org/pdf/1705.02894.pdf)|2017|1
|238|Face Super-Resolution Through Wasserstein GANs  [[pdf]](https://arxiv.org/pdf/1705.02438)|2017|1
|239|Training Triplet Networks with GAN  [[pdf]](https://arxiv.org/pdf/1704.02227)|2017|1
|240|Generative Adversarial Networks for Multimodal Representation Learning in Video Hyperlinking [[pdf]](https://arxiv.org/abs/1705.05103)|2017|1
|241|Continual Learning in Generative Adversarial Nets [[pdf]](https://arxiv.org/abs/1705.08395)|2017|1
|242|TextureGAN: Controlling Deep Image Synthesis with Texture Patches  [[pdf]](https://arxiv.org/abs/1706.02823)|2017|1
|243|Gradient descent GAN optimization is locally stable  [[pdf]](https://arxiv.org/abs/1706.04156)|2017|1
|244|CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms [[pdf]](https://arxiv.org/pdf/1706.07068.pdf)|2017|1
|245|Generative Mixture of Networks  [[pdf]](https://arxiv.org/pdf/1702.03307.pdf)|2017|0
|246|Stopping GAN Violence: Generative Unadversarial Networks  [[pdf]](https://arxiv.org/pdf/1703.02528.pdf)|2016|0
|247|An Adversarial Regularisation for Semi-Supervised Training of Structured Output Neural Networks  [[pdf]](https://arxiv.org/pdf/1702.02382)|2017|0
|248|On the effect of Batch Normalization and Weight Normalization in Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.03971)|2017|0
|249|Generative Cooperative Net for Image Generation and Data Augmentation  [[pdf]](https://arxiv.org/pdf/1705.02887.pdf)|2017|0
|250|Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN  [[pdf]](https://arxiv.org/pdf/1705.03387)|2017|0
|251|Depth Structure Preserving Scene Image Generation  [[pdf]](https://arxiv.org/abs/1706.00212)|2017|0
|252|Synthesizing Filamentary Structured Images with GANs  [[pdf]](https://arxiv.org/abs/1706.02185)|2017|0
|253|A Classification-Based Perspective on GAN Distributions [[pdf]](https://arxiv.org/pdf/1711.00970.pdf)|2017|0
|254|APE-GAN: Adversarial Perturbation Elimination with GAN [[pdf]](https://arxiv.org/pdf/1707.05474.pdf)|2017|0
|255|Bayesian Conditional Generative Adverserial Networks [[pdf]](https://arxiv.org/pdf/1706.05477.pdf)|2017|0
|256|Binary Generative Adversarial Networks for Image Retrieval [[pdf]](https://arxiv.org/pdf/1708.04150.pdf)|2017|0
|257|CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning [[pdf]](https://arxiv.org/pdf/1710.05106.pdf)|2017|0
|258|Dualing GANs [[pdf]](https://arxiv.org/pdf/1706.06216.pdf)|2017|0
|259|Generative Adversarial Networks with Inverse Transformation Unit [[pdf]](https://arxiv.org/pdf/1709.09354.pdf)|2017|0
|260|Generative Semantic Manipulation with Contrasting GAN [[pdf]](https://arxiv.org/pdf/1708.00315.pdf)|2017|0
|261|Image Quality Assessment Techniques Show Improved Training and Evaluation of Autoencoder Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1708.02237.pdf)|2017|0
|262|KGAN: How to Break The Minimax Game in GAN  [[pdf]](https://arxiv.org/pdf/1711.01744.pdf)|2017|0
|263|Learning Loss for Knowledge Distillation with Conditional Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.00513.pdf)|2017|0
|264|Linking Generative Adversarial Learning and Binary Classification [[pdf]](https://arxiv.org/pdf/1709.01509.pdf)|2017|0
|265|Parametrizing filters of a CNN with a GAN [[pdf]](https://arxiv.org/pdf/1710.11386.pdf)|2017|0
|266|Statistics of Deep Generated Images [[pdf]](https://arxiv.org/pdf/1708.02688.pdf)|2017|0
|267|Structured Generative Adversarial Networks  [[pdf]](http://papers.nips.cc/paper/6979-structured-generative-adversarial-networks.pdf)|2017|0
|268|Tensorizing Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1710.10772.pdf)|2017|0
|269|3D Object Reconstruction from a Single Depth View with Adversarial Learning [[pdf]](https://arxiv.org/pdf/1708.07969.pdf)|2017|0
|270|A step towards procedural terrain generation with GANs [[pdf]](https://arxiv.org/pdf/1707.03383.pdf)|2017|0
|271|Abnormal Event Detection in Videos using Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1708.09644.pdf)|2017|0
|272|Adversarial nets with perceptual losses for text-to-image synthesis [[pdf]](https://arxiv.org/pdf/1708.09321.pdf)|2017|0
|273|Adversarial Networks for Spatial Context-Aware Spectral Image Reconstruction from RGB [[pdf]](https://arxiv.org/pdf/1709.00265.pdf)|2017|0
|274|A Novel Approach to Artistic Textual Visualization via GAN [[pdf]](https://arxiv.org/pdf/1710.10553.pdf)|2017|0
|275|Anti-Makeup: Learning A Bi-Level Adversarial Network for Makeup-Invariant Face Verification [[pdf]](https://arxiv.org/pdf/1709.03654.pdf)|2017|0
|276|ARIGAN: Synthetic Arabidopsis Plants using Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1709.00938.pdf)|2017|0
|277|Artificial Generation of Big Data for Improving Image Classification: A Generative Adversarial Network Approach on SAR Data [[pdf]](https://arxiv.org/pdf/1711.02010.pdf)|2017|0
|278|Conditional Adversarial Network for Semantic Segmentation of Brain Tumor [[pdf]](https://arxiv.org/pdf/1708.05227)|2017|0
|279|Controllable Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1708.00598.pdf)|2017|0
|280|Data Augmentation in Classification using GAN [[pdf]](https://arxiv.org/pdf/1711.00648.pdf)|2017|0
|281|Deep Generative Adversarial Neural Networks for Realistic Prostate Lesion MRI Synthesis [[pdf]](https://arxiv.org/pdf/1708.00129.pdf)|2017|0
|282|Face Transfer with Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1710.06090.pdf)|2017|0
|283|Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1710.04835.pdf)|2017|0
|284|Freehand Ultrasound Image Simulation with Spatially-Conditioned Generative Adversarial Networks [[pdf]](https://arxiv.org/ftp/arxiv/papers/1707/1707.05392.pdf)|2017|0
|285|Generative Adversarial Network based on Resnet for Conditional Image Restoration  [[pdf]](https://arxiv.org/pdf/1707.04881.pdf)|2017|0
|286|GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks [[pdf]](https://arxiv.org/pdf/1710.00962.pdf)|2017|0
|287|How to Fool Radiologists with Generative Adversarial Networks? A Visual Turing Test for Lung Cancer Diagnosis [[pdf]](https://arxiv.org/pdf/1710.09762.pdf)|2017|0
|288|Hierarchical Detail Enhancing Mesh-Based Shape Generation with 3D Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1709.07581.pdf)|2017|0
|289|High-Quality Face Image SR Using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1707.00737.pdf)|2017|0
|290|Improved Adversarial Systems for 3D Object Generation and Reconstruction  [[pdf]](https://arxiv.org/pdf/1707.09557.pdf)|2017|0
|291|Improving image generative models with human interactions [[pdf]](https://arxiv.org/pdf/1709.10459.pdf)|2017|0
|292|Learning a Generative Adversarial Network for High Resolution Artwork Synthesis [[pdf]](https://arxiv.org/pdf/1708.09533.pdf)|2017|0
|293|Learning to Generate Chairs with Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1705.10413.pdf)|2017|0
|294|Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1709.07592.pdf)|2017|0
|295|Microscopy Cell Segmentation via Adversarial Neural Networks  [[pdf]](https://arxiv.org/pdf/1709.05860.pdf)|2017|0
|296|Neural Stain-Style Transfer Learning using GAN for Histopathological Images [[pdf]](https://arxiv.org/pdf/1710.08543.pdf)|2017|0
|297|Retinal Vasculature Segmentation Using Local Saliency Maps and Generative Adversarial Networks For Image Super Resolution [[pdf]](https://arxiv.org/pdf/1710.04783.pdf)|2017|0
|298|Sharpness-aware Low dose CT denoising using conditional generative adversarial network [[pdf]](https://arxiv.org/pdf/1708.06453.pdf)|2017|0
|299|Simultaneously Color-Depth Super-Resolution with Conditional Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1708.09105.pdf)|2017|0
|300|Socially-compliant Navigation through Raw Depth Inputs with Generative Adversarial Imitation Learning [[pdf]](https://arxiv.org/pdf/1710.02543.pdf)|2017|0







----------

### :notebook_with_decorative_cover: Theory
- Improved Techniques for Training GANs [[pdf]](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans.pdf) 	
- Energy-Based GANs & other Adversarial things by Yann Le Cun [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Mode RegularizedGenerative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1612.02136.pdf)

----------

### :nut_and_bolt: Presentations
- Generative Adversarial Networks (GANs) by Ian Goodfellow [[pdf]](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf) 	
- Learning Deep Generative Models by Russ Salakhutdinov [[pdf]](http://www.cs.toronto.edu/~rsalakhu/talk_Montreal_2016_Salakhutdinov.pdf) 	

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
|1|Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)|https://github.com/junyanz/CycleGAN|5302
|2|Image super-resolution through deep learning|https://github.com/david-gpu/srez|4631
|3|Image-to-image translation with conditional adversarial nets (pix2pix)|https://github.com/phillipi/pix2pix|4172
|4|Τensorflow implementation of Deep Convolutional Generative Adversarial Networks (DCGAN)|https://github.com/carpedm20/DCGAN-tensorflow|3378
|5|Deep Convolutional Generative Adversarial Networks (DCGAN)|https://github.com/Newmu/dcgan_code|2572
|6|Generative Models: Collection of generative models, e.g. GAN, VAE in Pytorch and Tensorflow|https://github.com/wiseodd/generative-models|2521
|7|Generative Visual Manipulation on the Natural Image Manifold (iGAN)|https://github.com/junyanz/iGAN|2353
|8|Neural Photo Editing with Introspective Adversarial Networks|https://github.com/ajbrock/Neural-Photo-Editor|1604
|9|Wasserstein GAN|https://github.com/martinarjovsky/WassersteinGAN|1526
|10|Generative Adversarial Text to Image Synthesis |https://github.com/paarthneekhara/text-to-image|1508
|11|cleverhans: A library for benchmarking vulnerability to adversarial examples|https://github.com/openai/cleverhans|1475"
|12|Improved Techniques for Training GANs|https://github.com/openai/improved-gan|1140
|13|StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks|https://github.com/hanzhanggit/StackGAN|945
|14|Semantic Image Inpainting with Perceptual and Contextual Losses (2016) |https://github.com/bamos/dcgan-completion.tensorflow|870
|15|Improved Training of Wasserstein GANs|https://github.com/igul222/improved_wgan_training|843
|16|HyperGAN|https://github.com/255bits/HyperGAN|679
|17|Unsupervised Cross-Domain Image Generation|https://github.com/yunjey/domain-transfer-network|600
|18|Learning to Discover Cross-Domain Relations with Generative Adversarial Networks|https://github.com/carpedm20/DiscoGAN-pytorch|593
|19|Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)|https://github.com/devnag/pytorch-generative-adversarial-networks|574
|20|Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (KERAS-DCGAN)|https://github.com/jacobgil/keras-dcgan|555
|21|Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (The Eyescream Project)|https://github.com/facebook/eyescream|546
|22|Generating Videos with Scene Dynamics|https://github.com/cvondrick/videogan|501
|23|Image-to-image translation using conditional adversarial nets|https://github.com/yenchenlin/pix2pix-tensorflow|454
|24|Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space|https://github.com/Evolving-AI-Lab/ppgn|421
|25|Synthesizing the preferred inputs for neurons in neural networks via deep generator networks|https://github.com/Evolving-AI-Lab/synthesizing|402
|26|Learning from Simulated and Unsupervised Images through Adversarial Training|https://github.com/carpedm20/simulated-unsupervised-tensorflow|401
|27|Deep multi-scale video prediction beyond mean square error|https://github.com/dyelax/Adversarial_Video_Generation|380
|28|A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection|https://github.com/xiaolonw/adversarial-frcnn|302
|29|Conditional Image Synthesis With Auxiliary Classifier GANs|https://github.com/buriburisuri/ac-gan|285
|30|Learning What and Where to Draw|https://github.com/reedscot/nips2016|276
|31|Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling|https://github.com/zck119/3dgan-release|274
|32|Generating images with recurrent adversarial networks (sequence_gan)|https://github.com/ofirnachum/sequence_gan|260
|33|Adversarially Learned Inference (2016) (ALI)|https://github.com/IshmaelBelghazi/ALI|230
|34|Precomputed real-time texture synthesis with markovian generative adversarial networks|https://github.com/chuanli11/MGANs|219
|35|Unrolled Generative Adversarial Networks|https://github.com/poolio/unrolled_gan|209
|36|Autoencoding beyond pixels using a learned similarity metric|https://github.com/andersbll/autoencoding_beyond_pixels|195
|37|Energy-based generative adversarial network|https://github.com/buriburisuri/ebgan|194
|38|Sampling Generative Networks|https://github.com/dribnet/plat|176
|39|Pixel-Level Domain Transfer|https://github.com/fxia22/PixelDTGAN|161
|40|Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network|https://github.com/leehomyc/Photo-Realistic-Super-Resoluton|160
|41|Invertible Conditional GANs for image editing|https://github.com/Guim3/IcGAN|156
|42|Adversarial Autoencoders |https://github.com/musyoku/adversarial-autoencoder|129
|43|SalGAN: Visual Saliency Prediction with Generative Adversarial Networks|https://github.com/imatge-upc/saliency-salgan-2017|129
|44|Generative face completion (2017)|https://github.com/Yijunmaverick/GenerativeFaceCompletion|110
|45|InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets|https://github.com/buriburisuri/supervised_infogan|108
|46|C-RNN-GAN: Continuous recurrent neural networks with adversarial training|https://github.com/olofmogren/c-rnn-gan|108
|47|Coupled Generative Adversarial Networks|https://github.com/mingyuliutw/CoGAN|99
|48|Generative Image Modeling using Style and Structure Adversarial Networks (ss-gan)|https://github.com/xiaolonw/ss-gan|95
|49|Context Encoders: Feature Learning by Inpainting (2016)|https://github.com/jazzsaxmafia/Inpainting|87
|50|Conditional Generative Adversarial Nets|https://github.com/zhangqianhui/Conditional-Gans|68
|51|Reconstruction of three-dimensional porous media using generative adversarial neural networks |https://github.com/LukasMosser/PorousMediaGan|18
|52|Improving Generative Adversarial Networks with Denoising Feature Matching|https://github.com/hvy/chainer-gan-denoising-feature-matching|10
|53|Least Squares Generative Adversarial Networks|https://github.com/pfnet-research/chainer-LSGAN|8




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

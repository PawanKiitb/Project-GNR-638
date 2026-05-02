import argparse
import os
import re
import pandas as pd
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==========================================
# THE EXAM PROMPT
# ==========================================
FINAL_PROMPT = """You are an expert PyTorch developer and Deep Learning researcher. 
Analyze the multiple-choice question in the provided image.

[Materials: # Deep Learning MCQ Prompt Notes

## 1) Image features and traditional vision
- Images are affected by illumination, blur, sensor noise, compression, lens effects, and viewpoint change. The point of classical features is to keep the useful structure and suppress nuisance variation.
- Convolutional filtering:
  \[
  g(x,y)=\sum_u\sum_v k(u,v)\,f(x-u,y-v)
  \]
  In DL, the operation is usually cross-correlation, i.e. the kernel is not flipped.
- Common visual feature families:
  edges / gradients, texture, keypoints, and global descriptors.
- HOG-style descriptors are robust to small brightness / contrast changes because they work on gradient directions and block normalization.
- GLCM texture idea: count how often gray level \(i\) occurs next to gray level \(j\) at a fixed offset; from it one uses contrast, homogeneity, energy/ASM, and correlation.
- Bag of visual words:
  1. detect keypoints,
  2. extract local patches / descriptors,
  3. quantize to a codebook,
  4. build a histogram over codewords.
  The key MCQ trap is that BoVW keeps "what" appears but discards "where" it appears.

## 2) Neural networks: perceptron, MLP, softmax, autodiff
- A neuron computes
  \[
  z=w^\top x+b,\qquad a=\phi(z)
  \]
  where \(\phi\) can be sigmoid, tanh, ReLU, etc.
- A feed-forward network stacks layers:
  \[
  a^{(l)}=\phi\!\big(W^{(l)}a^{(l-1)}+b^{(l)}\big)
  \]
- Universal approximation theorem: a feed-forward network with at least one hidden layer, enough hidden units, and a nonlinear activation can approximate any continuous function on a compact set arbitrarily well.
  Important caveat: this says nothing about trainability, sample efficiency, or generalization.
- Softmax for logits \(g_k\):
  \[
  p_k=\frac{e^{g_k}}{\sum_j e^{g_j}}
  \]
- Multiclass cross-entropy for one-hot target \(y\):
  \[
  L=-\sum_k y_k \log p_k
  \]
- For binary classification:
  \[
  p=\sigma(z)=\frac{1}{1+e^{-z}},\qquad
  L=-\big[y\log p+(1-y)\log(1-p)\big]
  \]
- Backprop is just the chain rule on the computation graph. Reverse-mode autodiff is preferred for deep nets because one scalar loss depends on many parameters.

## 3) CNN fundamentals and convolution variants
- For an input \(X\in\mathbb{R}^{H\times W\times C_{in}}\) and kernel \(K\times K\), output size is
  \[
  H_{out}=\left\lfloor\frac{H+2P-K_{eff}}{S}\right\rfloor+1,\qquad
  W_{out}=\left\lfloor\frac{W+2P-K_{eff}}{S}\right\rfloor+1
  \]
  where \(S\) is stride and \(P\) is padding.
- Dilated convolution effective kernel:
  \[
  K_{eff}=(K-1)d+1
  \]
  for dilation rate \(d\). It increases receptive field without increasing parameter count.
- Standard conv parameter count:
  \[
  K^2 C_{in} C_{out} + C_{out}
  \]
  if bias is used.
- Depthwise separable convolution:
  - depthwise part: \(K^2 C_{in}\)
  - pointwise \(1\times1\): \(C_{in} C_{out}\)
  - total:
    \[
    K^2 C_{in} + C_{in}C_{out}
    \]
- Transposed convolution output size:
  \[
  H_{out}=(H-1)S-2P+K+\text{output\_pad}
  \]
  (similarly for width).
  It is not the mathematical inverse of convolution; it is a learnable upsampling operator.
- Receptive field grows with depth, so early layers see local patterns and deeper layers see larger context.
- CNNs are translation equivariant; pooling / global aggregation can make the final prediction more translation invariant.

## 4) CNN architecture case studies
- AlexNet: early deep CNN breakthrough; ReLU, dropout, large kernels in early layers.
- VGG: many stacked \(3\times3\) convolutions. Simple design, but heavy in parameters and compute.
- Inception: parallel branches with different kernel sizes; \(1\times1\) convolutions are used to reduce channel dimension before expensive convolutions.
- ResNet:
  \[
  y=x+F(x)
  \]
  Skip connections improve gradient flow and optimization.
- MobileNet: built from depthwise separable convolution for efficiency.
- EfficientNet: compound scaling balances depth, width, and resolution together rather than scaling only one axis.

## 5) CNN engineering and design choices
- Design trade-offs in CNNs: representation capacity, optimization stability, receptive field, efficiency, and task alignment.
- Bottlenecks compress channels before expensive operations; expansion layers recover capacity.
- Downsampling should be placed deliberately because early excessive downsampling can destroy spatial detail.
- Global average pooling often replaces large fully connected heads when classification is the goal.
- Better initialization matters:
  - Xavier/Glorot is commonly used for tanh / sigmoid-like activations.
  - He/Kaiming is used for ReLU-like activations.
- Large learning rates can destabilize training even when gradients are not huge.
- Gradient exploding / vanishing are optimization failures, not just architecture issues.

## 6) Attention modules in CNNs
### SE / Squeeze-and-Excitation
- Let \(F\in\mathbb{R}^{C\times H\times W}\).
- Squeeze with global average pooling:
  \[
  z_c=\frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W F_c(i,j)
  \]
- Excitation with a small MLP:
  \[
  s=\sigma\!\big(W_2\,\delta(W_1 z)\big)
  \]
- Scale the channels:
  \[
  \tilde F_c=s_c F_c
  \]
- This is channel attention only.

### BAM
- BAM combines channel and spatial attention in parallel:
  \[
  M=\sigma(M_c+M_s),\qquad F' = F\odot(1+M)
  \]
- Channel branch gives \(M_c\in\mathbb{R}^{C\times1\times1}\); spatial branch gives \(M_s\in\mathbb{R}^{1\times H\times W}\).
- Broadcasting creates a full \(C\times H\times W\) gate.
- The residual-like \((1+M)\) helps avoid over-suppression.

### CBAM
- Sequential design:
  \[
  F \xrightarrow{\text{channel attn}} F_c \xrightarrow{\text{spatial attn}} F_{cs}
  \]
- Channel gate uses both average and max pooling:
  \[
  s=\sigma\!\big(\text{MLP}(\text{GAP}(F))+\text{MLP}(\text{GMP}(F))\big)
  \]
- Spatial gate pools across channels, concatenates the two maps, then applies a conv (often \(7\times7\)):
  \[
  a_s=\sigma(\text{Conv}_{7\times7}([\text{AvgPool}_c(F_c);\text{MaxPool}_c(F_c)]))
  \]
- MCQ trap: BAM is parallel and more aggressive; CBAM is sequential and usually simpler / stabler.

### Self-attention / Non-local attention
- Reshape \(F\in\mathbb{R}^{C\times H\times W}\) to \(X\in\mathbb{R}^{C\times N}\) with \(N=HW\).
- Form queries, keys, values:
  \[
  Q=W_qX,\quad K=W_kX,\quad V=W_vX
  \]
- Scaled dot-product attention:
  \[
  A=\text{softmax}\!\left(\frac{Q^\top K}{\sqrt d}\right),\qquad
  Y=VA^\top
  \]
- Then project back and often add a residual:
  \[
  F' = F + \text{Proj}(\text{reshape}(Y))
  \]
- Cost is \(O(N^2)\), so it is expensive at high resolution.

## 7) Visualization methods
- Class Activation Mapping (CAM) requires GAP before the classifier. If class \(c\) has FC weights \(w_k^c\) on final conv maps \(A_k\), then:
  \[
  M_c(i,j)=\sum_k w_k^c A_k(i,j)
  \]
- Grad-CAM uses gradients instead of requiring a special architecture:
  \[
  \alpha_k^c=\frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A_k(i,j)}
  \]
  \[
  L^c=\text{ReLU}\!\left(\sum_k \alpha_k^c A_k\right)
  \]
- Guided backprop changes the backward pass of ReLU so only positive gradients through positive activations are passed.
- Use logits rather than softmax probabilities for these methods when possible, because softmax saturates.

## 8) Regularization and normalization
- Dropout:
  \[
  Y=\frac{X\odot M}{p_{keep}}
  \]
  during training, where \(M\sim\text{Bernoulli}(p_{keep})\). At inference, use the full activations.
- DropConnect masks weights instead of activations.
- BatchNorm for a mini-batch \(X\):
  \[
  \mu=\frac{1}{m}\sum_i x_i,\qquad
  \sigma^2=\frac{1}{m}\sum_i (x_i-\mu)^2
  \]
  \[
  \hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
  y=\gamma \hat x+\beta
  \]
- LayerNorm normalizes across features within each sample.
- InstanceNorm normalizes each sample and each channel across spatial positions.
- GroupNorm normalizes within channel groups.
- BatchNorm inference uses running mean and variance, not the current mini-batch.
- Gradient clipping:
  \[
  g \leftarrow g\cdot \min\left(1,\frac{\tau}{\lVert g\rVert}\right)
  \]
  if clipping by global norm \(\tau\).

## 9) Generalization gap and compute cost
- Generalization gap = training performance minus validation/test performance.
- Parameter count does not alone determine test performance.
- FLOPs/MACs measure compute cost, not accuracy.
- A model can be large but still generalize well if its inductive bias, optimization, and regularization are good.

## 10) Loss functions
### Classification losses
- Binary cross-entropy:
  \[
  L=-[y\log p+(1-y)\log(1-p)]
  \]
- Multiclass cross-entropy:
  \[
  L=-\sum_k y_k\log p_k
  \]
- Weighted cross-entropy:
  \[
  L=-\sum_k w_k y_k\log p_k
  \]
  used for class imbalance.
- Focal loss:
  \[
  L=-\alpha (1-p_t)^\gamma \log(p_t)
  \]
  where \(p_t\) is the predicted probability assigned to the true class.
  It down-weights easy examples.
- Dice loss:
  \[
  L_{dice}=1-\frac{2\sum_i p_i y_i+\epsilon}{\sum_i p_i+\sum_i y_i+\epsilon}
  \]
  Common in segmentation, especially for imbalance.
- Contrastive loss for a pair:
  \[
  L=y\,d^2+(1-y)\max(m-d,0)^2
  \]
  where \(d\) is distance and \(m\) is the margin.

### Regression losses
- MSE:
  \[
  \text{MSE}=\frac1n\sum_i (y_i-\hat y_i)^2
  \]
- MAE:
  \[
  \text{MAE}=\frac1n\sum_i |y_i-\hat y_i|
  \]
- Huber loss:
  \[
  L_\delta(r)=
  \begin{cases}
  \frac12 r^2, & |r|\le \delta\\
  \delta(|r|-\frac12\delta), & |r|>\delta
  \end{cases}
  \]
  It behaves like MSE near zero and MAE for large residuals.
- Smooth L1 is a scaled / variant form of Huber used in many detection models.
- Log-cosh is smooth and less sensitive to outliers than MSE.

### Good-loss reminders
- Differentiability, stable gradients, metric alignment, robustness, and calibration all matter.
- For class imbalance, focal or dice is often better than plain CE.
- For noisy regression, Huber is often safer than MSE.

## 11) Optimization
### Gradient descent family
- Vanilla SGD:
  \[
  w_{t+1}=w_t-\eta g_t
  \]
- Momentum:
  \[
  v_t=\beta v_{t-1}+g_t,\qquad
  w_{t+1}=w_t-\eta v_t
  \]
- Nesterov momentum:
  \[
  g_t=\nabla f(w_t-\eta\beta v_{t-1}),\qquad
  v_t=\beta v_{t-1}+g_t
  \]
- AdaGrad:
  \[
  s_t=s_{t-1}+g_t^2,\qquad
  w_{t+1}=w_t-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}
  \]
- RMSProp:
  \[
  s_t=\beta s_{t-1}+(1-\beta)g_t^2,\qquad
  w_{t+1}=w_t-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}
  \]
- Adam:
  \[
  m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
  v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
  \]
  \[
  \hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
  \hat v_t=\frac{v_t}{1-\beta_2^t}
  \]
  \[
  w_{t+1}=w_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
  \]
- AdamW decouples weight decay:
  \[
  w_{t+1}=w_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\eta\lambda w_t
  \]
  or equivalently \((1-\eta\lambda)w_t\) before the Adam step.

### Regularization interactions
- Weight decay is not always identical to L2 regularization in adaptive methods.
- Label smoothing:
  \[
  y^{LS}=(1-\epsilon)y+\frac{\epsilon}{K}
  \]
- Gradient clipping is useful when gradients explode.

## 12) Sequence models: RNN, LSTM, GRU
### Vanilla RNN
- Hidden update:
  \[
  h_t=\phi(W_x x_t + W_h h_{t-1}+b)
  \]
- Output:
  \[
  \hat y_t=\text{softmax}(Vh_t)
  \]
- RNNs are trained with backpropagation through time (BPTT): unfold the network over time, compute gradients through the unfolded graph, then accumulate them.
- Main failure mode: vanishing / exploding gradients over long sequences.

### LSTM
- Forget gate:
  \[
  f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)
  \]
- Input gate:
  \[
  i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)
  \]
- Candidate state:
  \[
  \tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c)
  \]
- Cell update:
  \[
  c_t=f_t\odot c_{t-1}+i_t\odot \tilde c_t
  \]
- Output gate:
  \[
  o_t=\sigma(W_o[x_t,h_{t-1}]+b_o)
  \]
- Hidden state:
  \[
  h_t=o_t\odot\tanh(c_t)
  \]
- Intuition: forget gate decides what to erase, input gate decides what to write, output gate decides what to expose.

### GRU
- Update gate:
  \[
  z_t=\sigma(W_z[x_t,h_{t-1}]+b_z)
  \]
- Reset gate:
  \[
  r_t=\sigma(W_r[x_t,h_{t-1}]+b_r)
  \]
- Candidate:
  \[
  \tilde h_t=\tanh(W_h[x_t,r_t\odot h_{t-1}]+b_h)
  \]
- Hidden update:
  \[
  h_t=(1-z_t)\odot h_{t-1}+z_t\odot \tilde h_t
  \]
- GRU removes the separate cell state of LSTM and uses fewer gates.

### Bidirectional RNN
- Run one RNN forward and another backward, then combine their hidden states.
- Useful when the whole sequence is known at once and both past and future context matter.

## 13) Autoencoders and PCA
### Standard autoencoder
- Encoder and decoder:
  \[
  h=f_\theta(x),\qquad \hat x=g_\phi(h)
  \]
- Loss:
  \[
  L_{AE}=\sum_i \ell\!\big(x^{(i)},\hat x^{(i)}\big)
  \]
- Common reconstruction losses:
  - MSE for real-valued data
  - BCE for normalized binary-valued data
- Undercomplete AE means latent dimension \(m<d\).

### Linear AE and PCA
- In the linear case with linear activations and tied weights, the AE learns a principal subspace similar to PCA.
- PCA minimizes reconstruction error over a linear subspace; the linear autoencoder reaches the same subspace under suitable constraints.
- MCQ trap: nonlinear autoencoders are strictly more expressive than PCA.

### Sparse autoencoder
- Average hidden activation:
  \[
  \hat\rho_j=\frac{1}{N}\sum_i h_j(x^{(i)})
  \]
- Sparsity penalty:
  \[
  \sum_j KL(\rho\|\hat\rho_j),\qquad
  KL(\rho\|\hat\rho_j)=\rho\log\frac{\rho}{\hat\rho_j}+(1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}
  \]
- This pushes hidden units to be active only rarely.

### Denoising autoencoder
- Corrupt input:
  \[
  \tilde x\sim q(\tilde x|x)
  \]
- Reconstruct clean target:
  \[
  L_{DAE}=\sum_i \mathbb E_{\tilde x^{(i)}\sim q(\tilde x|x^{(i)})}\left[\ell\big(x^{(i)},g_\phi(f_\theta(\tilde x^{(i)}))\big)\right]
  \]
- Used to learn robust features and local manifold structure.

### Contractive autoencoder
- Encoder Jacobian:
  \[
  J_f(x)=\frac{\partial h}{\partial x}
  \]
- Contractive penalty:
  \[
  \|J_f(x)\|_F^2
  \]
- Objective:
  \[
  L_{CAE}=\sum_i \ell\!\big(x^{(i)},\hat x^{(i)}\big)+\lambda \sum_i \|J_f(x^{(i)})\|_F^2
  \]
- It encourages local invariance around the data manifold.

## 14) Variational Autoencoders
- Latent-variable model:
  \[
  p_\theta(x,z)=p(z)p_\theta(x|z)
  \]
- Encoder approximates posterior:
  \[
  q_\phi(z|x)\approx p_\theta(z|x)
  \]
- ELBO:
  \[
  \log p_\theta(x)\ge
  \mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]
  - KL\!\big(q_\phi(z|x)\|p(z)\big)
  \]
- The first term is reconstruction; the second term pushes the posterior toward the prior.
- Reparameterization trick:
  \[
  z=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
  \]
  so gradients can flow through \(\mu\) and \(\sigma\).
- For a Gaussian prior and diagonal Gaussian posterior, the KL term has a closed form and is often the part that causes the "Gaussian latent regularization" MCQ traps.
- Beta-VAE modifies the objective:
  \[
  \mathcal L=\mathbb E[\log p_\theta(x|z)]-\beta\,KL(q_\phi(z|x)\|p(z))
  \]
  Larger \(\beta\) encourages more disentanglement but can hurt reconstruction.

## 15) GANs
### Vanilla GAN
- Objective:
  \[
  \min_G \max_D
  \mathbb E_{x\sim p_{data}}[\log D(x)]
  +
  \mathbb E_{z\sim p(z)}[\log(1-D(G(z)))]
  \]
- Optimal discriminator for fixed \(G\):
  \[
  D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}
  \]
- Substituting \(D^*\) into the objective gives:
  \[
  V(D^*,G)=-\log 4 + 2\,JS(p_{data}\|p_g)
  \]
  so the generator minimizes Jensen-Shannon divergence, not Wasserstein or KL.
- Non-saturating generator loss:
  \[
  L_G=-\mathbb E_z[\log D(G(z))]
  \]
  is often used instead of \(\mathbb E_z[\log(1-D(G(z)))]\) because it gives stronger gradients early in training.
- DCGAN design heuristics:
  strided conv instead of pooling, fractionally strided conv for upsampling, batch norm, ReLU in generator except tanh output, leaky-ReLU in discriminator, avoid heavy fully connected stacks.

### GAN failure modes and fixes
- Common failure modes: unstable game dynamics, vanishing gradients, mode collapse, non-convergence.
- Practical fixes include:
  lower learning rate, separate generator/discriminator learning rates, feature matching, label smoothing, and better architecture design.

### LSGAN / WGAN / WGAN-GP / cGAN / InfoGAN
- Least-squares GAN replaces the sigmoid loss with squared error targets.
- Wasserstein-1 distance:
  \[
  W(P_r,P_g)=\sup_{\|f\|_L\le 1}\left(\mathbb E_{x\sim P_r}[f(x)]-\mathbb E_{x\sim P_g}[f(x)]\right)
  \]
  by Kantorovich-Rubinstein duality.
- WGAN critic tries to maximize the difference of expected scores under the Lipschitz constraint.
- WGAN-GP adds gradient penalty:
  \[
  \lambda\,\mathbb E_{\hat x}\left(\|\nabla_{\hat x}D(\hat x)\|_2-1\right)^2
  \]
- Conditional GAN:
  \[
  G(z,y),\quad D(x,y)
  \]
  where \(y\) is a label or condition.
- InfoGAN encourages interpretable latent codes by maximizing mutual information between latent variables and generated samples:
  \[
  I(c;G(z,c))=H(c)-H(c|G(z,c))
  \]
  Usually an auxiliary network \(Q(c|x)\) is used to lower-bound this term.

## 16) High-yield MCQ traps
- Jensen-Shannon divergence appears in the vanilla GAN derivation after plugging in the optimal discriminator.
- Wasserstein distance belongs to WGAN, not vanilla GAN.
- BCE / softmax CE are classification losses; MSE is the default regression loss.
- BatchNorm normalizes across a batch; LayerNorm does not.
- SE is channel attention; CBAM does channel then spatial; BAM combines both in parallel.
- Grad-CAM uses gradients and can be applied without a special architecture, unlike CAM.
- RNNs struggle on long dependencies; LSTM and GRU are designed to fix this.
- Autoencoders reconstruct; VAEs are probabilistic latent-variable models with an ELBO; GANs are adversarial generative models.
]


Follow these steps strictly:
1. Extract the question and options. Note that options might be labeled A, B, C, D instead of 1, 2, 3, 4.
2. THINK STEP-BY-STEP. First refer to the "Materials" I have given and if you can answer using that, then use that, else use your own thinking. 
3. If there is math, explicitly write out the floor() formula and calculate it carefully. If there is code, trace the tensor shapes layer by layer.
4. Evaluate all options to find the correct one.
5. You MUST output your final chosen option inside XML tags. 
6. Map the options to numbers: A=1, B=2, C=3, D=4.
7. If you are unsure or the image is completely unreadable, output 5 to skip.

Example format(you must ALWAYS follow this):
Thought Process: [Write your step-by-step reasoning, calculations, and tensor tracing here]
<answer>1</answer>
"""

def resolve_image_path(test_dir, img_filename):
    """
    Robustly resolve image path handling two known ambiguities:
    1. Folder name: README says 'image/' but folder structure shows 'images/'
    2. Extension: image_name in test.csv may or may not include '.png'
    """
    for folder in ['images', 'image']:
        for fname in [img_filename, img_filename + '.png']:
            candidate = os.path.join(test_dir, folder, fname)
            if os.path.exists(candidate):
                return candidate
    # Fallback: return the original guess so the error message is meaningful
    return os.path.join(test_dir, 'images', img_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help="Absolute path to test directory")
    args = parser.parse_args()

    # 1. Read the Test CSV
    test_csv_path = os.path.join(args.test_dir, 'test.csv')
    df = pd.read_csv(test_csv_path)

    # 2. Load Model (Bypassing 16GB CPU RAM Limit)
    model_path = "./weights/qwen2_vl_7b"
    print("Loading model from local storage directly to GPU...")
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True
    ).eval()

    results = []
    print(f"Starting inference on {len(df)} images...")

    # 3. Inference Loop
    for index, row in df.iterrows():
        img_filename = row['image_name']

        # FIX: Robustly resolve path for both 'image/' and 'images/' folders,
        # and with or without '.png' extension in the image_name column
        img_path = resolve_image_path(args.test_dir, img_filename)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path, "max_pixels": 1280 * 1280},
                    {"type": "text", "text": FINAL_PROMPT},
                ],
            }
        ]

        # Prepare Inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # FIX: Removed temperature=0.2 — it has no effect when do_sample=False
        # and causes warnings/errors in newer transformers versions
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=768, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 4. Extract Answer using Regex
        ans = 5  # Default fallback
        match = re.search(r'<answer>\s*([12345ABCDabcd])\s*</answer>', output_text)
        if match:
            raw_ans = match.group(1).upper()
            mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            if raw_ans in mapping:
                ans = mapping[raw_ans]
            else:
                ans = int(raw_ans)
                
        print(f"[{index+1}/{len(df)}] {img_filename} -> Output: {ans}")

        results.append({
            'id': img_filename,
            'image_name': img_filename,
            'option': ans
        })

    # 5. Save submission.csv in the current working directory (NOT test_dir)
    submission_df = pd.DataFrame(results)
    submission_df = submission_df[['id', 'image_name', 'option']]
    submission_df.to_csv('submission.csv', index=False)
    
    print("✅ Inference complete. submission.csv generated successfully.")

if __name__ == "__main__":
    main()

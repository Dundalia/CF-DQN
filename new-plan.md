# CVI Architecture: Previous vs. Proposed Implementation

This document outlines the transition from a discrete, grid-based Characteristic Value Iteration (CVI) model to a continuous, analytic approach. The new design eliminates the computationally heavy IFFT collapse step and stabilizes high-frequency training.

---

## 1. Inputs & Frequencies ($\omega$)
* **Previous:** Evaluated the state $x$ across a fixed, hardcoded grid of frequencies. 
* **Proposed:** We now sample a batch of continuous frequencies $\omega$ on the fly (similar to IQN) and feed them into the network alongside the state. 
    * *Why?* We can sample from distributions that prioritize low frequencies. Low frequencies dictate the main shape of the return distribution, while high frequencies mostly contain noise. The goal will be to gradually move to higher frequency samples as training progresses, but to start with lower frequencies to learn the main shape of the distribution.

## 2. Network Architecture & Embeddings
* **Previous:** A standard network outputted raw complex numbers directly. We had to use a forced normalization layer to make sure the value at $\omega=0$ was exactly $1+0j$.
* **Proposed:** The network now splits into two strictly real-valued heads:
    * **Head 1:** Outputs $m$ (acting as the mean/location).
    * **Head 2:** Outputs $\sigma$ (acting as the spread/variance).
    * *The Cosine Assumption:* Before hitting the heads, the $\omega$ inputs are embedded using cosine functions. Because characteristic functions must be perfectly symmetric (even functions where $f(-\omega) = f(\omega)$), using cosines mathematically forces the network's outputs to respect this symmetry without extra work.

## 3. Forward Pass & The Analytic Formula
* **Previous:** The complex outputs were used directly to compute the Distributional Bellman Error.
* **Proposed:** The network outputs real numbers, and we construct the complex characteristic function $\phi(x, \omega)$ using this specific formula:
  $$\phi(x, \omega) = \exp\left( -\frac{1}{2} \kappa(\omega)\sigma(x, \omega) + i \omega m(x, \omega) \right)$$
    * *Built-in Normalization:* When $\omega=0$, the entire exponent becomes $0$, naturally forcing the output to $1+0j$. No hacky normalization layer needed.
    * *Not Just Gaussian:* Even though it looks like a Gaussian CF, because $m$ and $\sigma$ change dynamically depending on the $\omega$ passed in, this formula can warp to model skewed or multi-modal distributions. 
    * *Fixing the $2\pi$ Phase Wrap:* At high frequencies, large phases ($>2\pi$) confuse the network because it cannot distinguish between angles like $\pi$ and $3\pi$. By introducing $\kappa(\omega)$ (which grows quickly at high frequencies), the real part of the exponent rapidly shrinks the magnitude of the complex vector to zero. When the magnitude is zero, the phase no longer matters, completely protecting the network from high-frequency chaos.

## 4. Inference (Action Selection)
* **Previous (The Collapse):** To pick an action, we had to compute the complex function, run an Inverse Fast Fourier Transform (IFFT) to convert it back into a probability distribution, and then calculate the expected Q-value. This was a massive computational bottleneck.
* **Proposed (Direct Extraction):** The IFFT is completely gone. To select an action, we simply feed the state $x$ and a hardcoded $\omega=0$ into the network. 
    * *Why it works:* By design, when $\omega=0$, the output of the $m$ head *is* the exact expected Q-value. We just read the output of Head 1 and apply our `argmax`. Fast, cheap, and mathematically sound.
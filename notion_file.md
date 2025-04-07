# Enhanced Study Notes: Global Image Transforms (Lecture 2)

Generated from Lecture Notes

## 1. Introduction: Why Bother with Transforms?

*(Approximate Time: 00:00:00 - 00:04:45)*

### 1.1. The Starting Point: The Spatial Domain

An image, as we typically see it, exists in the **spatial domain**. It's a grid (matrix) of pixels, where each pixel at location $(i, j)$ has a value $s(i,j)$ representing intensity or color. While intuitive, this representation isn't always the most effective for processing or analysis.

***Further Explanation:*** Why isn't the spatial domain always best? Pixel values are often highly correlated (nearby pixels tend to have similar values), meaning there's redundancy. Also, features like texture or periodic noise are defined by relationships between pixels, which aren't explicit in the raw pixel values.

### 1.2. The Goal: Moving to a Transform Domain

The core idea of an image transform is to represent this same image information in a different *domain* or coordinate system. We convert the matrix of pixel values $S$ into a set of *transform coefficients* $C$. Why?

- **Revealing Hidden Structure:** The transform domain might make certain properties of the image (like textures, edges, or periodic patterns) much more explicit and easier to analyze than they are in the spatial domain.
- **Efficiency for Tasks:** Different domains are better suited for different tasks. For example:
    - **Compression:** Some transforms "pack" the image's energy into just a few coefficients, allowing us to discard the rest with minimal visual impact (lossy compression).
***Further Explanation:*** This "packing" is often called energy compaction. If most of the image's visual information can be represented by a small number of significant coefficients, the rest can be ignored or stored with less precision.
    - **Noise Reduction:** Noise often has different characteristics in the transform domain than the actual image signal, allowing us to separate and remove it more easily.
***Further Explanation:*** For instance, random noise might spread its energy thinly across many transform coefficients, while the image signal's energy is concentrated in a few. Filtering can then target the low-energy noise coefficients.
    - **Feature Extraction:** The transform coefficients themselves can serve as descriptive features for tasks like image recognition or analysis.
***Further Explanation:*** The set of coefficients acts like a "fingerprint" of the image in the chosen domain, summarizing its key characteristics according to the transform's basis.

### 1.3. Analogy: Different Perspectives

Think of looking at a complex machine. Viewing it from the front (spatial domain) gives one perspective. Viewing the blueprints (a transform domain) reveals the internal structure and relationships between parts, which might be better for understanding how it works or modifying it.

### 1.4. The Encoder-Decoder Concept

Many modern image processing pipelines, especially in machine learning, utilize an encoder-decoder structure which mirrors the transform concept:

- **Encoder:** Transforms the input image into a (usually compact) representation in a *latent space* (the transform domain).
- **Decoder:** Transforms the latent representation back into the spatial domain, ideally reconstructing the original or a processed version.

### 1.5. Focus of this Lecture: Global Linear Transforms

We'll focus on *global* transforms, meaning they consider the entire image (or large blocks) at once. We'll also focus on *linear* transforms, which have well-defined mathematical properties based on linear algebra.

**Teacher:** Started the course focusing on global transforms [00:19:48 - 00:24:40]. The slides emphasize defining these directly for *discrete* images (pixels), which is crucial for practical computation.

## 2. The General Model: Images as Weighted Sums of Basis Images

*(Approximate Time: 00:04:45 - 00:13:49)*

### 2.1. The Fundamental Idea

Any linear transform represents an image as a weighted sum of predefined "building blocks" called **basis images** (or basis functions/vectors).

**Teacher:** Introduced the core idea: representing an image pixel $s(i, j)$ as a weighted sum of basis images $B_{kl}(i, j)$ multiplied by corresponding coefficients $c_{kl}$ [00:05:04 - 00:05:11].

***Further Explanation:*** Think of this like representing any vector in standard 3D space as a combination of the basis vectors $\mathbf{i}, \mathbf{j}, \mathbf{k}$. Linear transforms generalize this concept to the high-dimensional space where images live, using a potentially different set of basis vectors tailored for images.

**Analogy: Lego Bricks:** Imagine you have a standard set of Lego bricks (the basis images, $B_{kl}$). Any Lego structure (your image, $S$) can be built by combining specific quantities (the coefficients, $c_{kl}$) of these standard bricks. The choice of brick set (the transform type) determines how efficiently you can build typical structures.

### 2.2. Mathematical Formulation (Inverse Transform / Synthesis)

This describes how to build the image $s(i,j)$ from the coefficients $c_{kl}$ and basis images $b_{kl}(i,j)$:
The sums go over all the basis images used (up to $L$ of them). This equation essentially says the image is synthesized by adding together scaled versions of the basis images.

$$
s(i,j) = \sum_{k} \sum_{l} c_{kl} \cdot b_{kl}(i,j)
$$

### 2.3. Vector/Matrix Notation (for Convenience)

Dealing with 2D indices $(i,j)$ and $(k,l)$ can be cumbersome. Linear algebra provides a more compact way.

1. **Vectorization:** Reshape the $N \times M$ image matrix $S$ into a single long column vector $\mathbf{s}$ of size $P = NM$. This is often done by stacking columns (or rows) - a process called *lexicographical ordering*. Similarly, reshape each $N \times M$ basis image $B_k$ into a column vector $\mathbf{b}_k$ ($P \times 1$).

**Teacher:** Introduced the convenience of using vector/matrix notation by raster scanning the image S into a long vector s (dimensionality P) and basis images B into vectors $v_k$ [00:09:07 - 00:09:59].

***Further Explanation:*** Vectorization allows us to leverage powerful tools from linear algebra, treating the image as a single point in a P-dimensional vector space.
2. **Synthesis Equation:** The sum becomes a matrix-vector product:
Where:
$$
\mathbf{s} = \sum_{k=1}^{L} c_k \mathbf{b}_k = \mathbf{B} \mathbf{c}
$$
- $\mathbf{s}$ is the image vector ($P \times 1$).
- $\mathbf{c}$ is the coefficient vector ($L \times 1$).
- $\mathbf{B}$ is the *transform matrix* ($P \times L$), whose columns are the basis vectors $[\mathbf{b}_1 | \mathbf{b}_2 | ... | \mathbf{b}_L]$.
**Teacher:** Presented the transform as $\mathbf{s} = \sum_{k=0}^{L-1} c_k \mathbf{v}_k$ and in matrix form $\mathbf{s} = \mathbf{V} \mathbf{c}$ where V is the matrix whose columns are the basis vectors v\_k [00:10:13 - 00:10:35]. (Using V for $\mathbf{B}$ and v for $\mathbf{b}$).

***Further Explanation:*** This equation $\mathbf{s} = \mathbf{B} \mathbf{c}$ is fundamental. It states the image vector $\mathbf{s}$ is a linear combination of the basis vectors (columns of $\mathbf{B}$), with the coefficients given by the vector $\mathbf{c}$.
### 2.4. Dimensionality and Reversibility

- An image $\mathbf{s}$ lives in a $P$-dimensional space.
- **Teacher:** Discussed dimensionality [00:05:37 - 00:07:25]. An M x N image has P = M * N pixels. A complete, lossless representation generally requires P basis images and P coefficients.
To represent *any* possible image perfectly, we generally need $L=P$ basis vectors that are *linearly independent* (they span the entire $P$-dimensional space). In this case, the transform is *lossless* or *reversible*.
***Further Explanation:*** Linear independence means no basis vector can be represented as a combination of the others. Spanning the space means any vector (image) can be reached by combining the basis vectors.
- **Teacher:** Explicitly asked what happens if we use fewer basis images ($L < P$) and confirmed it leads to a lossy representation where the goal is to minimize the loss [00:06:51 - 00:07:39]. If we use fewer basis vectors ($L < P$), our reconstruction $\mathbf{s}' = \mathbf{B} \mathbf{c}$ can only represent images lying within the $L$-dimensional *subspace* spanned by those basis vectors. Information outside this subspace is lost, making the transform *lossy*. The goal in lossy compression is to choose the L basis vectors that capture *most* of the important information for typical images.

### 2.5. Forward vs. Inverse Transform

- **Teacher:** Defined Forward Transform (spatial domain s to transform domain c) and Inverse Transform (transform domain c back to spatial domain s) [00:08:11 - 00:08:36].
- **Inverse (Synthesis):** $\mathbf{s} = \mathbf{B} \mathbf{c}$ (Build image from coefficients).
- **Forward (Analysis):** Find $\mathbf{c}$ given $\mathbf{s}$ and $\mathbf{B}$ (Analyze image into coefficients).
- **Teacher:** Stressed the desire for reversibility (losslessness) where Forward then Inverse returns the original signal, but noted not all transforms achieve this [00:08:36 - 00:09:06].

### 2.6. Trivial Example (Identity Transform)

- **Teacher:** Provided a trivial example using basis vectors that are all zeros except for a single '1' [00:11:47 - 00:13:22]. He showed that in this case, the transform matrix V is the identity matrix, and the coefficients c are just the original pixel values s.
- **Teacher:** He explicitly stated this is "not a transform, doesn't do anything" but serves to illustrate the mathematical formula [00:12:21 - 00:13:49].

## 3. Key Questions for Evaluating Transforms

*(Approximate Time: 00:13:49 - 00:21:30)*

**Teacher:** Posed several key questions [00:15:34 - 00:17:15] when evaluating transforms:

1. **Basis Images:** What are they? Fixed (DFT/DCT mentioned) or data-dependent?
2. **Number (L):** How many needed? What if $L < P$? (Loss of information).
3. **Coefficients:** Properties/meaning? Correlation? ([00:15:34]).
4. **Processing:** How is it done? (Stated processing happens in transform domain).
5. **Computation:** Efficient? Fast algorithms? ([00:16:32]).
6. **Resources:** Storage/speed? ([00:16:44]).
7. **Reversibility:** When possible? ([00:16:50]).
8. **Applications:** What good for? (Recapped).

### 3.1. Geometric Interpretation \& Error

- **Teacher:** Introduced the geometric idea of minimizing approximation error using a 3D vector example [00:17:24 - 00:19:24]. If using one basis vector ($\mathbf{v}_1$), the best approximation ($\mathbf{s}'$) is the projection of $\mathbf{s}$ onto $\mathbf{v}_1$. Extended to projecting onto the plane (subspace) spanned by $\mathbf{v}_1, \mathbf{v}_2$. The error vector $(\mathbf{s}-\mathbf{s}')$ is orthogonal to the subspace.
- **Teacher:** Stated the general principle: to minimize error when using L basis vectors, project the original signal $\mathbf{s}$ onto the subspace spanned by those L vectors [00:19:24 - 00:20:04]. Representing $\mathbf{s}$ using $L$ basis vectors $\{\mathbf{b}_1, ..., \mathbf{b}_L\}$ means finding the best approximation $\mathbf{s}' = \sum_{k=1}^{L} c_k \mathbf{b}_k$ within the *subspace* spanned by these basis vectors. The best approximation (LSE) is the *orthogonal projection*.

## 4. Calculating Coefficients (Forward Transform)

*(Approximate Time: 00:21:30 - 00:30:44)*

### 4.1. The Goal

Given $\mathbf{s}$ and $\mathbf{B}$, find the "best" $\mathbf{c}$ such that $\mathbf{s} \approx \mathbf{B} \mathbf{c}$.

### 4.2. Least Squares Error (LSE) Approach

We find the $\mathbf{c}$ that minimizes this $E$ by using calculus (setting the derivative $\partial E / \partial \mathbf{c}$ to zero).
***Further Explanation:*** Minimizing the squared error is a standard technique; it heavily penalizes large errors and leads to mathematically tractable solutions.

**Teacher:** Stated the goal: Find coefficients c that minimize the least squares error $E = ||\mathbf{s} - \mathbf{s}'||^2 = ||\mathbf{s} - \mathbf{B} \mathbf{c}||^2$.
Mathematically, minimizing the error norm is equivalent to minimizing the *squared* error:

$$
E = ||\mathbf{s} - \mathbf{s}'||^2 = ||\mathbf{s} - \mathbf{B} \mathbf{c}||^2
$$

**Teacher:** Sketched the derivation using calculus [00:23:37 - 00:24:15]. He expanded the norm $E = (\mathbf{s} - \mathbf{B}\mathbf{c})^T (\mathbf{s} - \mathbf{B}\mathbf{c})$ and mentioned differentiating.

**Teacher:** Presented required matrix calculus identities [00:24:36 - 00:25:05] and clarified conventions [00:25:05 - 00:28:12].

**Teacher:** Showed the result of the differentiation leads to $2 \mathbf{B}^T \mathbf{B} \mathbf{c} - 2 \mathbf{B}^T \mathbf{s} = \mathbf{0}$ [00:29:38 - 00:29:50].

### 4.3. General Solution (Pseudo-Inverse)

Solving for $\mathbf{c}$ yields the LSE solution:

$$
\mathbf{c} = (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T \mathbf{s}
$$

**Teacher:** Identified the matrix $\mathbf{B}^+ = (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T$ as the **pseudo-inverse** of $\mathbf{B}$ [00:29:50 - 00:30:06]. This provides the best LSE coefficients $\mathbf{c}$ even when the basis isn't orthogonal or $L < P$.

### 4.4. Simplified Solution: Orthonormal Basis

If the basis vectors are **orthonormal** ($\mathbf{B}^T \mathbf{B} = \mathbf{I}$), life is simpler.

**Teacher:** Presented the much simpler solution when the basis V (using his notation for B) is orthonormal [00:30:29 - 00:30:44]:

$$
\mathbf{c} = \mathbf{I}^{-1} \mathbf{B}^T \mathbf{s} = \mathbf{B}^T \mathbf{s}
$$

Each coefficient $c_k$ is just the **dot product** (scalar product) of $\mathbf{s}$ with $\mathbf{b}_k$: $c_k = \mathbf{b}_k^T \mathbf{s}$.

***Further Explanation:*** This means $c_k$ measures "how much of the pattern $\mathbf{b}_k$ is present in the image $\mathbf{s}$" via direct projection. Orthonormal bases make analysis (finding $\mathbf{c}$) and synthesis (reconstructing $\mathbf{s}$) computationally symmetric and efficient.

## 5. Separable Transforms: A Shortcut for 2D

*(Approximate Time: 00:30:44 - 00:39:00)*

### 5.1. The Challenge

Avoid explicit formation and multiplication by the potentially huge transform matrix $\mathbf{B}$.

### 5.2. Separability

A 2D transform is **separable** if $b_{kl}(i,j) = p_k(i) \cdot q_l(j)$.

**Teacher:** Defined a separable transform as one where the 2D basis can be written as a product of 1D basis functions [00:31:02 - 00:31:34]. Mentioned symmetric separable if $p=q$.

### 5.3. Computational Advantage

Allows 2D transform via sequential 1D transforms along rows and columns.

**Teacher:** Explained the computational advantage [00:32:00 - 00:32:26].

**Teacher:** Gave the 2D Fourier transform row-column computation example [00:33:20 - 00:33:55].

***Further Explanation:*** This reduces complexity significantly, e.g., from $O((NM)^2)$ to $O(NM(N+M))$ or better if fast 1D algorithms like the FFT exist.

### 5.4. Matrix Notation (Orthonormal)

With 1D orthonormal basis matrices P, Q:

**Teacher:** Provided the matrix formulation [00:33:47 - 00:34:00]:

- Forward: $C = P^T S Q$
- Inverse: $S = P C Q^T$

### 5.5. Reversibility

If 1D transforms P, Q are reversible, the 2D transform is too.

**Teacher:** Discussed reversibility [00:34:03 - 00:35:05].

**Teacher:** Named examples: DCT, DFT, Haar.

## 6. Discrete Karhunen-Loève Transform (KLT / PCA): The Optimal Transform

*(Approximate Time: 00:46:08 - 01:08:10)*

### 6.1. The Goal

Design a transform tailored to a *specific class* of images for optimal *energy compaction* and *decorrelation*, minimizing MSE for compression.

**Teacher:** Introduced KLT/PCA as optimized for a specific population/ensemble [00:46:08 - 00:46:56], aiming for optimal error minimization with fewer coefficients [00:47:00 - 00:48:41].

### 6.2. KLT Philosophy

Align basis vectors with directions of maximum data variance, found via covariance analysis.

### 6.3. Dependence on Data Statistics

KLT basis is derived from the image ensemble's statistics.

**Teacher:** Defined the setup [00:49:14 - 00:50:31]: Treat images as random vectors, calculate mean $\mathbf{\mu}_s$ and covariance $\mathbf{C}_s$.

1. **Mean Removal:** Work with $\mathbf{s}_0 = \mathbf{s} - \mathbf{\mu}_s$.
2. **Covariance Matrix:** $\mathbf{C}_s = E[\mathbf{s}_0 \mathbf{s}_0^T]$. Encodes pixel variance and pairwise covariance.

**Teacher:** Explained $\mathbf{C}_s$ captures statistical links [00:50:31 - 00:52:53], showed example [00:52:56 - 00:54:27]. Stressed using representative data [00:54:51 - 00:55:43].
3. **Eigen-decomposition:** Find eigenvalues $\lambda_k$ and eigenvectors $\mathbf{b}_k$ of $\mathbf{C}_s$: $\mathbf{C}_s \mathbf{b}_k = \lambda_k \mathbf{b}_k$.

**Teacher:** Defined KLT basis construction via eigenvectors/values [00:55:43 - 00:57:15].
    - **Eigenvectors $\mathbf{b}_k$ (Principal Components):** The orthonormal KLT basis vectors representing principal directions of variance.
    - **Eigenvalues $\lambda_k$:** Variance along $\mathbf{b}_k$; \$ \lambda_k = E[c_k^2]\$.
4. **Ordering:** Sort $\mathbf{b}_k$ by descending $\lambda_k$ ($\lambda_1 \ge \lambda_2 \ge ...$) to form $\mathbf{B}$.

### 6.4. KLT Equations

**Teacher:** Defined KLT forward/inverse transforms [00:57:15 - 00:57:43, 01:00:17 - 01:00:35]:

- Forward: $\mathbf{c} = \mathbf{B}^T (\mathbf{s} - \mathbf{\mu}_s)$
- Inverse: $\mathbf{s} = \mathbf{B} \mathbf{c} + \mathbf{\mu}_s$

### 6.5. Why KLT is "Optimal"

**Teacher:** Listed key properties [00:58:02 - 00:59:20, 01:21:00 - 01:24:43]: Orthonormal basis, zero mean coefficients, uncorrelated coefficients, optimal energy compaction, minimum reconstruction error.

1. **Decorrelation:** Coefficients $c_k$ are uncorrelated ($\mathbf{C}_c = \mathbf{\Lambda} = \text{diag}(\lambda_1, ...)$). Removes redundancy.
2. **Energy Compaction:** Packs maximum variance into the first $L$ coefficients.
3. **Minimum MSE:** Discarding $c_{L+1}, ..., c_P$ gives minimum average MSE for $L$ terms: $MSE_L = \sum_{k=L+1}^{P} \lambda_k$.

**Teacher:** Explained MSE property [01:01:15 - 01:03:30].

**Proof Outline:**

**Teacher:** Sketched proof using Lagrange multipliers [01:13:44 - 01:19:11]: Maximizing $c_1$'s variance subject to normalization shows $\mathbf{b}_1$ must be eigenvector for largest $\lambda_1$.

### 6.6. Practical Challenges

- **Computational Cost:** Diagonalizing huge $\mathbf{C}_s$ is often infeasible.

**Teacher:** Highlighted computational cost, gave 512x512 example [01:08:10 - 01:09:52].
- **Data Dependence:** $\mathbf{B}$, $\mathbf{\mu}_s$ needed per dataset.
- **Approximations:** Block-based KLT or modeling $\mathbf{C}_s$.

**Teacher:** Mentioned need for approximations: block-based (e.g., 16x16, assumes inter-block independence) or modeling covariance (e.g., exponential decay [01:27:38 - 01:28:11]) [01:09:52 - 01:10:00, 01:25:43 - 01:29:16].
- **Performance:**

**Teacher:** Showed performance graph [01:29:16 - 01:30:08] where KLT is minimal.

**Teacher:** Concluded practical gain over DCT might be small; DCT is faster and often near-optimal for natural signals [01:30:08 - 01:31:05].

## 7. Discrete Fourier Transform (DFT): Analyzing Frequency

*(Approximate Time: 01:10:00 - End)*

### 7.1. Concept

Fixed, separable transform using complex exponentials $e^{j\theta}$ to represent signals as sums of sinusoids of different spatial frequencies.

**Teacher:** Introduced DFT, assuming familiarity. Mentioned CFT basis.

### 7.2. Basis Functions

Complex exponentials representing waves.

- 1D: $b_k(i) \propto e^{j 2\pi ik / N}$
- 2D: $b_{kl}(i,j) \propto e^{j 2\pi (ik/N + jl/M)}$

### 7.3. Interpretation

Coefficients $C(k,l)$ represent amplitude and phase of spatial frequencies. Low frequencies = slow variations; High frequencies = details/edges/noise. DC = average value.

### 7.4. DFT Equations

(Normalization may vary)

**Teacher:** Gave 1D DFT formulas [01:34:21 - 01:35:36]:

- Forward:

$$
C(k) = \sum_{i=0}^{N-1} s(i) e^{-j 2\pi ik / N}
$$
- Inverse:

$$
s(i) = \frac{1}{N} \sum_{k=0}^{N-1} C(k) e^{j 2\pi ik / N}
$$

**Teacher:** Gave 2D DFT formulas [01:35:41 - 01:36:38]:

- Forward:

$$
C(k,l) = \sum_{i=0}^{N-1} \sum_{j=0}^{M-1} s(i,j) e^{-j 2\pi (\frac{ik}{N} + \frac{jl}{M})}
$$
- Inverse:

$$
s(i,j) = \frac{1}{NM} \sum_{k=0}^{N-1} \sum_{l=0}^{M-1} C(k,l) e^{j 2\pi (\frac{ik}{N} + \frac{jl}{M})}
$$

### 7.5. Key Properties

Separable (allows FFT); Linear; Shift Theorem; Convolution Theorem; Periodicity; Conjugate Symmetry (real input $\implies$ symmetric amplitude spectrum).

**Teacher:** Stated DFT is separable, allows row-column computation.

### 7.6. Uses

Frequency analysis, linear filtering.

### 7.7. Visualization

Amplitude spectra often log-scaled; DC shifted to center.

## 8. Discrete Cosine Transform (DCT): The Compression Workhorse

*(Note: Detailed in slides, mentioned as example in timed transcript sections)*

### 8.1. Concept

Fixed, real-valued, separable transform using cosine basis functions.

### 8.2. Basis Functions (Type II)

Cosine waves at different frequencies.

$$
b_{kl}(i,j) \propto \cos\left[\frac{(2i+1)k\pi}{2N}\right] \cos\left[\frac{(2j+1)l\pi}{2M}\right]
$$

### 8.3. Key Advantage

Excellent energy compaction for correlated data (natural images). Implicitly handles boundaries better than DFT, reducing artifacts. Basis resembles KLT basis for common image models.

### 8.4. Uses

Core of JPEG, MPEG etc. due to high compaction and fast algorithms.

## 9. Summary of Transform Applications

*(Based on slides)*

- **Noise Reduction:** Filter coefficients in transform domain.
- **Feature Extraction:** Use coefficients or basis vectors as features.
- **Data Compression:** Transform (DCT) -> Quantize -> Entropy Code.

## 10. Exam Notes

*(Based on teacher comments)*

- **Teacher:** Stated exam is closed book. Annex (formula sheet) provided.
- **Teacher:** Strongly emphasized need for understanding intuition and math over rote memorization [01:31:22 - 01:32:31].
- **Teacher:** Assumed prerequisite knowledge (linear algebra, possibly Fourier basics) [01:32:41 - 01:33:51].

# ZERO DEPENDENCY (DENSE MLP) NEURAL NETWORK 
<code><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></code>
<code><img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" /></code>
<code><img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" /></code>

## Key Architectural Features

* **Zero-Dependency Core:** Numpy is fully used as the main mathematical source.
* **Numerical Stability Engine:** Handles forward and backpropagation like a champ. Has following modes:
    * **Sigmoid** : Has Overflow clipping, anti-vaneshing mechanism with hybrid RelU style gradients.
    * **RelU**    : Has fixed dead neuron problems with Leaky RelU.
* **Explicit Gradient-Wise Updates:** Hand-rolled backpropagation executing exact partial derivatives ($\frac{\partial L}{\partial W}$, $\frac{\partial L}{\partial b}$) to optimize weights layer-by-layer.
* **Modular Pipeline Design:** Highly flexible design with various pipeline such as:
    * **Image Pipeline** : Auto cropping of handwritten numbers using cv2 (format: --> Non_Inverse)
    * **Zip Pipeline**   : Handles zipped folders `.zip` to extract required data into specific format for training samples.
      * Image extraction (MINST Data)
      * Prediction data (Titanic Data sets and e.t.c)
* **Handwritten Mini Batch Trainin: ** Small amount data are calculated normally but anything that exceeds 32 examples are taken mini batch training process.

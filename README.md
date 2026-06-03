# ** ZERO DEPENDENCY (DENSE MLP) NEURAL NETWORK 
![Python](https://img.shields.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

# Key Architectural Features

* **Zero-Dependency Core:** Numpy is fully used as the main mathematical source.
* **Numerical Stability Engine:** Handles forward and backpropagation like a champ. Has following modes:
                                * **Sigmoid** : Has Overflow clipping, anti-vaneshing mechanism with hybrid RelU style gradients.
                                * **RelU**    : Has fixed dead neuron problems with Leaky RelU.
* **Explicit Gradient-Wise Updates:** Hand-rolled backpropagation executing exact partial derivatives ($\frac{\partial L}{\partial W}$, $\frac{\partial L}{\partial b}$) to optimize weights layer-by-layer.
* **Modular Pipeline Design:** Highly flexible design with various pipeline such as:
                                * **Image Pipeline** : Auto cropping of handwritten numbers using cv2 (format: --> Non_Inverse)
                                * **Zip Pipeline**   : Handles zipped folders `.zip` to extract required data into specific format for training samples.
                                          * - Image extraction (MINST Data)
                                          * - Prediction data (Titanic Data sets and e.t.c)

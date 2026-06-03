# ** ZERO DEPENDENCY (DENSE MLP) NEURAL NETWORK **
[![Python](https://img.shields.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)

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

<h1>PCOSense-AI Based PCOS Detection System</h1>

<h2>Introduction</h2>

PCOS is a hormonal disorder affecting women of reproductive age. It's characterized by irregular periods, excess androgen (male hormone), and the presence of multiple cysts on the ovaries.

1) **irregular periods:** Menstrual cycles can be infrequent, irregular, or absent altogether.
2) **Excess Androgen:** High levels of androgen can lead to acne, excess hair growth, and hair loss.
3) **Cysts on Ovaries:** These cysts are small, fluid-filled sacs that can develop on the ovaries.
4) **Other symptoms:** PCOS can also cause weight gain, infertility, and an increased risk of diabetes and heart disease.

<h2>Dataset Description</h2>

* Dataset link : <a href="https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images/data">PCOS detection using ultrasound images</a>

* The open-source dataset was sourced from Kaggle. 

* The dataset is a collection of ultrasound images of the woman ovaries, categorized into normal and infected with Polycystic Ovarian Syndrome. 

* A total of 3848 ultrasound images were divided into training and testing sets to facilitate model development and evaluation. The training corpus consists of images from which 1,143 are healthy, and 781 admit infection; the test corpus comprises images from which 1,145 are healthy and 787 infected. 


<h2>System Architecture</h2>

![Screenshot 2025-03-15 175512](https://github.com/user-attachments/assets/aed5af25-bba9-4ebe-8daf-980e3919a168)

<h2>Results</h2>

![pcos1](https://github.com/user-attachments/assets/e1e59b4d-bc34-4271-b1a7-af25f08e2fb0)

<br>

![pcos2](https://github.com/user-attachments/assets/c267cd85-5214-41f6-8b3f-5a2f310f2275)


<h2> Limitations</h2>

* Pre-trained Models Usage: The use of pre-trained models like ResNet50, VGG16, and Xception limits the customization to PCOS-specific features. These models are trained on generic datasets (e.g., ImageNet) and may not optimally represent medical imaging characteristics.

* Binary Classification: The project is limited to binary outcomes: "PCOS" or "No PCOS." It does not consider different stages or severity levels of PCOS, which could provide more actionable insights.

* Overfitting Risk: Feature extraction and model training on limited data can lead to overfitting, especially if proper validation techniques are not rigorously applied.

* Generalization Across Populations:If the training dataset is not diverse (e.g., limited to specific demographics), the model may fail to generalize across different populations, leading to biases in predictions.

<h2>Future Scope</h2>

* Expanded Dataset: Collect a larger, more diverse dataset. Including more images from different demographics and capturing varying PCOS stages would improve the model’s robustness.

* Explainable AI (XAI): Incorporate interpretability tools (e.g., Grad-CAM, LIME, or SHAP) to visualize and explain which features of the images influence predictions, enhancing trust in medical settings.

* Multi-class Classification: Extend the project to classify the severity or stages of PCOS, providing more granular insights for clinicians.

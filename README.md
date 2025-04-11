![Abnormal](https://github.com/user-attachments/assets/638733ae-5964-41db-a5c0-c6d5eb1ff8ff)
# Lung-Cancer-Detection-Using-Image-Processing
This project detects lung cancer from CT scan images using image processing and a fine-tuned DeiT model. Trained on 1400 labeled images, it classifies scans as cancerous or non-cancerous with high accuracy. Future plans include real-time deployment and 3D scan support.

#In Simple Terms:-
This project focuses on detecting lung cancer using CT scan images through advanced image processing and deep learning techniques. The goal is to assist in early diagnosis by automatically classifying CT scan images as cancerous or non-cancerous.

#Overview
1.)Dataset: 1400 labeled CT scan images of lungs.
2.)Model Used: DeiT (Data-efficient Image Transformer) from the transformers library by Hugging Face.
3.)Task: Image classification — Detect the presence of lung cancer.
4.)Tools & Libraries: Python, PyTorch, Hugging Face Transformers, NumPy, Matplotlib, Scikit-learn.

#Key Features
1.)Preprocessed CT scan images for consistent input to the model.
2.)Applied data augmentation techniques to improve generalization.
3.)Fine-tuned a pre-trained DeiT model for binary classification.
4.)Achieved promising accuracy and model performance on validation data.
5.)Evaluated using metrics like accuracy, precision, recall, and F1-score.

#Results
1.)High accuracy on test set.
2.)Clear visualization of predictions.
3.)Confusion matrix and classification report included.

#Future Enhancements
1.)Integration with a web or mobile interface for user-friendly diagnosis.
2.)Deployment as a real-time diagnostic tool.
3.)Incorporation of 3D scan data for improved precision.

Project Proposal

### **Team Members & Project Title**

| Avani Joshi | Daisy Phung | Phuong Hoang | Tigist Wujira | William Acosta Lora |
| :---- | :---- | :---- | :---- | :---- |
| Skin Lesion Classifier  \- Audit Fairness  |  |  |  |  |

## Topic & Summary

### **Topic of Interest**

| Our group is investigating using machine learning and computer vision to improve the diagnosis of skin diseases from medical images. Using the Diverse Dermatology Images (DDI) and HAM10000 datasets, we will evaluate whether AI can accurately classify dermatological conditions across individuals with diverse skin tones. By analyzing model accuracy across diverse patient groups and exploring methods to reduce bias, our project aims to contribute to the development of a fair and more reliable model for the healthcare industry. |
| :---- |

### **Summary:  1\. Why Is This Topic Important?  2\. How Will Data Be Used?  3\. Potential Impact**

| Early detection of skin cancer, particularly melanoma, can dramatically improve survival rates, but many people delay seeking dermatological care due to uncertainty about symptoms or limited access to specialists. This problem is intensified by the well-known lack of dermatological image data representing darker-skinned individuals, which contributes to misdiagnoses and poorer outcomes in these communities. Skin abnormalities manifest differently across individuals of different races, skin colors, ages, and other factors, making it especially challenging to train models that account for these nuances. |
| :---- |
| We will analyze two public dermatology image datasets: HAM10000 (10,015 dermatoscopic images across 7 lesion categories, collected at the Medical University of Vienna) and the Diverse Dermatology Images (DDI) dataset (656 clinical photographs from 578 patients, curated by Stanford AIMI specifically to improve the representation of darker skin types. HAM10000 will serve as the primary training set given its size and quality, while DDI will be expanded through data augmentation to address its small size and class imbalance, allowing us to evaluate model performance across both light and dark skin tones. Together, these datasets allow us to train CNN-based classifiers on the seven lesion types and directly test our research question: whether existing deep learning architectures can achieve equitable classification performance (measured via accuracy, precision, and recall/sensitivity) across skin tones. |
| The findings could help encourage patients to have early skin screening, avoiding exposure to procedures that may be more expensive, less accessible or generate higher health issue risk. Instead of direct contact to the skin layers, our model examines skin images and texture patterns to notify patients for potential skin disease. The model assists the patients, and medical professionals in distinguishing between benign and malignant tumor on the skin surface in its early stage, accelerating the treatment process and method decision. |

### **Machine Learning Algorithm(s)**

| Convolutional Neural Network (CNN) | Artificial Neural Network (ANN)- baseline comparison model | Transfer learning/fine-tuning of a pretrained model like EfficientNet |
| :---- | :---- | :---- |

## Research Question

| Can existing deep learning architectures be fine-tuned to classify skin lesions with equitable accuracy and recall across both light and dark skin tones, or does the underrepresentation of darker skin in dermatology datasets fundamentally limit model performance for these populations? |
| :---- |

## Dataset(s)

| Name: Diverse Dermatology Images Source: Stanford University Link: [https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965) | Name: Skin Cancer MNIST: HAM10000 Source: Kaggle Link: [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)  |
| :---- | :---- |

### **Training & Testing Considerations**

| HAM1000 contains 10015 labeled images across seven diagnostic categories, which is large enough to support a standard 72/8/20 train, validation, and test split while still leaving sufficient samples per class for the model to learn meaningful patterns. The DDI dataset original contains only 656 images, which will require data augmentation (ASK FOR HELP FOR THIS) that will generate roughly 15 variants per original image to reach a comparable scale to HAM10000. This combined size will allow us to train deep learning models like CNNs, which typically require thousands of examples to converge reliably, while still holding out enough unseen data in testing to validate generalization and detect overfitting |
| :---- |

## Sources of Bias

| The underrepresentation of darker skin tones in public dermatology datasets and skin cancer screenings, reflecting an overall lack of representation across Fitzpatrick skin types, can increase the likelihood of misdiagnosing or failing to detect malignant skin lesions in individuals with darker skin tones. | Learning models are trained on datasets that underrepresent older adults, despite skin cancer being much more prevalent in seniors.  | A lack of access to specialist care among low-income populations can lead to an underrepresentation of the individuals who may not be able to afford a formal screening for their skin lesion. |
| :---- | :---- | :---- |

### **Mitigation Strategies**

| Supplement the underrepresented HAM10000 dataset with the DDI dataset, which exclusively has darker skin tones, and then we will apply data augmentation to expand its size to be comparable to that of the HAM10000 dataset. This directly increases the volume and diversity of darker skin training examples available to the models. | Prioritize recall (sensitivity) over raw accuracy as our primary evaluation metric, since a high accuracy model that simply predicts the majority benign class would still fail the populations we care the most about. By explicitly tracking recall across skin tones during training and validation, we can detect and correct for the model’s tendency to favor the majority class. |
| :---- | :---- |

## Citations

| Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., … Wicke, M. (2016). TensorFlow: A System for Large-Scale Machine Learning This paper is included in the Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’16). TensorFlow: A system for large-scale machine learning. Retrieved from https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf  | AI Dermatologist. (2025). AI Dermatologist: Skin scanner. Ai-Derm.com. [https://ai-derm.com/?gad\_source=1\&gad\_campaignid=23147301790\&gbraid=0AAAAAq1nAfbuwsx5Ktnf0T7JauNuoCoo7\&gclid=Cj0KCQiAuvTJBhCwARIsAL6DemjKFX-0v1XbNIx7D6BpQ\_EYNMcf059oAmxQc5XgT5nEcW\_kxkxhqHAaApd3EALw\_wcB](https://ai-derm.com/?gad_source=1&gad_campaignid=23147301790&gbraid=0AAAAAq1nAfbuwsx5Ktnf0T7JauNuoCoo7&gclid=Cj0KCQiAuvTJBhCwARIsAL6DemjKFX-0v1XbNIx7D6BpQ_EYNMcf059oAmxQc5XgT5nEcW_kxkxhqHAaApd3EALw_wcB)  | Alipour, N., Burke, T., & Courtney, J. (2024). Skin Type Diversity in Skin Lesion Datasets: A Review. Current dermatology reports, 13(3), 198–210. https://doi.org/10.1007/s13671-024-00440-0 | Cleveland Clinic. (2022, October 17). Skin Lesions: What They Are, Types, Causes & Treatment. Cleveland Clinic. https://my.clevelandclinic.org/health/diseases/24296-skin-lesions |
| :---- | :---- | :---- | :---- |


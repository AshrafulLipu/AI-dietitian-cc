"""
Inference Script for Food Classification Model

Use this to test your trained model on new images
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from timm.models import create_model
import pvt
import pvt_v2
import pandas as pd
import os

class FoodClassifier:
    def __init__(self, model_path, data_path, model_name='pvt_small', device='cuda'):
        """
        Initialize the food classifier
        
        Args:
            model_path: Path to the trained model checkpoint
            data_path: Path to dataset (to load class labels)
            model_name: Name of the model architecture
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load class labels from training CSV
        train_csv = os.path.join(data_path, 'train.csv')
        df = pd.read_csv(train_csv)
        self.classes = sorted(df['Food_Label'].unique())
        self.num_classes = len(self.classes)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Loading model: {model_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.classes}")
        
        # Create model
        self.model = create_model(
            model_name,
            pretrained=False,
            num_classes=self.num_classes,
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, image_path, top_k=5):
        """
        Predict the class of an image
        
        Args:
            image_path: Path to the image file
            top_k: Return top-k predictions
            
        Returns:
            List of tuples (class_name, probability)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.idx_to_label[idx.item()]
            results.append((class_name, prob.item()))
        
        return results
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            List of predictions for each image
        """
        all_predictions = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image)
                batch_images.append(image_tensor)
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            for prob in probabilities:
                top_prob, top_idx = torch.max(prob, dim=0)
                class_name = self.idx_to_label[top_idx.item()]
                all_predictions.append((class_name, top_prob.item()))
        
        return all_predictions


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    # Initialize classifier
    model_path = '/content/checkpoints/food_classification/checkpoint.pth'
    data_path = '/content/food_dataset'
    
    classifier = FoodClassifier(
        model_path=model_path,
        data_path=data_path,
        model_name='pvt_small',
        device='cuda'
    )
    
    # Test on a single image
    test_image = '/content/food_dataset/images/Biriyani/165.jpg'
    
    print("\n" + "="*50)
    print("SINGLE IMAGE PREDICTION")
    print("="*50)
    print(f"Image: {test_image}")
    print("\nTop 5 predictions:")
    
    predictions = classifier.predict(test_image, top_k=5)
    for i, (class_name, prob) in enumerate(predictions, 1):
        print(f"{i}. {class_name}: {prob*100:.2f}%")
    
    # Test on test dataset
    print("\n" + "="*50)
    print("BATCH PREDICTION ON TEST SET")
    print("="*50)
    
    test_csv = os.path.join(data_path, 'test.csv')
    test_df = pd.read_csv(test_csv)
    
    # Test on first 10 images
    test_images = [os.path.join(data_path, 'images', img) for img in test_df['Image_Path'].head(10)]
    true_labels = test_df['Food_Label'].head(10).tolist()
    
    predictions = classifier.predict_batch(test_images)
    
    correct = 0
    for img_path, (pred_class, prob), true_label in zip(test_images, predictions, true_labels):
        is_correct = "✓" if pred_class == true_label else "✗"
        if pred_class == true_label:
            correct += 1
        print(f"{is_correct} {os.path.basename(img_path)}: Predicted={pred_class} ({prob*100:.1f}%), True={true_label}")
    
    print(f"\nAccuracy on sample: {correct}/{len(test_images)} = {correct/len(test_images)*100:.1f}%")
# 📘 Segunda Semana - Entregable: Dataset Preprocesado y Notebook Documentado

## ✅ Objetivo
El propósito de esta etapa es preparar el dataset para tareas de clasificación de imágenes como etapa preliminar del proyecto **FruitScan**, permitiendo establecer un baseline de calidad visual. Se realiza:

- Carga y preprocesamiento del dataset.
- División en conjunto de entrenamiento y validación.
- Visualización de muestras.
- Guardado de metadatos.

---

## 🔧 Paso 1: Importar librerías necesarias

```python
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')
```

---

## 📁 Paso 2: Configuración de rutas y parámetros

```python
data_dir = "/content/drive/MyDrive/FruitScan/data/fruits"
batch_size = 8
img_size = 224
val_split = 0.2
seed = 42
```

---

## 🔄 Paso 3: Transformaciones

```python
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 📦 Paso 4: Carga del dataset

```python
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
print(f"Clases detectadas: {class_names}")
```

---

## 🔀 Paso 5: División en entrenamiento y validación

```python
torch.manual_seed(seed)
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

---

## 🚚 Paso 6: Creación de DataLoaders

```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
```

---

## 🖼️ Paso 7: Visualización de muestras

```python
def show_sample(loader):
    images, labels = next(iter(loader))
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        img = images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Desnormalizar
        ax[i].imshow(img)
        ax[i].set_title(class_names[labels[i]])
        ax[i].axis("off")
    plt.show()

show_sample(train_loader)
```

---

## 💾 Paso 8: Guardado de metadatos (opcional)

```python
torch.save({
    'class_names': class_names
}, '/content/drive/MyDrive/FruitScan/data/fruitscan_meta.pth')
```

---

## 📝 Paso 9: Comentario final

> En futuras etapas, este dataset se extenderá con anotaciones para tareas de **detección de objetos**.  
> En esta fase inicial, se emplea clasificación por carpetas para establecer una base funcional de control de calidad visual.

✅ **Estado del entregable:**  
El dataset fue correctamente cargado, preprocesado, dividido y visualizado. Está listo para entrenar modelos de clasificación como baseline en el proyecto **FruitScan**.

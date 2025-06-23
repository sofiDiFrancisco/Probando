import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --- 1. Configuración del Dispositivo ---
# Detecta si hay una GPU disponible y la utiliza; de lo contrario, usa la CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Definición del Modelo (Debe ser idéntica a la del entrenamiento) ---
# Esta función carga un modelo ResNet34 y sus pesos entrenados.
# Es crucial que la arquitectura de la capa final (fc) coincida con la que se usó
# durante el entrenamiento, ya que fue adaptada a tus clases específicas de frutas.
def load_resnet34_model(num_classes, model_path):
    # Carga el modelo ResNet34. 'pretrained=False' porque cargaremos nuestros propios pesos.
    model = models.resnet34(pretrained=False)
    
    # Reemplaza la capa final completamente conectada (fc) para que coincida con el número de clases.
    # Asegúrate de que esta modificación sea EXACTAMENTE la misma que se hizo durante el entrenamiento.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Carga los pesos entrenados del archivo .pth
    try:
        # map_location='cpu' asegura que el modelo se cargue correctamente incluso si fue
        # entrenado en GPU y se ejecuta en CPU (o viceversa).
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Pone el modelo en modo de evaluación (deshabilita dropout, batchnorm, etc.)
        st.success("✅ Modelo ResNet34 cargado exitosamente.")
        return model
    except FileNotFoundError:
        st.error(f"❌ Error: El archivo del modelo '{model_path}' no se encontró. Asegúrate de que el modelo esté entrenado y guardado en la misma carpeta que 'app.py'.")
        st.stop() # Detiene la ejecución de la aplicación si el modelo no se encuentra.
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}. Verifica que el archivo no esté corrupto y que la arquitectura coincida.")
        st.stop()

# --- 3. Parámetros y Transformaciones de Imagen (Igual que en el entrenamiento) ---
# Tamaño de imagen al que se redimensionarán todas las imágenes para la entrada del modelo.
img_size = 224
# Las clases deben estar en el MISMO ORDEN en que fueron indexadas durante el entrenamiento.
# ImageFolder de torchvision suele ordenar las carpetas alfabéticamente.
# Por ejemplo, si tus carpetas son 'fresh' y 'rotten', entonces ['fresh', 'rotten'] es probable.
class_names = ['fresh', 'rotten'] # ¡Ajusta esto si tus nombres de clase o su orden son diferentes!

# Define las transformaciones que se aplicarán a la imagen antes de la predicción.
# Son idénticas a las usadas durante el entrenamiento para asegurar consistencia.
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)), # Redimensiona la imagen
    transforms.ToTensor(), # Convierte la imagen a un tensor de PyTorch (escala a [0, 1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normaliza con la media de ImageNet
                         std=[0.229, 0.224, 0.225])  # Normaliza con la desviación estándar de ImageNet
])

# --- 4. Cargar el Modelo Entrenado ---
# Ruta al archivo de pesos del modelo. Asegúrate de que este archivo exista.
model_path_file = "fruitscan_model_resnet34.pth"
# Carga el modelo a la memoria y lo mueve al dispositivo seleccionado (GPU/CPU).
model = load_resnet34_model(len(class_names), model_path_file)
model.to(device)


# --- 5. Función de Predicción ---
# Esta función toma una imagen PIL, la preprocesa y realiza la predicción.
def predict_image(image):
    # Aplica las transformaciones a la imagen y añade una dimensión de lote (batch dimension).
    # .unsqueeze(0) cambia la forma de (C, H, W) a (1, C, H, W).
    image_tensor = transform(image).unsqueeze(0) 
    # Mueve el tensor de la imagen al dispositivo de procesamiento (GPU/CPU).
    image_tensor = image_tensor.to(device)
    
    # Deshabilita el cálculo de gradientes para la inferencia, lo que ahorra memoria y acelera la predicción.
    with torch.no_grad():
        outputs = model(image_tensor) # Pasa la imagen a través del modelo.
        # Aplica softmax para obtener probabilidades de confianza.
        probabilities = torch.softmax(outputs, dim=1)[0]
        # Obtiene el índice de la clase con la probabilidad más alta.
        _, predicted_idx = torch.max(outputs, 1)
        
    # Obtiene el nombre de la clase predicha a partir de su índice.
    predicted_class = class_names[predicted_idx.item()]
    # Obtiene la confianza de la predicción y la convierte a porcentaje.
    confidence = probabilities[predicted_idx.item()].item() * 100
    
    return predicted_class, confidence

# --- 6. Interfaz de Usuario de Streamlit ---

# Configuración básica de la página de Streamlit.
st.set_page_config(
    page_title="Clasificador de Frutas", # Título que aparece en la pestaña del navegador
    page_icon="🍎", # Icono de la pestaña del navegador
    layout="centered", # Diseño de la página: 'centered' o 'wide'
    initial_sidebar_state="auto" # Estado inicial de la barra lateral
)

# Título principal de la aplicación.
st.title("🍎 Clasificador de Frutas")
# Descripción de la aplicación.
st.markdown("Sube una imagen de una fruta para clasificarla como **fresca** o **podrida** usando un modelo **ResNet34** de PyTorch.")

# Sección de la barra lateral (sidebar) para información adicional.
with st.sidebar:
    st.header("Acerca de")
    st.info(
        "Esta aplicación web utiliza un modelo de **red neuronal convolucional (ResNet34)** "
        "pre-entrenado en ImageNet y ajustado para clasificar imágenes de frutas en dos categorías: **fresca** o **podrida**.\n\n"
        "El objetivo principal es servir como un prototipo para sistemas de control de calidad automatizados en la industria alimentaria, "
        "ayudando a identificar la calidad de las frutas de manera eficiente."
    )
    st.markdown("---")
    st.markdown("Desarrollado con ❤️.")


# Widget para subir archivos. Acepta imágenes JPG, JPEG y PNG.
uploaded_file = st.file_uploader("Elige una imagen de fruta...", type=["jpg", "jpeg", "png"])

# Lógica principal cuando se sube un archivo.
if uploaded_file is not None:
    # Intenta abrir la imagen subida.
    try:
        image = Image.open(uploaded_file)
        # Muestra la imagen subida en la interfaz.
        st.image(image, caption='Imagen subida', use_column_width=True)
        st.write("") # Espacio en blanco para mejor presentación
        st.write("🔎 Clasificando la imagen, por favor espera...")

        # Realiza la predicción usando el modelo.
        predicted_class, confidence = predict_image(image)

        # Muestra los resultados de la predicción.
        st.subheader("Resultados de la Clasificación:")
        st.success(f"**Predicción:** La fruta es **{predicted_class.upper()}**")
        st.info(f"**Confianza:** {confidence:.2f}%")

        # Mensajes adicionales basados en la predicción.
        if "rotten" in predicted_class.lower():
            st.warning("¡Advertencia! Esta fruta parece estar podrida. 🤢 No es apta para el consumo.")
        elif "fresh" in predicted_class.lower():
            st.balloons() # Pequeña animación de globos para una predicción positiva.
            st.success("¡Excelente! Esta fruta parece estar fresca y en buen estado. 🎉")
        else:
            # En caso de una clase no esperada (poco probable si class_names está bien definido).
            st.error("⚠️ Predicción desconocida. Hubo un problema al determinar la clase.")
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}. Asegúrate de subir un archivo de imagen válido.")
else:
    # Mensaje inicial cuando no hay ningún archivo subido.
    st.info("Por favor, sube una imagen para que el modelo pueda analizarla y clasificarla.")

st.markdown("---")
st.markdown("Si tienes problemas, verifica que el archivo `fruitscan_model_resnet34.pth` esté en la misma carpeta que este script.")

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

# ==========================================
# 1. CONFIGURACION
# ==========================================

RUTA_DATASET = "dataset/balones/basketball.npy"
CARPETA_RESULTADOS = "resultados"

TAMANO_LOTE = 128
EPOCAS = 10
TAMANO_RUIDO = 100
CANTIDAD_DIBUJOS = 20000

DISPOSITIVO = torch.device("cpu")

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

print("DISPOSITIVO:", DISPOSITIVO)
print("Cargando dibujos de baloncesto...")

# ==========================================
# 2. CARGAR DATASET
# ==========================================

datos = np.load(RUTA_DATASET, mmap_mode="r")

cantidad = min(CANTIDAD_DIBUJOS, len(datos))

imagenes = np.array(datos[:cantidad], dtype=np.float32)

# Normalizar los pixeles de [0,255] a [-1,1]
imagenes = imagenes / 127.5 - 1.0

imagenes = torch.tensor(imagenes)

dataset = TensorDataset(imagenes)
cargador = DataLoader(
    dataset,
    batch_size=TAMANO_LOTE,
    shuffle=True
)

print("Dibujos utilizados:", cantidad)
print("Dimensiones:", imagenes.shape)

# ==========================================
# 3. GENERADOR
# ==========================================

class Generador(nn.Module):

    def __init__(self):
        super().__init__()

        self.modelo = nn.Sequential(
            nn.Linear(TAMANO_RUIDO, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, ruido):
        return self.modelo(ruido)


# ==========================================
# 4. DISCRIMINADOR
# ==========================================

class Discriminador(nn.Module):

    def __init__(self):
        super().__init__()

        self.modelo = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(self, imagen):
        return self.modelo(imagen)


generador = Generador().to(DISPOSITIVO)
discriminador = Discriminador().to(DISPOSITIVO)

# ==========================================
# 5. FUNCION DE PERDIDA Y OPTIMIZADORES
# ==========================================

funcion_perdida = nn.BCEWithLogitsLoss()

optimizador_G = optim.Adam(
    generador.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizador_D = optim.Adam(
    discriminador.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

perdidas_G = []
perdidas_D = []

# ==========================================
# 6. ENTRENAMIENTO DE LA GAN
# ==========================================

print("\nINICIANDO ENTRENAMIENTO DE LA GAN\n")

for epoca in range(EPOCAS):

    suma_perdida_G = 0
    suma_perdida_D = 0
    suma_real = 0
    suma_falso = 0

    for (imagenes_reales,) in cargador:

        imagenes_reales = imagenes_reales.to(DISPOSITIVO)
        lote_actual = imagenes_reales.size(0)

        etiquetas_reales = torch.ones(
            lote_actual, 1, device=DISPOSITIVO
        )

        etiquetas_falsas = torch.zeros(
            lote_actual, 1, device=DISPOSITIVO
        )

        # ----------------------------------
        # ENTRENAR DISCRIMINADOR
        # ----------------------------------

        ruido = torch.randn(
            lote_actual, TAMANO_RUIDO, device=DISPOSITIVO
        )

        imagenes_falsas = generador(ruido)

        salida_real = discriminador(imagenes_reales)
        salida_falsa = discriminador(imagenes_falsas.detach())

        perdida_real = funcion_perdida(
            salida_real, etiquetas_reales
        )

        perdida_falsa = funcion_perdida(
            salida_falsa, etiquetas_falsas
        )

        perdida_D = (perdida_real + perdida_falsa) / 2

        optimizador_D.zero_grad()
        perdida_D.backward()
        optimizador_D.step()

        # ----------------------------------
        # ENTRENAR GENERADOR
        # ----------------------------------

        for parametro in discriminador.parameters():
            parametro.requires_grad_(False)

        ruido = torch.randn(
            lote_actual, TAMANO_RUIDO, device=DISPOSITIVO
        )

        imagenes_generadas = generador(ruido)
        salida_generada = discriminador(imagenes_generadas)

        perdida_G = funcion_perdida(
            salida_generada, etiquetas_reales
        )

        optimizador_G.zero_grad()
        perdida_G.backward()
        optimizador_G.step()

        for parametro in discriminador.parameters():
            parametro.requires_grad_(True)

        # ----------------------------------
        # REGISTRAR RESULTADOS
        # ----------------------------------

        suma_perdida_D += perdida_D.item()
        suma_perdida_G += perdida_G.item()

        suma_real += torch.sigmoid(salida_real).mean().item()
        suma_falso += torch.sigmoid(salida_falsa).mean().item()

    promedio_D = suma_perdida_D / len(cargador)
    promedio_G = suma_perdida_G / len(cargador)

    promedio_real = suma_real / len(cargador)
    promedio_falso = suma_falso / len(cargador)

    perdidas_D.append(promedio_D)
    perdidas_G.append(promedio_G)

    print(f"Epoca {epoca + 1}/{EPOCAS}")
    print(f"Perdida Discriminador: {promedio_D:.4f}")
    print(f"Perdida Generador: {promedio_G:.4f}")
    print(f"Discriminador - dibujos reales: {promedio_real:.4f}")
    print(f"Discriminador - dibujos falsos: {promedio_falso:.4f}")
    print("-" * 45)


# ==========================================
# 7. GENERAR NUEVOS DIBUJOS
# ==========================================

generador.eval()
discriminador.eval()

with torch.no_grad():

    ruido = torch.randn(
        64, TAMANO_RUIDO, device=DISPOSITIVO
    )

    nuevas_imagenes = generador(ruido)

    # Guardar imagenes generadas
    ruta_imagenes = os.path.join(
        CARPETA_RESULTADOS, "balones_generados.png"
    )

    save_image(
        nuevas_imagenes.reshape(-1, 1, 28, 28),
        ruta_imagenes,
        nrow=8,
        normalize=True,
        value_range=(-1, 1)
    )

    # Mostrar valores de pixeles
    primera_imagen = nuevas_imagenes[0]
    pixeles = ((primera_imagen + 1) * 127.5).clamp(0, 255)

    print("\nPRIMEROS 20 VALORES DE PIXELES:")
    print(pixeles[:20].cpu().numpy())

    # Mostrar puntuacion del discriminador
    puntuaciones = torch.sigmoid(
        discriminador(nuevas_imagenes)
    )

    print("\nPUNTUACIONES DEL DISCRIMINADOR:")
    print(puntuaciones[:10].cpu().numpy())


# ==========================================
# 8. GRAFICAR LAS PERDIDAS
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(perdidas_G, label="Perdida del Generador")
plt.plot(perdidas_D, label="Perdida del Discriminador")

plt.xlabel("Epocas")
plt.ylabel("Perdida")
plt.title("Entrenamiento de la GAN - Baloncesto")

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(CARPETA_RESULTADOS, "grafica_perdidas.png")
)

plt.close()


# ==========================================
# 9. GUARDAR MODELOS
# ==========================================

torch.save(
    generador.state_dict(),
    os.path.join(CARPETA_RESULTADOS, "generador.pth")
)

torch.save(
    discriminador.state_dict(),
    os.path.join(CARPETA_RESULTADOS, "discriminador.pth")
)

print("\nENTRENAMIENTO FINALIZADO")
print("Resultados guardados en:", CARPETA_RESULTADOS)

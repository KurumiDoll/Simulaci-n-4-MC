import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
N = 50  # Número de espines
J = 1.0  # Interacción entre espines
k_B = 1.0  # Constante de Boltzmann
temperaturas = np.linspace(0.5, 5.0, 20)  # Rango de temperaturas
Campos = [0.1, 0.5, 1.0]  # Valores del campo magnético

def calcular_energia(espines, J, B):
    E = -J * np.sum(espines * np.roll(espines, 1)) - B * np.sum(espines)
    return E

def paso_metropolis(espines, T, J, B):
    for _ in range(N):
        i = np.random.randint(N)
        delta_E = 2 * J * espines[i] * (espines[(i-1)] + espines[(i+1)%N]) + 2 * B * espines[i]
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
            espines[i] *= -1
    return espines

# Función para realizar la simulación y calcular promedios
def simulacion_ising(T, B, pasos=1000):
    espines = np.ones(N)  # Configuración inicial
    energia_total = []
    magnetizacion_total = []
    for _ in range(1000):
        paso_metropolis(espines, T, J, B)
        energia_total.append(calcular_energia(espines, J, B))
        magnetizacion_total.append(np.sum(espines))
    
    energia_media = np.mean(energia_total) / N
    magnetizacion_media = np.mean(magnetizacion_total) / N
    return energia_media, magnetizacion_media

# Inicializar gráfica
plt.figure(figsize=(12, 6))

# Subplot para Energía vs Temperatura
plt.subplot(1, 2, 1)
for B in Campos:
    energias = []
    for T in temperaturas:
        energia, _ = simulacion_ising(T, B)
        energias.append(energia)
    plt.plot(temperaturas, energias, label=f'B = {B}')
plt.xlabel('Temperatura (T)')
plt.ylabel('Energía')
plt.title("Energía vs Temperatura")
plt.legend()

# Subplot para Magnetización vs Temperatura
plt.subplot(1, 2, 2)
for B in Campos:
    magnetizaciones = []
    for T in temperaturas:
        _, magnetizacion = simulacion_ising(T, B)
        magnetizaciones.append(magnetizacion)
    plt.plot(temperaturas, magnetizaciones, label=f'B = {B}')
plt.xlabel('Temperatura (T)')
plt.ylabel('Magnetización')
plt.title("Magnetización vs Temperatura")
plt.legend()

# Mostrar gráfica
plt.tight_layout()
plt.show()

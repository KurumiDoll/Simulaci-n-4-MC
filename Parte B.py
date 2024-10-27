#INCISO B
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
N = 50  # Número de espines
J = 1.0  # Interacción entre espines
k_B = 1.0  # Constante de Boltzmann
temperaturas = np.linspace(0.1, 5.0, 10)  # Rango de temperaturas
campos_magneticos = [0]  # Valores del campo magnético

def calcular_energia(espines, J, h=0):
    E = -J * np.sum(espines * np.roll(espines, 1)) - h * np.sum(espines)
    return E

def paso_metropolis(espines, T,J, h=0):
    for _ in range(N):
        i = np.random.randint(N)
        delta_E = 2 * J * espines[i] * (espines[(i-1)] + espines[(i+1)%N]) + 2 * h * espines[i]
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
            espines[i] *= -1
    return espines

# Función para realizar la simulación y calcular promedios
def simulacion_ising(T, h=0, pasos=1000):
    espines = np.ones(N)  # Configuración inicial
    energia_total = []
    magnetizacion_total = []
    for _ in range(pasos * N):
        paso_metropolis(espines, T, h)
        energia_total.append(calcular_energia(espines, J, h))
        magnetizacion_total.append(np.sum(espines))
    
    energia_media = np.mean(energia_total)
    magnetizacion_media = np.mean(np.abs(magnetizacion_total))
    sus_magnetica= (np.var(magnetizacion_total))/(k_B*T)
    capacidad_c= (np.var(energia_total)) / (k_B * T**2)
    return energia_media, magnetizacion_media, sus_magnetica, capacidad_c

energias = []
magnetizaciones = []
susceptibilidades= []
capacidad_calorifica = []
e_exacta= []
for T in temperaturas:
    energia, magnetizacion, susceptibilidad, capacidad = simulacion_ising(T, 0)
    energias.append(energia)
    magnetizaciones.append(magnetizacion)
    susceptibilidades.append(susceptibilidad)
    capacidad_calorifica.append(capacidad)
    e_exacta.append(-N * np.tanh(1 / T))
    
# Grafica # 
plt.figure(figsize=(10, 8))
plt.plot(temperaturas, energias, label='Energía')
plt.plot(temperaturas, e_exacta, label="Energia Exacta", color="red", linestyle="dashed")
plt.xlabel('Temperatura (T)')
plt.ylabel('Energía media')
plt.legend()
plt.show()

#Magnetización"
plt.figure(figsize=(10, 8))
plt.plot(temperaturas, magnetizaciones, label='Magnetización')
plt.xlabel('Temperatura (T)')
plt.ylabel('Magnetización media')
plt.legend()
plt.show()

#Sus"
plt.figure(figsize=(10, 8))
plt.plot(temperaturas, susceptibilidades, label='Susceptibilidad')
plt.xlabel('Temperatura (T)')
plt.ylabel('Susceptibilidad')
plt.legend()
plt.show()

#Cap_C"
plt.figure(figsize=(10, 8))
plt.plot(temperaturas, magnetizaciones, label='Capacidad Calorifica')
plt.xlabel('Temperatura (T)')
plt.ylabel('Capacidad Calorifica')
plt.legend()
plt.show()

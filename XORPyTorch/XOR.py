import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Configuración para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)


class XORNet(nn.Module):
    """
    Una red neuronal para resolver el problema XOR utilizando PyTorch.
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        """
        Inicializa la arquitectura de la red.

        Args:
            input_size: Número de neuronas en la capa de entrada
            hidden_size: Número de neuronas en la capa oculta
            output_size: Número de neuronas en la capa de salida
        """
        super(XORNet, self).__init__()

        # Definición de capas
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

        # Inicialización de pesos (Xavier/Glorot)
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Propagación hacia adelante.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor de salida
        """
        return self.model(x)


def train_xor_network(model, x_data, y_data, learning_rate=0.01,
                      epochs=1000, batch_size=None, early_stopping=True, patience=100):
    """
    Entrena la red neuronal con los datos proporcionados.

    Args:
        model: Modelo de PyTorch a entrenar
        x_data: Datos de entrada
        y_data: Datos de salida esperados
        learning_rate: Tasa de aprendizaje
        epochs: Número máximo de épocas
        batch_size: Tamaño del lote (None = usar todos los datos)
        early_stopping: Si se debe detener temprano cuando no hay mejora
        patience: Número de épocas sin mejora antes de detenerse

    Returns:
        history: Historial de pérdidas durante el entrenamiento
    """
    # Convertir datos a tensores
    x = torch.FloatTensor(x_data)
    y = torch.FloatTensor(y_data)

    # Definir función de pérdida y optimizador
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Parámetros para early stopping
    best_loss = float('inf')
    no_improvement = 0

    # Historial de entrenamiento
    history = {'loss': [], 'accuracy': []}

    # Bucle de entrenamiento
    for epoch in range(epochs):
        # Poner el modelo en modo de entrenamiento
        model.train()

        # Propagación hacia adelante
        outputs = model(x)

        # Calcular pérdida
        loss = criterion(outputs, y)

        # Calcular precisión
        with torch.no_grad():
            predicted = (outputs > 0.5).float()
            accuracy = accuracy_score(y.numpy(), predicted.numpy())

        # Actualizar historial
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)

        # Mostrar progreso periódicamente
        if epoch % 100 == 0:
            print(f'Época {epoch}/{epochs}, Pérdida: {loss.item():.6f}, Precisión: {accuracy:.4f}')

        # Early stopping
        if early_stopping:
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= patience and accuracy > 0.99:
                print(f'Early stopping en época {epoch}')
                break

        # Retropropagación y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Imprimir resultado final
    print(f'Entrenamiento finalizado. Pérdida final: {loss.item():.6f}, Precisión: {accuracy:.4f}')

    return history


def visualize_training(history):
    """
    Visualiza el progreso del entrenamiento.

    Args:
        history: Historial de entrenamiento con pérdidas y precisión
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de pérdida
    ax1.plot(history['loss'])
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.grid(True)

    # Gráfico de precisión
    ax2.plot(history['accuracy'])
    ax2.set_title('Precisión durante el entrenamiento')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('xor_training_progress.png')
    plt.show()


def visualize_decision_boundary(model, resolution=100):
    """
    Visualiza la frontera de decisión del modelo.

    Args:
        model: Modelo entrenado
        resolution: Resolución de la cuadrícula para visualización
    """
    # Crear cuadrícula de puntos
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))

    # Evaluar modelo en cada punto
    model.eval()
    with torch.no_grad():
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(grid).reshape(xx.shape)

    # Crear gráfico
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z.numpy(), levels=20, cmap='RdBu', alpha=0.5)
    plt.contour(xx, yy, Z.numpy(), levels=[0.5], colors='black')

    # Agregar puntos de entrenamiento
    colors = ['red', 'blue', 'blue', 'red']
    markers = ['o', 'o', 'o', 'o']
    labels = ['0', '1', '1', '0']

    for i, (x, y, color, marker, label) in enumerate(zip(
            [0, 0, 1, 1], [0, 1, 0, 1], colors, markers, labels)):
        plt.scatter(x, y, color=color, marker=marker, s=100, edgecolors='black')
        plt.text(x + 0.05, y + 0.05, label, fontsize=12)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Frontera de decisión de la red para XOR')
    plt.grid(True)
    plt.savefig('xor_decision_boundary.png')
    plt.show()


def main():
    """Función principal para ejecutar el ejemplo completo."""
    # Datos XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print("Problema XOR:")
    for i in range(len(X)):
        print(f"Entrada: {X[i]} -> Salida esperada: {y[i][0]}")

    # Crear modelo
    model = XORNet(input_size=2, hidden_size=4, output_size=1)
    print(f"\nEstructura del modelo:\n{model}")

    # Entrenar modelo
    print("\nIniciando entrenamiento...")
    history = train_xor_network(
        model, X, y,
        learning_rate=0.05,
        epochs=2000,
        early_stopping=True
    )

    # Evaluar modelo
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X)
        predictions = model(inputs)

    print("\nEvaluación del modelo:")
    for i in range(len(X)):
        print(f"Entrada: {X[i]} -> Predicción: {predictions[i].item():.4f}, Esperado: {y[i][0]}")

    # Visualizar resultados
    visualize_training(history)
    visualize_decision_boundary(model)

    # Guardar modelo
    torch.save(model.state_dict(), 'xor_model.pt')
    print("\nModelo guardado como 'xor_model.pt'")


# Ejecutar si se llama como script principal
if __name__ == "__main__":
    main()
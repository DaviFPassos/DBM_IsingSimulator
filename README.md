# Modelo de Ising 2D + Deep Boltzmann Machine (DBM)

Projeto completo de simulação Monte Carlo do modelo de Ising 2D usando o algoritmo de Wolff, seguido de aprendizado não-supervisionado com Deep Boltzmann Machine (DBM) para gerar novas configurações de spin.

## 📋 Descrição

Este projeto combina **física computacional** e **aprendizado de máquina** em duas etapas:

1. **Simulação de Monte Carlo**: Gera configurações realistas do modelo de Ising usando o algoritmo de Wolff
2. **Deep Boltzmann Machine**: Aprende a distribuição de probabilidade dos spins e gera novas configurações

### Pipeline do Projeto

```
Simulação Monte Carlo (Wolff) → Dados de Treino (.npy) → DBM → Configurações Geradas
```

## 🎯 O que o Projeto Faz?

### Parte 1: Simulação de Ising (Monte Carlo)
- Simula sistema de spins 2D próximo à transição de fase
- Usa algoritmo de cluster de Wolff para amostragem eficiente
- Gera **~80 configurações independentes** após termalização
- Salva dados em formato `.npy` para treinamento

### Parte 2: Deep Boltzmann Machine
- Treina rede neural probabilística de 2 camadas
- **Camada 1**: 16384 → 1024 neurônios (extração de features)
- **Camada 2**: 1024 → 64 neurônios (representação latente)
- Aprende distribuição de spins sem supervisão
- Gera novas configurações ("fantasy particles")

## 🚀 Características

✅ Algoritmo de Wolff para geração de dados físicos realistas  
✅ DBM com pré-treinamento camada por camada  
✅ Reconstrução de configurações de entrada  
✅ Geração de novas configurações (amostragem de Gibbs)  
✅ Visualização comparativa (dados reais vs. reconstruídos vs. gerados)  
✅ Suporte a GPU para treinamento acelerado  
✅ Salvamento completo de modelos e resultados  

## 📦 Requisitos

```bash
pip install numpy matplotlib pillow torch torchvision seaborn
```

**Nota**: Para GPU, instale PyTorch com suporte CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🎮 Como Usar

### Passo 1: Gerar Dados com Monte Carlo

```python
python ising_simulation.py
```

**Saída**: `ising_data/ising_configurations_T2.3_L128.npy`

### Passo 2: Treinar DBM

```python
python dbm_ising.py
```

**Saídas**:
- Imagem comparativa: `DBM_Ising-critical_L128.png`
- Modelo treinado: `ising_data/DBM_ising_training_data_T2.3_L=128.npy`

## 📊 Estrutura dos Arquivos

```
.
├── ising_simulation.py              # Simulação Monte Carlo
├── dbm_ising.py                     # Treinamento DBM
├── ising_data/                      # Dados e modelos
│   ├── ising_configurations_T2.3_L128.npy      # Dados simulados
│   └── DBM_ising_training_data_T2.3_L=128.npy  # Modelo treinado
├── wolff_ising_simulation_T2.3_L128.gif        # Animação da simulação
└── DBM_Ising-critical_L128.png                 # Resultados visuais
```

## 🔬 Detalhes Técnicos

### Parâmetros da Simulação

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `L` | 128 | Tamanho da rede (128×128 = 16384 spins) |
| `T` | 2.3 | Temperatura (próximo de Tc ≈ 2.269) |
| `n_steps` | 5000 | Passos de Monte Carlo |
| `save_interval` | 25 | Intervalo de salvamento (após termalização) |

### Arquitetura da DBM

```
Camada de Entrada (Visível):  16384 neurônios (128×128 spins)
        ↓
Camada Oculta 1:              1024 neurônios
        ↓
Camada Oculta 2:              64 neurônios (representação latente)
```

### Parâmetros de Treinamento

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `batch_size` | 15 | Tamanho do lote |
| `num_epochs` | 500 | Épocas por camada |
| `learning_rate` | 0.007 | Taxa de aprendizado (Adam) |
| `num_fantasy_steps` | 250 | Passos de amostragem de Gibbs |

## 🧠 Como Funciona a DBM?

### 1. Pré-treinamento Camada por Camada

#### Camada 1 (RBM1)
```
v (dados) → h1 (features)
```
- Aprende features básicas dos dados
- Treina com **Contrastive Divergence**

#### Camada 2 (RBM2)
```
h1 → h2 (representação latente)
```
- Aprende representação compacta
- Usa saídas de RBM1 como entrada

### 2. Reconstrução

```
v (entrada) → h1 → h2 → h1' → v' (reconstruída)
```
Testa capacidade do modelo de reproduzir os dados.

### 3. Geração de Fantasy Particles

```
v (seed) → [250 passos de Gibbs sampling] → v_gen (nova configuração)
```
Gera configurações completamente novas a partir da distribuição aprendida.

## 📈 Análise dos Resultados

### Carregar e Analisar Modelo Treinado

```python
import numpy as np
import torch

# Carregar dados salvos
data = np.load('ising_data/DBM_ising_training_data_T2.3_L=128.npy', 
               allow_pickle=True)

dbm_models = data[0]
true_examples = data[1]
fantasy_samples = data[2]
reconstructions = data[3]

print(f"Configurações reais: {true_examples['critical'].shape}")
print(f"Configurações geradas: {fantasy_samples['critical'].shape}")
```

### Calcular Magnetização

```python
# Dados reais
real_configs = true_examples['critical']
real_mag = np.mean(real_configs, axis=1)

# Dados gerados
gen_configs = fantasy_samples['critical']
gen_mag = np.mean(gen_configs, axis=1)

print(f"Magnetização média (real): {np.mean(np.abs(real_mag)):.3f}")
print(f"Magnetização média (gerada): {np.mean(np.abs(gen_mag)):.3f}")
```

### Comparar Distribuições

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(real_mag, bins=20, alpha=0.7, label='Real', edgecolor='black')
plt.xlabel('Magnetização')
plt.ylabel('Frequência')
plt.title('Dados Reais')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(gen_mag, bins=20, alpha=0.7, label='Gerada', color='orange', edgecolor='black')
plt.xlabel('Magnetização')
plt.title('Dados Gerados (DBM)')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(real_mag, bins=20, alpha=0.5, label='Real', edgecolor='black')
plt.hist(gen_mag, bins=20, alpha=0.5, label='Gerada', color='orange', edgecolor='black')
plt.xlabel('Magnetização')
plt.title('Comparação')
plt.legend()

plt.tight_layout()
plt.show()
```

## 🎨 Visualização

A imagem gerada (`DBM_Ising-critical_L128.png`) contém 3 linhas:

1. **data**: Configurações reais da simulação Monte Carlo
2. **reconst**: Reconstruções da DBM (teste de aprendizado)
3. **fantasy**: Novas configurações geradas pela DBM

### Criar Visualizações Personalizadas

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plotar configuração individual
config = fantasy_samples['critical'][0].reshape(128, 128)

plt.figure(figsize=(8, 8))
sns.heatmap(config, cmap='coolwarm', cbar=True, square=True)
plt.title('Configuração Gerada pela DBM')
plt.axis('off')
plt.show()
```

## 🔧 Personalização

### Mudar Temperatura da Simulação

```python
# Em ising_simulation.py
T = 2.0  # Fase ordenada (ferromagnética)
T = 2.3  # Próximo do ponto crítico
T = 3.0  # Fase desordenada (paramagnética)
```

### Ajustar Arquitetura da DBM

```python
# Em dbm_ising.py
num_hidden_units = [2048, 128]  # DBM mais expressiva
num_hidden_units = [512, 32]    # DBM mais compacta
```

### Aumentar Tamanho da Rede

```python
L = 256  # Rede 256×256 (requer mais memória)
num_hidden_units = [4096, 256]  # Ajustar proporcionalmente
```

## 📊 Métricas de Avaliação

### 1. Erro de Reconstrução

```python
mse = np.mean((true_examples['critical'] - reconstructions['critical'])**2)
print(f"MSE de Reconstrução: {mse:.4f}")
```

### 2. Função de Correlação

```python
def correlation_function(config):
    """Calcula correlação spin-spin"""
    L = int(np.sqrt(len(config)))
    spins = config.reshape(L, L)
    corr = []
    for r in range(1, L//2):
        c = np.mean(spins * np.roll(spins, r, axis=0))
        corr.append(c)
    return np.array(corr)

real_corr = correlation_function(true_examples['critical'][0])
gen_corr = correlation_function(fantasy_samples['critical'][0])

plt.plot(real_corr, label='Real')
plt.plot(gen_corr, label='Gerada', linestyle='--')
plt.xlabel('Distância r')
plt.ylabel('Correlação ⟨S(0)S(r)⟩')
plt.legend()
plt.show()
```

## 🎓 Conceitos de Machine Learning

### Restricted Boltzmann Machine (RBM)

Modelo probabilístico não-supervisionado que aprende:
```
P(v, h) = exp(-E(v, h)) / Z
```

Onde:
- `v`: neurônios visíveis (dados)
- `h`: neurônios ocultos (features)
- `E(v, h)`: função energia

### Contrastive Divergence (CD)

Algoritmo de treinamento que:
1. Calcula gradiente aproximado da log-verossimilhança
2. Usa amostragem de Gibbs truncada (k-steps)
3. Minimiza diferença entre dados e modelo

### Amostragem de Gibbs

Processo iterativo alternado:
```
v → h → v → h → ... (250 passos)
```

Converge para amostras da distribuição aprendida `P(v)`.

## 📚 Aplicações

Este projeto demonstra:

1. **Simulação Física**: Geração de configurações de equilíbrio térmico
2. **Aprendizado Generativo**: Modelagem de distribuições complexas
3. **Redução de Dimensionalidade**: 16384 → 64 dimensões
4. **Data Augmentation**: Geração de novos dados para treinamento

### Extensões Possíveis

- [ ] Treinar em múltiplas temperaturas
- [ ] Comparar com GANs ou VAEs
- [ ] Calcular função de partição `Z`
- [ ] Estimar temperatura crítica via DBM
- [ ] Aplicar a outros sistemas físicos (Potts, XY model)

## 🐛 Solução de Problemas

### Erro de Memória (GPU)

```python
# Reduzir batch size ou hidden units
batch_size = 10
num_hidden_units = [512, 32]
```

### Convergência Lenta

```python
# Aumentar learning rate ou epochs
learning_rate = 0.01
num_epochs = 1000
```

### Fantasy Particles Ruins

```python
# Aumentar passos de Gibbs
num_fantasy_steps = 500
```

## 📖 Referências

### Física Estatística
1. **Wolff, U.** (1989). "Collective Monte Carlo Updating for Spin Systems". *Phys. Rev. Lett.*
2. **Onsager, L.** (1944). "Crystal Statistics". *Phys. Rev.*

### Machine Learning
3. **Salakhutdinov, R. & Hinton, G.** (2009). "Deep Boltzmann Machines". *AISTATS*.
4. **Hinton, G.** (2002). "Training Products of Experts by Minimizing Contrastive Divergence". *Neural Computation*.
5. **Carleo, G. & Troyer, M.** (2017). "Solving the quantum many-body problem with artificial neural networks". *Science*.

## 📝 Citação

Se usar este código em pesquisa, considere citar:

```bibtex
@misc{ising_dbm_2024,
  title={Ising Model Simulation and Deep Boltzmann Machine Learning},
  author={Seu Nome},
  year={2024},
  publisher={GitHub},
  url={https://github.com/seu-usuario/seu-repositorio}
}
```

## 👤 Contato

Desenvolvido para estudos de física computacional e aprendizado de máquina aplicado à física estatística.

---

**Nota**: Este projeto combina métodos clássicos de Monte Carlo com técnicas modernas de deep learning para explorar sistemas físicos complexos.

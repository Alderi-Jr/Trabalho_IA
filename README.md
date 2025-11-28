# 🧟‍♂️ Apocalypse RL – Q-Learning com Mapas Aleatórios

Este projeto implementa um ambiente pós-apocalipse utilizando **Gym**, **PyGame** e **Q-Learning**, onde um agente deve sobreviver, coletar suprimentos, evitar zumbis e chegar até a zona segura.  
O ambiente é totalmente aleatório a cada execução, e o treinamento é exibido visualmente via HUD em tempo real.

> Baseado nos arquivos reais do projeto:
> - `environment.py`
> - `qlearning.py`
> - `main.py`

---

## 📌 Funcionalidades Principais

- ✔️ Geração **totalmente aleatória** de mapas 10×10  
- ✔️ Paredes, rochas, suprimentos, zumbis e safe zone distribuídos **sem sobreposição**  
- ✔️ Ambiente compatível com **Gym**  
- ✔️ Renderização completa via **PyGame** com HUD lateral  
- ✔️ Agente aprende via **Q-Learning**  
- ✔️ Replay do melhor episódio encontrado  
- ✔️ Controles interativos:
  - **R** → Reproduzir melhor episódio  
  - **N** → Novo mapa + treinar  
  - **SPACE** → Avançar um passo  
  - **ESC** → Sair  

---

## 🗺️ Ambiente – `environment.py`

O ambiente (`ApocalypseEnvironment`) funciona com:

### 🔹 Observação
Tupla com 5 valores:  
`(x, y, f0, f1, f2)`  
Onde `f0-f2` são flags binárias indicando suprimentos coletados.

### 🔹 Ações
| ID | Significado |
|----|-------------|
| 0  | ↑           |
| 1  | →           |
| 2  | ↓           |
| 3  | ←           |

### 🔹 Recompensas
| Situação                          | Recompensa |
|----------------------------------|------------|
| Movimento normal                 | -0.2       |
| Coletar suprimento               | +30        |
| Safe zone sem suprimentos        | -20        |
| Safe zone com todos suprimentos  | +120       |
| Encontrar zumbi                  | -15 (terminal) |

### 🔹 Renderização PyGame
- Tabuleiro em grid 10×10  
- Sprites: chão, parede, rocha, suprimento, zumbi, agente e safe zone  
- HUD com:
  - Passos
  - Recompensa total
  - Episódio atual
  - Melhor recompensa
  - Estado dos suprimentos coletados
  - Última ação

---

## 🤖 Q-Learning – `qlearning.py`

Implementação de Q-Learning com:

### 🔹 Estrutura da Q-table

Q[grid_x][grid_y][f0][f1][f2][action]

## 🔹 Parâmetros principais
- `learning_rate = 0.1`
- `discount_factor = 0.93`
- `exploration_rate` com decaimento automático

### 🔹 Armazenamento do melhor episódio REAL
O código salva:

- Melhor recompensa já obtida
- Sequência REAL de ações do melhor episódio → `best_actions`

### 🔹 Critérios de parada
- Recompensa ≥ 205  
- Ou "paciência" de 150 episódios sem melhora

---

## 🎯 Loop Principal – `main.py`

O arquivo `main.py` faz:

### ✔️ Treinamento visual
Chamando `train_with_hud()` que vai atualizando o PyGame durante cada episódio.

### ✔️ Salvamento automático
Após o treinamento:
q_table_stateflags.pkl
best_actions.pkl


### ✔️ Replay
Executa o melhor caminho encontrado:
- Mostrado passo a passo no PyGame
- Respeita teclas R / N / SPACE / ESC

### ✔️ Novo mapa completo
Tecla **N** gera outro mapa totalmente aleatório e reinicia treinamento.

---

## 🧱 Estrutura do Projeto

/
├── environment.py
├── qlearning.py
├── main.py
├── assets/
│ ├── agent.png
│ ├── zombie.png
│ ├── floor.png
│ ├── rock.png
│ ├── wall.png
│ ├── supply.png
│ └── safe.png
└── README.md


---

## ▶️ Como Executar

### 1️⃣ Instalar dependências
```bash
pip install pygame gym numpy
python main.py


🧪 Comportamento Esperado

O agente nasce no canto superior esquerdo

Coleta suprimentos espalhados

Desvia de paredes, rochas e zumbis

Vai até a safe zone

HUD mostra toda a evolução do treinamento

Replay permite visualizar o melhor trajeto real

🚀 Melhorias Futuras

Usar DQN (Deep Q-Learning)

Criar mapas conectados via BFS/DFS

Inserir múltiplos agentes

Inserir diferentes tipos de inimigos

Balancear recompensas

Geração procedural mais inteligente

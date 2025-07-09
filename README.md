# BitMaze - Evolutionary Neural Network Pathfinding Game

An AI-powered evolutionary simulation where small creatures learn to navigate mazes using neural networks and genetic algorithms. Watch as generations of creatures evolve to become better at finding food while avoiding obstacles!

## 🎮 What is BitMaze?

BitMaze is an interactive evolutionary simulation that demonstrates machine learning concepts through gameplay. Small colorful creatures (dots) start with random neural networks and gradually evolve over generations to become smarter at pathfinding. Each generation learns from the successes and failures of the previous one, creating increasingly intelligent behavior.

## 🧠 How It Works

### Neural Networks
- Each creature has its own neural network brain
- **Inputs:** Distance to food, nearest obstacle location, wall distances
- **Outputs:** Movement directions (X and Y velocity)
- **Architecture:** 8 inputs → 12 hidden neurons → 2 outputs

### Evolutionary Algorithm
- **Population:** 50 creatures per generation
- **Selection:** Tournament selection favors fitter individuals
- **Crossover:** Top performers breed to create offspring
- **Mutation:** Random changes introduce genetic diversity
- **Elitism:** Best 5 creatures always survive to next generation

### Fitness Function
Creatures are scored based on:
- ✅ **+1000 points** for reaching food
- ✅ **+2 points** per pixel closer to food
- ✅ **+0.5 points** per frame stayed alive
- ✅ **Efficiency bonus** for direct paths
- ❌ **-200 points** for hitting obstacles or walls

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Required Libraries
```bash
pip install pygame numpy
```

### Running the Game
1. Clone or download this repository
2. Navigate to the BitMaze directory
3. Run the game:
```bash
python BitMaze.py
```

## 🎯 How to Play

### Setup Phase
1. **Set Spawn Point** - Left-click where creatures should start
2. **Place Food** - Right-click to place the green food target
3. **Draw Obstacles** - Middle-click to create red obstacle blocks
4. **Start Simulation** - Press SPACE to begin evolution

### During Simulation
- **Watch Evolution** - Observe creatures learning over generations
- **Add Obstacles** - Middle-click to add more challenges
- **Control Speed** - Use UP/DOWN arrows to adjust simulation speed
- **Pause/Resume** - Press SPACE to pause or resume
- **Reset** - Press R to start over

## 🎮 Controls

| Control | Action |
|---------|--------|
| **Left Click** | Place creature spawn point (setup only) |
| **Right Click** | Place food target (setup only) |
| **Middle Click** | Draw obstacles (setup & during simulation) |
| **SPACE** | Start simulation / Pause / Resume |
| **UP Arrow** | Increase simulation speed |
| **DOWN Arrow** | Decrease simulation speed |
| **R** | Reset entire simulation |
| **ESC** | Quit game |

## ⚡ Speed Settings

Choose from 6 different simulation speeds:
- **Slow** (15 FPS) - Watch individual creature behavior
- **Normal** (30 FPS) - Standard observation speed
- **Fast** (60 FPS) - Default speed
- **Very Fast** (120 FPS) - Quick evolution viewing
- **Ultra Fast** (240 FPS) - Rapid generation progression
- **Maximum** (480 FPS) - Fastest possible evolution

## 📊 Understanding the Interface

### Real-time Statistics
- **Generation Number** - Current evolutionary generation
- **Population Stats** - Alive, dead, and successful creatures
- **Timer** - Time remaining in current generation
- **Best Fitness** - Highest score achieved in current generation
- **Current Speed** - Active simulation speed setting

### Visual Indicators
- 🔵 **Blue Circle** - Creature spawn point
- 🟢 **Green Squares** - Food targets
- 🔴 **Red Squares** - Deadly obstacles
- 🎨 **Colored Dots** - Individual creatures (various colors)
- 🟢 **Green Creatures** - Successfully reached food
- 🔴 **Red Creatures** - Died from obstacle collision

## 🧬 Evolutionary Progress

### What to Expect
- **Generation 1-5:** Random, chaotic movement
- **Generation 5-15:** Basic obstacle avoidance emerges
- **Generation 15-30:** Directional movement toward food
- **Generation 30+:** Efficient pathfinding and sophisticated navigation

### Tips for Better Evolution
- Create challenging but solvable mazes
- Allow multiple paths to the food
- Use obstacles to create interesting navigation problems
- Run at higher speeds to see long-term evolution trends

## 🔧 Customization

### Evolution Parameters (in code)
```python
POPULATION_SIZE = 50        # Creatures per generation
GENERATION_TIME = 10        # Seconds per generation
MUTATION_RATE = 0.1         # Probability of genetic mutation
CROSSOVER_RATE = 0.7        # Probability of breeding
ELITE_SIZE = 5              # Top performers kept each generation
```

### Neural Network Parameters
```python
INPUT_SIZE = 8              # Sensor inputs to brain
HIDDEN_SIZE = 12            # Hidden layer neurons
OUTPUT_SIZE = 2             # Movement outputs
```

## 🎓 Educational Value

BitMaze demonstrates key concepts in:
- **Machine Learning:** Neural networks, supervised learning
- **Artificial Intelligence:** Pathfinding, decision making
- **Evolutionary Computation:** Genetic algorithms, natural selection
- **Game Development:** Real-time simulation, interactive systems

Perfect for:
- Students learning AI/ML concepts
- Educators demonstrating evolution and learning
- Anyone curious about how AI learns to solve problems
- Game developers interested in emergent behavior

## 🐛 Troubleshooting

### Common Issues
- **Black screen:** Ensure Background.jpg exists in Assets folder
- **Slow performance:** Lower population size or reduce speed
- **No evolution:** Check that obstacles allow possible paths to food
- **Crashes:** Verify pygame and numpy are properly installed

### Performance Tips
- Start with simple mazes for faster convergence
- Use higher speeds to observe long-term trends
- Reset if population gets stuck in local optimum

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements! Some ideas:
- Better neural network architectures
- Additional sensor inputs
- Different selection algorithms
- Enhanced visualization features
- Performance optimizations

---

**Watch evolution in action! 🧬 Create your maze and see AI learn to solve it!**

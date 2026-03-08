# Mental Map Quick Reference

## What is the Mental Map?

The commander's internal belief about the environment state. It starts empty (only knows building structure) and learns through agent observations.

## Starting the Visualization

```bash
cd /Users/adyashapatra/Desktop/urban_seismic_resilience
python app.py
# Open browser: http://localhost:5002
# Click "Initialize" button
```

## Side-by-Side Display

**LEFT MAP (Ground Truth)**
- Shows actual state of environment
- All buildings, fires, victims, blockages visible
- This is what really happened

**RIGHT MAP (Mental Map)**
- Shows what commander believes
- Black areas = unexplored
- Only hazards/victims that agents reported
- This is what commander used to make decisions

## Visual Legend

| Color/Element | Meaning |
|---|---|
| **Black** | Unexplored (unknown to commander) |
| **Light Gray** | Clear road (known to be passable) |
| **Dark Gray** | Collapsed building |
| **Gray-Blue** | Standing building |
| **Brown** | Blocked road |
| **Orange/Red** | Fire (known location) |
| **Yellow** | Victims (known location) |
| **Cyan dot** | Scout agent |
| **Red dot** | Firefighter agent |
| **Green dot** | Medic agent |

## Key Observations to Make

1. **Exploration Progress**: Watch black areas shrink as agents explore
2. **Information Gaps**: Notice areas still unexplored after many steps
3. **Hidden Threats**: When ground truth shows fire in unexplored area of mental map
4. **Decision Impact**: Delays in response due to lack of information
5. **Agent Paths**: Follow how agents discover areas in order

## Understanding the Metrics

### In Mental Map Title:
- **Exploration: XX.X%** - Percentage of grid visited by agents
- **Known Victims: N** - Count of victims in explored areas

### Comparison:
- Ground truth shows actual victims (some hidden in unexplored areas)
- Mental map shows only victims agents have found
- Difference = **undetected victims**

## Common Scenarios

### Scenario 1: Early Exploration Phase
- Ground truth: Fires and victims everywhere
- Mental map: Mostly black except where scouts have been
- Commander strategy: Send scouts to quickly explore

### Scenario 2: Partial Information
- Ground truth: Fire spreading to unexplored area
- Mental map: Fire not visible yet
- Result: Firefighter might not be sent there until fire spreads to known area

### Scenario 3: Aftershock Event
- Ground truth: New collapses create blockages
- Mental map: Old paths thought to be clear
- Effect: Agent might take longer route due to stale information

## Making the Most of the Visualization

### Enable Fastest Rendering:
In app.py, set in config:
```python
'max_steps': 20  # Shorter simulation
'controller_mode': 'hierarchical'
'commander_type': 'heuristic'  # Faster than LLM
```

### Compare Performance:
1. Run with heuristic commander - watch decisions made with limited info
2. Run with more/fewer agents - see how exploration speed affects mental map
3. Run with different building densities - observe info complexity

### Research Ideas:
- How many steps before commander has ~50% exploration?
- What's the impact of information delay on rescue rates?
- How would communication between agents improve decisions?

## Files to Check

| File | Purpose |
|---|---|
| `env/mental_map.py` | Mental map implementation |
| `agents/commander.py` | How commander uses mental map |
| `env/environment.py` | Rendering code (lines 450-537) |
| `app.py` | Backend serving both maps |
| `templates/index.html` | Frontend display |
| `test_mental_map.py` | Verify functionality |

## Troubleshooting

**Mental map not showing?**
- Make sure you're using hierarchical mode with a commander
- Check browser console for JavaScript errors
- Verify backend is rendering the image

**Mental map looks wrong?**
- Check that field agents are sending observations
- Look for errors in `update_mental_map()` calls
- Verify agents have non-zero observation radius

**Performance issues?**
- Reduce grid size: `grid_width: 20, grid_height: 20`
- Reduce render frequency or increase update_delay_ms in JavaScript
- Use heuristic commander instead of LLM

## Advanced Usage

### Custom Analysis:
```python
# In Python REPL
from env import MentalMap, Grid

grid = Grid(30, 30)
grid.generate_city_layout()

mental_map = MentalMap(30, 30)
mental_map.initialize_from_grid(grid)

# Check exploration
print(mental_map.get_explored_fraction())

# Get uncertainty map
uncertainty = mental_map.get_uncertainty_map()
print(f"Avg staleness: {uncertainty.mean()}")
```

### Render Manually:
```python
from env import UrbanDisasterEnv

env = UrbanDisasterEnv()
env.reset()

# Render mental map to file
if env.commander and env.commander.mental_map:
    env.render_mental_map(
        env.commander.mental_map,
        save_path='mental_map.png'
    )
```

## Contact/Questions

See `MENTAL_MAP_GUIDE.md` for detailed documentation and `MENTAL_MAP_IMPLEMENTATION.md` for technical details.

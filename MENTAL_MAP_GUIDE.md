"""
Mental Map Integration Guide
=============================

This document describes the mental map visualization feature that has been
added to the urban seismic resilience simulation.

## Overview

The mental map is the commander's internal representation of the grid environment.
It gets initialized with the same structural information as the actual grid
(buildings, roads, cell types), but does NOT contain dynamic information about
hazards, victims, or blockages until field agents report these observations.

## Key Components Added

### 1. MentalMap Class (env/mental_map.py)
- MentalMapCell: Tracks individual cells with explored/unexplored status
- MentalMap: Main class managing commander's beliefs
  - Initialized with same structure as Grid
  - Only updated from field agent observations
  - Tracks information staleness
  - Can generate zone summaries based on known information

Key Methods:
- initialize_from_grid(grid): Copy static structural info from actual grid
- update_from_observation(): Update cells based on agent sensor data
- get_uncertainty_map(): Return staleness of information
- get_explored_fraction(): Overall exploration percentage
- get_zone_summary(): Summary of known hazards/victims in a zone
- shortest_path(): Pathfinding based on beliefs (may differ from reality)

### 2. Commander Integration (agents/commander.py)
- CommanderAgent now has mental_map attribute
- initialize_mental_map(): Called during controller setup
- update_mental_map(): Called each step to incorporate agent reports

The commander makes decisions based on its mental map, not ground truth.
This creates realistic information asymmetry.

### 3. Visualization (env/environment.py & templates/index.html)
- New render_mental_map() method in UrbanDisasterEnv
- Shows:
  * Dark cells = Unexplored areas
  * Light cells = Known roads/clear areas
  * Building colors = Known structural state
  * Fire/victim markers = Observed hazards
  * Exploration percentage in title

### 4. Web Interface Updates (app.py & templates/index.html)
- Side-by-side display of:
  * Ground Truth (actual state)
  * Mental Map (commander's beliefs)
- Mental map updates in real-time as agents report
- Shows which areas are explored vs. unexplored

## How It Works

### Simulation Flow:
1. Environment is initialized (actual grid is created)
2. Commander's mental map is initialized:
   - Same dimensions as actual grid
   - Buildings and road layouts are known
   - Dynamic state (victims, fires, blockages) is UNKNOWN
3. Each simulation step:
   - Field agents observe their surroundings
   - Agents send observation reports to message bus
   - Commander receives reports and updates mental map
   - Commander uses mental map to make decisions
4. Visualization:
   - Ground truth map shows actual state
   - Mental map shows what commander believes

### Information Flow:
```
Field Agents
    ↓ (send observations)
Message Bus
    ↓ (delivers to commander)
Commander.update_mental_map()
    ↓ (updates mental map from observations)
Commander.decide()
    ↓ (uses mental map beliefs)
Task Assignments
```

## Field Agent Reports

Field agents include observation data in their reports:
- position: Agent's current location
- observation: Local grid observation (2D array of cell info)
- observation_radius: Radius of observation
- victims_seen: Count of victims observed
- building_collapse: Structural damage observations

This information is used to update the mental map.

## Information Uncertainty

The mental map tracks:
- **explored**: Whether a cell has been observed
- **last_updated_step**: When the cell was last observed
- **staleness**: Current step - last update step

This allows the system to:
- Distinguish known from unknown areas
- Measure information age
- Prioritize re-exploration of stale information

## Decision Making

The commander uses mental map data to:
- Calculate zone summaries (only from explored areas)
- Find paths that avoid known blockages
- Assess victim/fire distribution in known areas
- Issue task assignments to agents

The mental map may differ from reality:
- Unknown areas might have hidden victims
- Blockages might have cleared
- Fires might have spread to unexplored areas

This creates realistic decision-making under incomplete information.

## Visualization Tips

When viewing the mental map:
1. **Black areas**: Unexplored regions
2. **Light gray**: Roads that are believed to be clear
3. **Darker gray**: Blocked or collapsed areas
4. **Yellow dots**: Known victim locations
5. **Red/orange areas**: Known fires
6. **Title bar**: Shows exploration percentage and known victims

Compare ground truth and mental map to see:
- What the commander knows vs. what's actually happening
- Why certain decisions were made
- Information gaps and surprises

## Configuration

To enable mental map visualization:
1. Use hierarchical controller mode (default)
2. Set up a commander (heuristic or LLM)
3. Run the app.py web interface
4. Both maps will be displayed side by side

## Performance Considerations

- Mental map updates are O(radius²) for each observation
- Pathfinding uses A* on the belief graph (same as actual grid)
- Rendering adds ~50-100ms per frame
- No significant performance impact from mental map

## Future Enhancements

Potential improvements:
- Decay of old observations (information becomes less reliable over time)
- Probability distributions instead of binary known/unknown
- Partial information about cells (e.g., "fire suspected but not confirmed")
- Communication between agents sharing observations
- Bayesian belief updates
"""

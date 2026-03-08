"""
Test script to verify mental map functionality.
Tests initialization, updates from agent observations, and uncertainty tracking.
"""

from env.grid import Grid, CellType, HazardType
from env.mental_map import MentalMap


def test_mental_map_initialization():
    """Test that mental map initializes with same structure as grid."""
    print("Test 1: Mental Map Initialization")
    print("-" * 50)
    
    # Create a real grid
    grid = Grid(width=20, height=20, seed=42)
    grid.generate_city_layout(building_density=0.2)
    grid.place_victims(10)
    grid.place_fires(5)
    
    # Create mental map
    mental_map = MentalMap(width=20, height=20, seed=42)
    mental_map.initialize_from_grid(grid)
    
    # Verify dimensions match
    assert mental_map.width == grid.width, "Width mismatch"
    assert mental_map.height == grid.height, "Height mismatch"
    print(f"✓ Dimensions match: {mental_map.width}x{mental_map.height}")
    
    # Verify building structure is copied
    assert len(mental_map.buildings) == len(grid.buildings), "Building count mismatch"
    print(f"✓ Building count matches: {len(mental_map.buildings)}")
    
    # Verify static information (cell types) are known
    for pos, grid_cell in grid.cells.items():
        mental_cell = mental_map.cells[pos]
        assert mental_cell.cell_type == grid_cell.cell_type, f"Cell type mismatch at {pos}"
    print(f"✓ Cell types match for all {len(grid.cells)} cells")
    
    # Verify dynamic information is NOT copied (initially unknown)
    unexplored_count = sum(1 for cell in mental_map.cells.values() if not cell.explored)
    assert unexplored_count == len(mental_map.cells), "Some cells marked as explored"
    print(f"✓ All {unexplored_count} cells initially unexplored")
    
    # Verify no victims in mental map (not yet observed)
    victim_count = sum(len(cell.victims) for cell in mental_map.cells.values())
    assert victim_count == 0, "Mental map should not have victims initially"
    print(f"✓ No victims in mental map initially (ground truth has {sum(len(cell.victims) for cell in grid.cells.values())})")
    
    print("\n✅ Test 1 PASSED: Mental map initialized correctly\n")
    return grid, mental_map


def test_observation_update():
    """Test that mental map updates from agent observations."""
    print("Test 2: Observation Update")
    print("-" * 50)
    
    # Create grid and mental map
    grid = Grid(width=20, height=20, seed=42)
    grid.generate_city_layout(building_density=0.2)
    grid.place_victims(10)
    grid.place_fires(5)
    
    mental_map = MentalMap(width=20, height=20, seed=42)
    mental_map.initialize_from_grid(grid)
    
    # Simulate agent observation at position (10, 10) with radius 2
    agent_pos = (10, 10)
    obs_radius = 2
    observation = grid.get_local_observation(agent_pos[0], agent_pos[1], obs_radius)
    
    print(f"Agent at {agent_pos} observes {2*obs_radius+1}x{2*obs_radius+1} area")
    
    # Update mental map
    mental_map.update_from_observation(agent_pos, observation, obs_radius, step=1)
    
    # Verify explored cells
    explored_count = sum(1 for cell in mental_map.cells.values() if cell.explored)
    expected_explored = (2 * obs_radius + 1) ** 2
    print(f"✓ Explored cells: {explored_count} (expected ~{expected_explored})")
    
    # Verify exploration fraction
    exploration_fraction = mental_map.get_explored_fraction()
    print(f"✓ Exploration fraction: {exploration_fraction:.2%}")
    assert exploration_fraction > 0, "No exploration recorded"
    
    # Verify observed area info is updated
    for dy in range(-obs_radius, obs_radius + 1):
        for dx in range(-obs_radius, obs_radius + 1):
            pos = (agent_pos[0] + dx, agent_pos[1] + dy)
            if pos in mental_map.cells:
                mental_cell = mental_map.cells[pos]
                assert mental_cell.explored, f"Cell {pos} should be explored"
                assert mental_cell.last_updated_step == 1, f"Cell {pos} update step incorrect"
    print(f"✓ All cells in observation radius marked as explored")
    
    # Verify hazard information is updated
    hazard_count_in_obs = 0
    for dy in range(-obs_radius, obs_radius + 1):
        for dx in range(-obs_radius, obs_radius + 1):
            pos = (agent_pos[0] + dx, agent_pos[1] + dy)
            if pos in grid.cells and grid.cells[pos].hazard == HazardType.FIRE:
                hazard_count_in_obs += 1
                # Check mental map has this info
                mental_cell = mental_map.cells[pos]
                assert mental_cell.hazard == HazardType.FIRE, f"Fire at {pos} not in mental map"
    
    if hazard_count_in_obs > 0:
        print(f"✓ {hazard_count_in_obs} fires correctly recorded in mental map")
    else:
        print(f"✓ No fires in observed area (as expected)")
    
    print("\n✅ Test 2 PASSED: Mental map updates from observations\n")
    return grid, mental_map


def test_uncertainty_tracking():
    """Test that mental map tracks information staleness."""
    print("Test 3: Uncertainty Tracking")
    print("-" * 50)
    
    grid = Grid(width=20, height=20, seed=42)
    grid.generate_city_layout(building_density=0.2)
    
    mental_map = MentalMap(width=20, height=20, seed=42)
    mental_map.initialize_from_grid(grid)
    
    # Agent observes area at step 1
    pos1 = (5, 5)
    obs1 = grid.get_local_observation(pos1[0], pos1[1], radius=1)
    mental_map.update_from_observation(pos1, obs1, 1, step=1)
    
    # Agent observes different area at step 5
    pos2 = (15, 15)
    obs2 = grid.get_local_observation(pos2[0], pos2[1], radius=1)
    mental_map.update_from_observation(pos2, obs2, 1, step=5)
    
    # Get uncertainty map at step 10
    mental_map.current_step = 10
    uncertainty = mental_map.get_uncertainty_map()
    
    # Check staleness for first observed area (observed at step 1, now step 10)
    staleness_pos1 = uncertainty[pos1[1], pos1[0]]
    assert staleness_pos1 == 9, f"Staleness at {pos1} should be 9, got {staleness_pos1}"
    print(f"✓ Staleness at {pos1} (observed at step 1): {staleness_pos1} steps")
    
    # Check staleness for second observed area (observed at step 5, now step 10)
    staleness_pos2 = uncertainty[pos2[1], pos2[0]]
    assert staleness_pos2 == 5, f"Staleness at {pos2} should be 5, got {staleness_pos2}"
    print(f"✓ Staleness at {pos2} (observed at step 5): {staleness_pos2} steps")
    
    # Check unexplored areas have -1
    unexplored_pos = (10, 10)
    if not mental_map.cells[unexplored_pos].explored:
        staleness_unexplored = uncertainty[unexplored_pos[1], unexplored_pos[0]]
        assert staleness_unexplored == -1, "Unexplored cells should have -1 staleness"
        print(f"✓ Unexplored cell {unexplored_pos} has staleness -1")
    
    print("\n✅ Test 3 PASSED: Uncertainty tracking works correctly\n")


def test_zone_summary():
    """Test zone summary with mental map."""
    print("Test 4: Zone Summary")
    print("-" * 50)
    
    grid = Grid(width=30, height=30, seed=42)
    grid.generate_city_layout(building_density=0.2)
    grid.place_fires(5)
    
    mental_map = MentalMap(width=30, height=30, seed=42)
    mental_map.initialize_from_grid(grid)
    
    # Agent explores zone (0, 0) to (10, 10)
    for x in range(0, 10, 3):
        for y in range(0, 10, 3):
            obs = grid.get_local_observation(x, y, radius=2)
            mental_map.update_from_observation((x, y), obs, 2, step=1)
    
    # Get zone summary
    zone_summary = mental_map.get_zone_summary(0, 0, zone_size=10)
    
    print(f"Zone (0, 0) summary:")
    print(f"  Known victims: {zone_summary['victims_known']}")
    print(f"  Known fires: {zone_summary['fires_known']}")
    print(f"  Exploration: {zone_summary['exploration']:.1%}")
    print(f"  Cells explored: {zone_summary['cells_explored']}/{zone_summary['total_cells']}")
    
    assert zone_summary['exploration'] > 0, "Zone should have some exploration"
    assert zone_summary['exploration'] <= 1.0, "Exploration should be <= 100%"
    print(f"✓ Zone summary generated correctly")
    
    # Compare with unexplored zone
    unexplored_summary = mental_map.get_zone_summary(20, 20, zone_size=10)
    print(f"\nUnexplored zone (20, 20) summary:")
    print(f"  Exploration: {unexplored_summary['exploration']:.1%}")
    
    assert unexplored_summary['exploration'] < zone_summary['exploration'], \
        "Unexplored zone should have lower exploration"
    print(f"✓ Unexplored zone has lower exploration rate")
    
    print("\n✅ Test 4 PASSED: Zone summaries work correctly\n")


def test_pathfinding_with_mental_map():
    """Test that pathfinding works on mental map's believed graph."""
    print("Test 5: Pathfinding with Mental Map")
    print("-" * 50)
    
    grid = Grid(width=20, height=20, seed=42)
    grid.generate_city_layout(building_density=0.1)
    
    mental_map = MentalMap(width=20, height=20, seed=42)
    mental_map.initialize_from_grid(grid)
    
    # Find two road positions
    road_positions = [pos for pos, cell in grid.cells.items() 
                     if cell.cell_type == CellType.ROAD]
    
    if len(road_positions) >= 2:
        start = road_positions[0]
        end = road_positions[-1]
        
        # Get path from mental map (before knowing about blockages)
        path_mental = mental_map.shortest_path(start, end)
        
        if path_mental:
            print(f"✓ Path found from {start} to {end}: {len(path_mental)} steps")
            assert path_mental[0] == start, "Path should start at start position"
            assert path_mental[-1] == end, "Path should end at end position"
        else:
            print(f"✓ No path found (disconnected road network)")
    else:
        print("✓ Insufficient road cells for path test")
    
    print("\n✅ Test 5 PASSED: Pathfinding works on mental map\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MENTAL MAP TEST SUITE")
    print("="*70 + "\n")
    
    try:
        test_mental_map_initialization()
        test_observation_update()
        test_uncertainty_tracking()
        test_zone_summary()
        test_pathfinding_with_mental_map()
        
        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()

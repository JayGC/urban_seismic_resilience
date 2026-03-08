"""
Test script to debug mental map updates via message bus.
Prints what agents are sending and what the commander receives.
"""

import json
from controller import SimulationController

def test_message_flow():
    """Test the message flow from agents to commander."""
    
    # Create a simple config
    config = {
        'grid_width': 30,
        'grid_height': 30,
        'building_density': 0.2,
        'num_victims': 10,
        'num_fires': 5,
        'max_steps': 5,
        'agents': {
            'num_scouts': 2,
            'num_firefighters': 1,
            'num_medics': 1,
        },
        'controller_mode': 'hierarchical',
        'commander_type': 'heuristic',
        'seed': 42,
    }
    
    # Create controller
    ctrl = SimulationController(config)
    ctrl.setup()
    
    print("="*80)
    print("MENTAL MAP DEBUG TEST")
    print("="*80)
    
    # Get initial state
    print("\n[INITIAL STATE]")
    print(f"Commander has mental map: {ctrl.commander.mental_map is not None}")
    if ctrl.commander.mental_map:
        print(f"Mental map size: {ctrl.commander.mental_map.width}x{ctrl.commander.mental_map.height}")
        print(f"Initial exploration: {ctrl.commander.mental_map.get_explored_fraction():.2%}")
    
    # Run a few steps
    for step in range(5):
        print(f"\n{'='*80}")
        print(f"STEP {step}")
        print('='*80)
        
        # Run step
        step_data = ctrl.run_step()
        
        # Check message bus
        print("\n[MESSAGE BUS - Pending Messages]")
        all_messages = ctrl.message_bus.receive_all()
        
        if all_messages:
            print(f"Total messages on bus: {len(all_messages)}")
            for i, msg in enumerate(all_messages):
                print(f"\nMessage {i+1}:")
                print(f"  From: {msg.sender}")
                print(f"  Type: {msg.msg_type.name}")
                print(f"  Step: {msg.step}")
                
                if msg.msg_type.name == 'REPORT':
                    metadata = msg.metadata
                    print(f"  Position: {metadata.get('position')}")
                    print(f"  Obs Radius: {metadata.get('observation_radius')}")
                    
                    findings = metadata.get('findings', {})
                    if findings:
                        print(f"  Findings:")
                        print(f"    - Fires: {findings.get('fires', [])}")
                        print(f"    - Victims: {findings.get('victims', [])}")
                        print(f"    - Blocked roads: {findings.get('blocked_roads', [])}")
                        print(f"    - Collapsed buildings: {findings.get('collapsed_buildings', [])}")
        else:
            print("No messages on message bus")
        
        # Check commander's mental map state
        print("\n[COMMANDER MENTAL MAP STATE]")
        if ctrl.commander.mental_map:
            mm = ctrl.commander.mental_map
            print(f"Exploration: {mm.get_explored_fraction():.2%}")
            
            # Count explored cells
            explored = sum(1 for c in mm.cells.values() if c.explored)
            print(f"Explored cells: {explored}/{len(mm.cells)}")
            
            # Count known hazards
            fires_known = sum(1 for c in mm.cells.values() if c.hazard and c.hazard.name == 'FIRE')
            victims_known = sum(len(c.victims) for c in mm.cells.values())
            blockages_known = sum(1 for c in mm.cells.values() if c.blocked)
            
            print(f"Known fires: {fires_known}")
            print(f"Known victims: {victims_known}")
            print(f"Known blocked roads: {blockages_known}")
            
            # Show a sample of explored cells
            explored_cells = [(pos, c) for pos, c in mm.cells.items() if c.explored]
            if explored_cells:
                print(f"\nSample explored cells:")
                for pos, cell in explored_cells[:5]:
                    hazard_name = cell.hazard.name if cell.hazard else 'NONE'
                    print(f"  {pos}: blocked={cell.blocked}, hazard={hazard_name}, "
                          f"fire_intensity={cell.fire_intensity if cell.fire_intensity else 0:.1f}, num_victims={len(cell.victims)}")
        
        # Check agent reports stored in commander
        print("\n[COMMANDER AGENT REPORTS]")
        if ctrl.commander.agent_reports:
            for agent_id, report in ctrl.commander.agent_reports.items():
                print(f"{agent_id}:")
                print(f"  Step: {report['step']}")
                findings = report['metadata'].get('findings', {})
                if findings:
                    print(f"  Findings: fires={len(findings.get('fires', []))}, "
                          f"victims={len(findings.get('victims', []))}, "
                          f"blocked={len(findings.get('blocked_roads', []))}, "
                          f"collapsed={len(findings.get('collapsed_buildings', []))}")
        
        # Check actual grid state
        print("\n[ACTUAL GRID STATE (Ground Truth)]")
        grid = ctrl.env.grid
        actual_fires = sum(1 for c in grid.cells.values() if c.hazard and c.hazard.name == 'FIRE')
        actual_victims = sum(len(c.victims) for c in grid.cells.values())
        actual_blocked = sum(1 for c in grid.cells.values() if c.blocked)
        actual_collapsed = sum(1 for b in grid.buildings.values() if b.collapsed)
        
        print(f"Actual fires: {actual_fires}")
        print(f"Actual victims: {actual_victims}")
        print(f"Actual blocked roads: {actual_blocked}")
        print(f"Actual collapsed buildings: {actual_collapsed}")
        
        # Print agent positions
        print("\n[AGENT POSITIONS]")
        for agent_id, pos in ctrl.env.agent_positions.items():
            print(f"  {agent_id}: {pos}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_message_flow()

import os
import io
import base64
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify

from controller import SimulationController

app = Flask(__name__)
global_ctrl = None

@app.route('/')
def index():
    # Flask automatically looks in the 'templates' folder for index.html
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global global_ctrl
    config_path = os.path.join('configs', 'low_hazard.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'grid_width': 30, 'grid_height': 30,
            'building_density': 0.25, 'num_victims': 30, 'num_fires': 8,
            'max_steps': 100,
            'agents': {'num_scouts': 4, 'num_firefighters': 4, 'num_medics': 5},
            'controller_mode': 'hierarchical', 'commander_type': 'heuristic'
        }
    global_ctrl = SimulationController(config)
    global_ctrl.setup()
    return jsonify({"status": "started"})

@app.route('/step', methods=['GET'])
def step():
    global global_ctrl
    if global_ctrl is None:
        return jsonify({'error': 'Not initialized'}), 400

    # 1. Step simulation
    step_data = global_ctrl.run_step()
    done = step_data['done']
    metrics = step_data['metrics']

    # 2. Extract Agent Counts
    agents = global_ctrl.env.agent_positions.keys()
    scouts = sum(1 for a in agents if 'scout' in a)
    firefighters = sum(1 for a in agents if 'firefighter' in a)
    medics = sum(1 for a in agents if 'medic' in a)

    # 3. Extract Real-Time Logs
    logs = []
    events = step_data.get('events', [])
    for ev in events:
        evt_type = ev.get('type', 'event').upper()
        if evt_type == 'RESCUE':
            logs.append(f"Unit {ev.get('agent')} rescued {ev.get('count')} victim(s).")
        elif evt_type == 'FIRE_OUT':
            logs.append(f"Fire extinguished at {ev.get('pos')}.")
        elif evt_type == 'AFTERSHOCK':
            logs.append(f"WARNING: Aftershock detected! New collapses: {ev.get('new_collapses')}")
        elif evt_type == 'BLACK_SWAN':
            logs.append(f"CRITICAL: Anomalous black swan event detected.")

    # 4. Render Dark-Themed Map (Ground Truth)
    with plt.style.context('dark_background'):
        fig = global_ctrl.env.render(show=False)
        fig.patch.set_facecolor('#000000') 
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # 5. Render Mental Map (Commander's Belief) if available
    mental_map_img_base64 = None
    if global_ctrl.commander and global_ctrl.commander.mental_map:
        with plt.style.context('dark_background'):
            fig = global_ctrl.env.render_mental_map(
                global_ctrl.commander.mental_map,
                show=False
            )
            fig.patch.set_facecolor('#000000')
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=100, facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            mental_map_img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # 6. Auto-dump final evaluation metrics when simulation ends
    if done:
        global_ctrl.get_evaluation_metrics()

    return jsonify({
        'done': done,
        'metrics': metrics,
        'image': img_base64,
        'mental_map': mental_map_img_base64,
        'agents': {'scouts': scouts, 'firefighters': firefighters, 'medics': medics},
        'logs': logs
    })


@app.route('/metrics', methods=['GET'])
def metrics():
    global global_ctrl
    if global_ctrl is None:
        return jsonify({'error': 'Not initialized'}), 400
    result = global_ctrl.get_evaluation_metrics()
    return jsonify(result)


if __name__ == '__main__':
    print("="*60)
    print("Starting SYS-COM Interface")
    print("Go to http://127.0.0.1:5003 in your web browser.")
    print("="*60)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5003)
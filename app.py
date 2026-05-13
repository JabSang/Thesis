import os
import traceback
import copy
import gzip
import json
from flask import Flask, render_template, request, jsonify, make_response  # type: ignore
from ega_dispatch import RoadNetwork, FleetManager, SimulationEngine  # type: ignore

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global instances – loaded ONCE at startup for speed.
# base_graph is an immutable deep-copy used to reset the network each run so
# that custom road-blockades never persist or compound between requests.
# ---------------------------------------------------------------------------
print("Initialising road network (this may take a moment)...")
network = RoadNetwork()
network.load_network()
base_graph = copy.deepcopy(network.graph)   # pristine copy ← never mutate this

fleet_manager = FleetManager(network)
engine = SimulationEngine(network, fleet_manager)
print("Server ready.")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    try:
        data = request.json or {}

        mode = data.get("mode", "free_play")

        if mode == 'case_study':
             scenario_id = data.get("scenario_id", "A")
             incident_type = data.get("incident_type", "Fire Problem")
             try:
                 alarm_level = max(1, min(5, int(data.get("alarm_level", 1))))
             except (ValueError, TypeError):
                 alarm_level = 1
                 
             network.graph = copy.deepcopy(base_graph)
             network.reset_scenarios()
             html_output = engine.run_thesis_scenario(scenario_id, incident_type, alarm_level)
             
             resp_data = json.dumps({"status": "success", "html": html_output})
             compressed_data = gzip.compress(resp_data.encode('utf8'))
             response = make_response(compressed_data)
             response.headers['Content-Length'] = len(compressed_data)
             response.headers['Content-Encoding'] = 'gzip'
             response.headers['Content-Type'] = 'application/json'
             return response

        # --- Parse inputs safely ---
        try:
            num_incidents = max(1, min(50, int(data.get("num_incidents", 5))))
        except (ValueError, TypeError):
            num_incidents = 5

        try:
            alarm_level = max(1, min(5, int(data.get("alarm_level", 1))))
        except (ValueError, TypeError):
            alarm_level = 1

        incident_types = data.get("incident_types", [])
        if not incident_types:
            incident_types = ["Fire Problem", "Medical Emergency", "Flood Evacuation / Rescue"]

        custom_hazards = data.get("custom_hazards", [])

        sim_time_str = str(data.get("sim_time", "17:30"))
        try:
            hours, minutes = map(int, sim_time_str.split(':'))
            emergency_call_time = hours + (minutes / 60.0)
        except Exception:
            emergency_call_time = 17.5

        print(
            f"\n[/run_simulation] incidents={num_incidents}, alarm={alarm_level}, "
            f"types={incident_types}, hazards={custom_hazards}, sim_time={sim_time_str} ({emergency_call_time})"
        )

        # --- SAFE STATE RESET: restore pristine graph before every run ---
        network.graph = copy.deepcopy(base_graph)
        network.reset_scenarios()           # re-initialise edge attributes

        # Apply user-selected road blockades (on the freshly-restored graph)
        if custom_hazards:
            network.apply_custom_hazards(custom_hazards)

        # --- Run the lightweight interactive simulation ---
        html_output = engine.run_custom_scenario(
            "Interactive Command Center",
            num_incidents,
            incident_types,
            alarm_level=alarm_level,
            sim_time=emergency_call_time
        )

        resp_data = json.dumps({"status": "success", "html": html_output})
        compressed_data = gzip.compress(resp_data.encode('utf8'))
        response = make_response(compressed_data)
        response.headers['Content-Length'] = len(compressed_data)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Type'] = 'application/json'
        return response

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)

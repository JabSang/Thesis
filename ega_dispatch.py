import os
import random
import time
import math
import numpy as np  # type: ignore
import networkx as nx  # type: ignore
import osmnx as ox  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import folium  # type: ignore
from folium import plugins  # type: ignore
import webbrowser
import json
from deap import base, creator, tools  # type: ignore
from typing import Dict, List, Any, cast
from itertools import islice
from branca.element import Template, MacroElement  # type: ignore

# Constants
BETA = 0.15
GAMMA = 10.0
ALPHA_COUNTERFLOW = 0.60
CONGESTION_THRESHOLD = 0.5

CDO_PEAK_HOURS = {
    "C.M. Recto Avenue": [(7.0, 9.0), (17.0, 20.0)],
    "Velez Street": [(7.0, 9.0), (16.5, 19.0)],
    "Osmena Street": [(11.0, 14.0), (15.0, 18.0)],
    "Yacapin Street": [(7.5, 8.5), (11.5, 12.5), (16.5, 17.5)]
}

def calculate_actual_execution_time(G, path, vehicle_type, emergency_call_time):
    if not path or len(path) < 2:
        return 0.0, 0.0
        
    vehicle_max_speed = 35.0 if vehicle_type == 'Ambulance' else 30.0
    
    total_time_secs = 0.0
    total_distance_km = 0.0
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        if not G.has_edge(u, v):
            total_time_secs += 999.0 * 60.0
            total_distance_km += 10.0
            continue
            
        edge_data = G[u][v]
        
        # Extract distance, revert artificial counterflow expansions if needed
        dist_km = edge_data.get('distance', 0.1)
        if edge_data.get('is_counterflow', False):
            if 'original_length' in edge_data:
                dist_km = edge_data['original_length'] / 1000.0
            else:
                dist_km = (edge_data.get('length', 100) / 1.05) / 1000.0
                
        total_distance_km += dist_km
        
        o_capacity = edge_data.get('capacity', 600)
        eff_capacity = max(1.0, o_capacity)
        ffs = edge_data.get('free_flow_speed', 30.0)
        
        # Revert artificial EGA penalites on FFS
        if edge_data.get('is_counterflow', False):
             ffs = ffs * 1.05
             
        # Target time is based purely on distance and BFP baseline capability (30 kph)
        eff_ffs = min(ffs, vehicle_max_speed)
        eff_capacity = o_capacity
        
        # Flooding Penalty
        if edge_data.get('flooded', False):
            eff_capacity *= 0.45
            
        # Peak Hour Penalty
        road_name = edge_data.get('name', '')
        if isinstance(road_name, list):
            road_name_str = ' '.join(str(n) for n in road_name).lower()
        else:
            road_name_str = str(road_name).lower()
            
        is_peak = False
        is_hard_penalty_road = False
        for street, windows in CDO_PEAK_HOURS.items():
            if street.lower() in road_name_str:
                for start, end in windows:
                    if start <= emergency_call_time <= end:
                        is_peak = True
                        if any(n in road_name_str for n in ["recto", "velez", "osmena"]):
                            is_hard_penalty_road = True
                        break
            if is_peak:
                break
                
        if is_peak:
            eff_capacity *= 0.70
            
        # Physical speed with dynamic BPR
        eff_capacity = max(1.0, eff_capacity)
        
        if is_peak:
            simulated_volume = eff_capacity * 1.5
        else:
            simulated_volume = eff_capacity * 0.4
            
        congestion_ratio = simulated_volume / eff_capacity
        dynamic_gamma = 10.0 if is_hard_penalty_road else GAMMA
        eff_speed = eff_ffs / (1 + BETA * (congestion_ratio ** dynamic_gamma))
            
        if eff_speed <= 0.1:
            eff_speed = 0.1
            
        edge_travel_time_secs = (dist_km / eff_speed) * 3600
        
        # Intersection Impedance with scaled queue density
        node_data = G.nodes[v]
        t_cycle = node_data.get('signal_cycle_time', 90)
        q_density = node_data.get('queue_density', 0.2)
        
        capacity_ratio = eff_capacity / max(1, o_capacity)
        eff_q_density = min(1.0, q_density / max(0.01, capacity_ratio))
        
        w_node = t_cycle * (1 + BETA * (eff_q_density ** GAMMA))
        
        total_time_secs += edge_travel_time_secs + w_node
        
    actual_exec_mins = total_time_secs / 60.0
    target_time_mins = (total_distance_km / 30.0) * 60.0
    
    return actual_exec_mins, target_time_mins

class RoadNetwork:
    def __init__(self, place_name="Cagayan de Oro City, Misamis Oriental, Philippines"):
        self.place_name = place_name
        self.graph: nx.DiGraph = nx.DiGraph()
        
    def load_network(self, osm_file="district2.osm"):
        print(f"Attempting to extract real road network for {self.place_name}...")
        try:
            if os.path.exists(osm_file):
                print(f"Loading local OSM file: {osm_file}...")
                try:
                    mG = ox.graph_from_xml(osm_file)
                except Exception as xml_err:
                    print(f"XML parse failed: {xml_err}. Attempting parse as Overpass JSON bounds...")
                    with open(osm_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Find min/max lat/lon from the nodes in the JSON
                    min_lat, max_lat = float('inf'), float('-inf')
                    min_lon, max_lon = float('inf'), float('-inf')
                    valid_nodes = False
                    for element in data.get('elements', []):
                        if element['type'] == 'node':
                            lat, lon = element['lat'], element['lon']
                            min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
                            min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
                            valid_nodes = True
                            
                    if valid_nodes:
                        print(f"Extracted bounds: Lat({min_lat} to {max_lat}), Lon({min_lon} to {max_lon})")
                        print("Building graph locally from JSON data to avoid network limits...")
                        import osmnx.graph  # type: ignore
                        mG = osmnx.graph._create_graph([data], bidirectional=False)
                        mG = ox.distance.add_edge_lengths(mG)
                        mG = ox.simplification.simplify_graph(mG)
                    else:
                        raise ValueError("No valid nodes found in JSON to extract bounding box.")
            else:
                print(f"File {osm_file} not found. Querying generic {self.place_name} map over network...")
                # Simplified using network type 'drive'
                mG = ox.graph_from_place(self.place_name, network_type="drive")
                
            # Add missing edge speeds and travel times
            mG = ox.add_edge_speeds(mG)
            mG = ox.add_edge_travel_times(mG)
            self.graph = nx.DiGraph(mG)
            self.graph.graph.update(mG.graph)
            print("Successfully extracted OSM network.")
        except Exception as e:
            print(f"OSM extraction failed: {e}. Falling back to synthetic graph...")
            self.generate_synthetic_network()
            
        self._initialize_attributes()

    def generate_synthetic_network(self):
        # Generate a connected directed grid graph
        G = nx.grid_2d_graph(20, 20)
        self.graph = nx.DiGraph(G)
        
        # Add x and y coordinates for plotting with OSMnx
        for node in self.graph.nodes():
            self.graph.nodes[node]['x'] = node[0] * 0.01 + 124.64  # approx longitude
            self.graph.nodes[node]['y'] = node[1] * 0.01 + 8.48    # approx latitude
            
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        self.graph.graph["crs"] = "EPSG:4326"
        
        # Add basic edge attributes
        for u, v, data in self.graph.edges(data=True):
            data['length'] = random.uniform(50, 500)  # meters
            data['speed_kph'] = random.uniform(30, 60)

    def _initialize_attributes(self):
        # Edge attributes based on empirical highway tags
        for u, v, data in self.graph.edges(data=True):
            if 'original_length' not in data:
                data['original_length'] = data.get('length', 100)
            data['length'] = data['original_length']
            
            distance_km = data['length'] / 1000.0
            data['distance'] = distance_km
            
            hw = data.get('highway', '')
            if isinstance(hw, list):
                hw = hw[0]
                
            if hw in ['trunk', 'primary']:
                free_flow_speed = 60
                capacity = 2000
            elif hw == 'secondary':
                free_flow_speed = 40
                capacity = 1200
            else:
                free_flow_speed = 30
                capacity = 600
                
            data['free_flow_speed'] = free_flow_speed
            data['capacity'] = capacity
            
            # Baseline volume is 20% of capacity
            volume = capacity * 0.20
            data['volume'] = volume
            data['congestion_level'] = volume / capacity
            data['current_speed'] = free_flow_speed
            data['flooded'] = False
            data['gridlocked'] = False
            
        # Node attributes
        for n, data in self.graph.nodes(data=True):
            data['signal_cycle_time'] = random.uniform(60, 120)  # seconds
            data['queue_density'] = 0.20

    def get_edge_travel_time(self, u, v, is_counterflow=False, mode='realistic', current_sim_time=0, call_time_hrs=None, vehicle_type='Fire Truck'):
        edge_data = self.graph[u][v]
        
        if edge_data.get('hard_roadblock', False):
            return 999999.0
            
        if mode == 'ideal':
            dist = edge_data['distance']
            speed = edge_data['free_flow_speed']
            base_time = (dist / speed) * 3600
            return base_time * 1.5 if vehicle_type == 'Ambulance' else base_time

        dist = edge_data['distance']
        speed = edge_data['current_speed']
        
        if call_time_hrs is not None:
            is_peak = False
            road_name = edge_data.get('name', '')
            road_name_str = ' '.join(str(n) for n in road_name).lower() if isinstance(road_name, list) else str(road_name).lower()
            
            for street, windows in CDO_PEAK_HOURS.items():
                if street.lower() in road_name_str:
                    for w_start, w_end in windows:
                        if w_start <= call_time_hrs <= w_end:
                            is_peak = True
                            break
                if is_peak: break
                
            if is_peak:
                eff_cap = max(1.0, edge_data.get('capacity', 600) * 0.70)
                sim_vol = eff_cap * 1.5
                ffs = edge_data.get('free_flow_speed', 30.0)
                if is_counterflow: ffs /= 4.0
                ratio = sim_vol / eff_cap
                dynamic_gamma = 10.0 if any(n in road_name_str for n in ["recto", "velez", "osmena"]) else GAMMA
                speed = ffs / (1 + BETA * (ratio ** dynamic_gamma))

        if edge_data.get('flooded', False) or (edge_data.get('gridlocked', False) and not is_counterflow):
            speed = 5.0 # Absolute 5.0 km/h floor for disasters instead of impassable Inf
            
        if speed <= 0:
            speed = 5.0
            
        if edge_data.get('is_black_wall', False) and not is_counterflow:
             return 9999.0 * 60.0

        base_time = (dist / speed) * 3600  # seconds
        if is_counterflow:
            base_time *= 1.05  # Reduce impedance to 1.05x penalty
        return base_time * 1.5 if vehicle_type == 'Ambulance' else base_time

    def intersection_impedance(self, node, mode='realistic', current_sim_time=0):
        if mode == 'ideal':
            return 0
            
        node_data = self.graph.nodes[node]
        T_cycle = node_data['signal_cycle_time']
        v_ratio = node_data['queue_density']
        W_node = T_cycle * (1 + BETA * (v_ratio ** GAMMA))
        return W_node

    def apply_scenario_gridlock(self):
        # Simulate 5:00 PM - 7:00 PM rush hour on major arteries
        for u, v, data in self.graph.edges(data=True):
            hw = data.get('highway', '')
            if isinstance(hw, list):
                hw = hw[0]
                
            if hw in ['trunk', 'primary', 'secondary']:
                # Volume between 95% and 110% of capacity
                multiplier = random.uniform(0.95, 1.10)
                data['volume'] = data['capacity'] * multiplier
                data['congestion_level'] = data['volume'] / data['capacity']
                
                # Speed drops based on extreme congestion
                drop_factor = 1 / (1 + BETA * (data['congestion_level'] ** GAMMA))
                data['current_speed'] = max(5.0, data['free_flow_speed'] * drop_factor)
                
                # Introduce true roadblocks (~5% of congested major arteries)
                if random.random() < 0.05:
                    data['gridlocked'] = True
                else:
                    data['gridlocked'] = False
                
                # Queue density matched to congestion, capped at 1.0 (standstill)
                self.graph.nodes[v]['queue_density'] = min(1.0, data['congestion_level'])

    def apply_scenario_flood(self):
        nodes = list(self.graph.nodes())
        flooded_centers = random.sample(nodes, min(5, len(nodes)))
        for center in flooded_centers:
            for n in nx.single_source_shortest_path_length(self.graph, center, cutoff=2).keys():
                for neighbor in self.graph.successors(n):
                    if random.random() < 0.8:
                        self.graph[n][neighbor]['flooded'] = True
                        
    def apply_custom_hazards(self, selected_roads):
        """Safely marks edges as gridlocked if their OSM name matches."""
        for u, v, data in self.graph.edges(data=True):
            road_name = data.get('name')
            if not road_name:
                continue
            
            # Safely handle lists or strings
            if isinstance(road_name, list):
                road_str = ' '.join(str(n) for n in road_name).lower()
            else:
                road_str = str(road_name).lower()
                
            for hazard in selected_roads:
                if hazard.lower() in road_str:
                    data['gridlocked'] = True
                        
    def reset_scenarios(self):
        self._initialize_attributes()

    def apply_ega_weights(self):
        """Pre-process graph for EGA.
        Adds reverse one-way edges with a 4.0x punitive travel time weight.
        The Baseline router will strictly ignore edges tagged 'is_counterflow'=True.
        """
        edges_to_add = []
        for u, v, attr in list(self.graph.edges(data=True)):
            
            road_name = attr.get('name', '')
            road_str = ' '.join(str(n) for n in road_name).lower() if isinstance(road_name, list) else str(road_name).lower()
            if any(r in road_str for r in ['c.m. recto', 'recto', 'velez', 'carmen bridge']):
                attr['is_black_wall'] = True
                attr['length'] = 9999.0
            
            if not attr.get('oneway', False):
                attr['is_counterflow'] = False
                continue
            
            attr['is_counterflow'] = False
            
            # If the reverse edge already exists don't touch it
            if self.graph.has_edge(v, u):
                continue
            
            # The Weighting Formula (The "Toll Road" Approach)
            reverse_attr = dict(attr)
            reverse_attr['is_counterflow'] = True
            
            # Multiply length/distance explicitly so traditional Dijkstra punishes it 1.05x
            if attr.get('is_black_wall', False):
                base_weight = attr.get('original_length', 100.0) # ignore 9999 overide
            else:
                base_weight = attr.get('length', 100.0)
            reverse_attr['length'] = float(base_weight) * 1.05
            
            # Divide speed by 1.05 explicitly so time-minimization punishes it 1.05x natively
            ffs = attr.get('free_flow_speed', 30.0)
            reverse_attr['free_flow_speed'] = ffs / 1.05
            reverse_attr['current_speed'] = attr.get('current_speed', ffs) / 1.05
            
            reverse_attr['congestion_level'] = attr.get('congestion_level', 0.0)
            reverse_attr['flooded'] = False
            reverse_attr['gridlocked'] = False
            edges_to_add.append((v, u, reverse_attr))
        self.graph.add_edges_from(edges_to_add)

class FleetManager:
    def __init__(self, network: RoadNetwork):
        self.network = network
        self.stations = {}
        self.vehicles: List[Dict[str, Any]] = []
        self.current_time = 0 # seconds from simulation start
        self._setup_stations()

    def _setup_stations(self):
        hubs = [
            # 10 BFP Fire Stations
            {'id': 'BFP_Central', 'name': 'BFP Central', 'type': 'Fire Truck', 'lat': 8.4795, 'lon': 124.6441, 'capacity': 2},
            {'id': 'BFP_Lapasan', 'name': 'Lapasan Substation', 'type': 'Fire Truck', 'lat': 8.4833, 'lon': 124.6567, 'capacity': 2},
            {'id': 'BFP_Cogon', 'name': 'Cogon Substation', 'type': 'Fire Truck', 'lat': 8.4782, 'lon': 124.6515, 'capacity': 2},
            {'id': 'BFP_Macabalan', 'name': 'Macabalan Substation', 'type': 'Fire Truck', 'lat': 8.4981, 'lon': 124.6620, 'capacity': 2},
            {'id': 'BFP_Camamanan', 'name': 'Camaman-an Substation', 'type': 'Fire Truck', 'lat': 8.4705, 'lon': 124.6580, 'capacity': 2},
            {'id': 'BFP_Macasandig', 'name': 'Macasandig Substation', 'type': 'Fire Truck', 'lat': 8.4621, 'lon': 124.6465, 'capacity': 2},
            {'id': 'BFP_Tablon', 'name': 'Tablon Substation', 'type': 'Fire Truck', 'lat': 8.4875, 'lon': 124.6922, 'capacity': 2},
            {'id': 'BFP_Agusan', 'name': 'Agusan Substation', 'type': 'Fire Truck', 'lat': 8.4892, 'lon': 124.7051, 'capacity': 2},
            {'id': 'BFP_Puerto', 'name': 'Puerto Substation', 'type': 'Fire Truck', 'lat': 8.4901, 'lon': 124.7210, 'capacity': 2},
            {'id': 'BFP_Bugo', 'name': 'Bugo Substation', 'type': 'Fire Truck', 'lat': 8.5020, 'lon': 124.7515, 'capacity': 2},
            # 4 Medical Stations
            {'id': 'Med_NMMC', 'name': 'NMMC', 'type': 'Ambulance', 'lat': 8.4815, 'lon': 124.6468, 'capacity': 2},
            {'id': 'Med_JRBorja', 'name': 'J.R. Borja', 'type': 'Ambulance', 'lat': 8.4716, 'lon': 124.6548, 'capacity': 2},
            {'id': 'Med_CampEvan', 'name': 'Camp Evangelista', 'type': 'Ambulance', 'lat': 8.5028, 'lon': 124.6369, 'capacity': 2},
            {'id': 'Med_Puerto', 'name': 'Puerto', 'type': 'Ambulance', 'lat': 8.4901, 'lon': 124.7210, 'capacity': 2},
        ]
        
        for hub in hubs:
            try:
                station_node = ox.distance.nearest_nodes(self.network.graph, hub['lon'], hub['lat'], max_distance=2000)
            except Exception:
                station_node = ox.distance.nearest_nodes(self.network.graph, hub['lon'], hub['lat'])
            self.stations[hub['id']] = {'type': hub['type'], 'node': station_node, 'name': hub['name'], 'capacity': hub['capacity'], 'current_count': hub['capacity']}
            
            turnaround = 3600 if hub['type'] == 'Fire Truck' else 1200
            for j in range(int(hub['capacity'])):
                self.vehicles.append({
                    'id': f"V_{hub['id']}_{j}",
                    'station': hub['id'],
                    'type': hub['type'],
                    'state': 'Available',
                    'available_time': 0,
                    'turnaround_time': turnaround
                })

    def request_vehicles(self, incident_type, count, incident_node, mode='realistic', algo='Baseline'):
        vehicle_type = 'Ambulance' if incident_type == 'Medical Emergency' else 'Fire Truck'
        
        # In Ideal mode, infinite fleet availability (always return distinct mocked vehicles to allow tracking)
        if mode == 'ideal':
            # We return a list of dummy vehicle dicts to satisfy the dispatcher structure
            # without affecting real fleet state.
            vehicles = [{
                'id': f"Ideal_V_{i}",
                'station': random.choice(list(self.stations.keys())), # Random station for ideal? Or nearest?
                # Actually, ideal dispatcher usually picks nearest station. 
                # But FleetManager is supposed to return vehicles. 
                # Let's return best available from real fleet but ignore status.
                # Or better, just return the nearest N vehicles regardless of state.
                'type': vehicle_type,
                'state': 'Available',
                'available_time': 0,
                'turnaround_time': 0
            } for i in range(count)]
            return vehicles, 0, []

        # Realistic Mode: Cascade logic relies on checking stations sequentially, no early exit.
            
        def distance_to_incident(veh):
            st_node = self.stations[veh['station']]['node']
            try:
                # Use ideal distance for fast sorting/lookup
                return nx.shortest_path_length(self.network.graph, source=st_node, target=incident_node, weight='length')
            except nx.NetworkXNoPath:
                return float('inf')
            except Exception:
                return float('inf')
                
        def get_station_dist(sid):
            try:
                return nx.shortest_path_length(self.network.graph, source=self.stations[sid]['node'], target=incident_node, weight='length')
            except (nx.NetworkXNoPath, Exception):
                return float('inf')

        valid_stations = [sid for sid, sdata in self.stations.items() if sdata['type'] == vehicle_type]
        valid_stations.sort(key=get_station_dist)
        primary_station_id = valid_stations[0] if valid_stations else None

        print(f"\n[Time: {self.current_time}s] Incident Occurs at Node {incident_node} (Type: {vehicle_type}, Severity: {count})")
        time.sleep(0.5)
        print(f"EGA computes optimal dispatch plan...")
        time.sleep(0.5)

        assigned = []
        checked_stations = []

        for sid in valid_stations:
            if len(assigned) >= count:
                break
                
            dist = get_station_dist(sid)
            if dist == float('inf'): continue
            
            station_name = self.stations[sid]['name']
            checked_stations.append(sid)
            print(f"Check Station: {station_name}...")
            time.sleep(0.5)
            
            st_vehicles = [
                v for v in self.vehicles 
                if v['station'] == sid 
                and v['state'] == 'Available'
                and v['available_time'] <= self.current_time
            ]
            
            needed = count - len(assigned)
            if len(st_vehicles) >= needed:
                assigned.extend(list(st_vehicles)[:needed]) # type: ignore
                self.stations[sid]['current_count'] -= needed
                print(f" -> Success: Dispatched {needed} vehicle(s) from {station_name}.")
                time.sleep(0.5)
                break
            elif len(st_vehicles) > 0:
                assigned.extend(st_vehicles)
                self.stations[sid]['current_count'] -= len(st_vehicles)
                print(f" -> Partial: Dispatched {len(st_vehicles)} vehicle(s). Station depleted.")
                time.sleep(0.5)
            else:
                print(f" -> Depleted: No available units at {station_name}.")
                time.sleep(0.5)
        
        if len(assigned) < count:
            if algo == 'Baseline':
                print("CRITICAL: Baseline failed to cover! Fleet completely empty.")
                time.sleep(0.5)
                if not assigned:
                    return None, 0, checked_stations
            elif algo == 'EGA':
                print("EGA Redeployment triggered: All stations empty. Pushing to pending incidents list...")
                time.sleep(0.5)
                deployed_vehicles = [
                    v for v in self.vehicles 
                    if v['type'] == vehicle_type 
                    and (v['state'] != 'Available' or v['available_time'] > self.current_time)
                ]
                if not deployed_vehicles:
                    return None, 0, checked_stations
                    
                deployed_vehicles.sort(key=lambda x: x['available_time'])
                
                needed = count - len(assigned)
                redeployed = []
                max_ff_time = self.current_time
                
                for i in range(min(needed, len(deployed_vehicles))):
                    returning_vehicle = deployed_vehicles[i]
                    max_ff_time = max(max_ff_time, returning_vehicle['available_time'])
                    redeployed.append(returning_vehicle)
                    
                self.current_time = max(self.current_time, max_ff_time)
                print(f"ANTI-CHEAT: Clock fast-forwarded to {self.current_time}s for EGA redeployment.")
                time.sleep(0.5)
                
                for v in redeployed:
                    assigned.append(v)
                    v['state'] = 'Available'
                    self.stations[v['station']]['current_count'] -= 1
                    if v['station'] not in checked_stations:
                        checked_stations.append(v['station'])
                    print(f" -> Redeployed unit returning at {self.current_time}s from {self.stations[v['station']]['name']}.")
                    time.sleep(0.5)
                    
            if not assigned:
                return None, 0, checked_stations

        cross_count = sum(1 for v in assigned if v['station'] != primary_station_id) if primary_station_id else 0
        if cross_count > 0:
            print(f"Cross-Station Reassignment Triggered for Incident at Node {incident_node}!")
            time.sleep(0.5)

        for v in assigned:
            v['state'] = 'Dispatched'
        print("Dispatch sequence complete.")
        time.sleep(0.5)
        return assigned, cross_count, checked_stations

    def get_fleet_utilization(self):
        if not self.vehicles:
            return 0.0
        active = sum(1 for v in self.vehicles if v['state'] != 'Available')
        return active / len(self.vehicles)

    def release_vehicle(self, vehicle_id):
        for v in self.vehicles:
            if v['id'] == vehicle_id:
                v['state'] = 'Available'

    def reset(self):
        """Resets the fleet state for a new simulation run."""
        self.vehicles = []
        self.stations = {}
        self.current_time = 0
        self._setup_stations()


class EnhancedGA:
    def __init__(self, network: RoadNetwork, start_node, target_node, mode='realistic', start_time=0, call_time_hrs=None, vehicle_type='Fire Truck'):
        self.network = network
        self.start_node = start_node
        self.target_node = target_node
        self.graph = network.graph
        self.mode = mode
        self.start_time = start_time
        self.call_time_hrs = call_time_hrs
        self.vehicle_type = vehicle_type

        # DEAP setup
        try:
            del creator.FitnessMin
        except AttributeError:
            pass
        try:
            del creator.Individual
        except AttributeError:
            pass

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_individual(self):
        try:
            # Generate top 5-10 shortest simple paths for diversity using distance weight
            k = random.randint(5, 10)
            paths_gen = nx.shortest_simple_paths(self.graph, source=self.start_node, target=self.target_node, weight='distance')
            paths = list(islice(paths_gen, k))
            if paths:
                path = random.choice(paths)
            else:
                path = nx.shortest_path(self.graph, source=self.start_node, target=self.target_node, weight='distance')
        except (nx.NetworkXNoPath, Exception):
            try:
                path = nx.shortest_path(self.graph, source=self.start_node, target=self.target_node)
            except (nx.NetworkXNoPath, Exception):
                path = [self.start_node, self.target_node]
        return creator.Individual(path)

    def _evaluate(self, individual):
        if len(individual) < 2 or individual[0] != self.start_node or individual[-1] != self.target_node:
            return float('inf'),

        total_time = 0
        current_sim_time = float(getattr(self, 'start_time', 0))

        for i in range(len(individual) - 1):
            u, v = individual[i], individual[i+1]
            if not self.graph.has_edge(u, v): # type: ignore
                return float('inf'),

            # The Fitness Function (Time + Distance Balancing)
            edge_dist = self.graph[u][v].get('distance', 1.0)
            is_cf = self.graph[u][v].get('is_counterflow_edge', False)
            
            t_link = self.network.get_edge_travel_time(
                u, v, is_counterflow=is_cf, mode=self.mode, 
                current_sim_time=current_sim_time, call_time_hrs=self.call_time_hrs, vehicle_type=self.vehicle_type) # type: ignore

            if t_link == float('inf'):
                return float('inf'),

            current_sim_time += t_link # type: ignore
            w_node = self.network.intersection_impedance(v, mode=self.mode, current_sim_time=current_sim_time) # type: ignore
            
            # W_dist = 0.4 (increased), W_time = 0.6 (slightly decreased)
            cost_link = (edge_dist * 100 * 0.4) + (t_link * 0.6) + w_node
            total_time += cost_link
            current_sim_time += w_node

        cost = float(total_time)
        return cost,

    def _crossover(self, ind1, ind2):
        common_nodes = list(set(ind1[1:-1]) & set(ind2[1:-1]))
        if common_nodes:
            cross_node = random.choice(common_nodes)
            idx1 = ind1.index(cross_node)
            idx2 = ind2.index(cross_node)
            new_ind1 = ind1[:idx1] + ind2[idx2:]
            new_ind2 = ind2[:idx2] + ind1[idx1:]
            return creator.Individual(new_ind1), creator.Individual(new_ind2)
        return ind1, ind2

    def _mutate(self, individual):
        if len(individual) > 3:
            mutate_idx1 = random.randint(1, len(individual)-3)
            mutate_idx2 = random.randint(mutate_idx1+1, len(individual)-2)
            u = individual[mutate_idx1]
            v = individual[mutate_idx2]
            try:
                sub_path = nx.shortest_path(self.graph, u, v)
                new_ind = individual[:mutate_idx1] + sub_path + individual[mutate_idx2+1:]
                return creator.Individual(new_ind),
            except (nx.NetworkXNoPath, Exception):
                pass
        return individual,

    def run(self, ngen=20, pop_size=10, mutpb=0.2, cxpb=0.5, elite_size=2):
        try:
            # Apply Counterflow edges: Allow GA to traverse gridlocked edges sequentially backward
            edges_to_add = []
            for u, v, data in self.graph.edges(data=True):
                if data.get('gridlocked', False) and not self.graph.has_edge(v, u):
                    rev_data = data.copy()
                    rev_data['is_counterflow_edge'] = True
                    edges_to_add.append((v, u, rev_data))
            for u, v, d in edges_to_add:
                self.graph.add_edge(u, v, **d)
                
            nx.shortest_path(self.graph, self.start_node, self.target_node)
        except nx.NetworkXNoPath:
            return [self.start_node, self.target_node], 999.0 * 60.0
            
        pop = self.toolbox.population(n=pop_size)
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        actual_elite_size = min(elite_size, len(pop))
        for g in range(ngen):
            elites = tools.selBest(pop, actual_elite_size)
            elites = list(map(self.toolbox.clone, elites))

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Cast to explicit list to silence linter slice error
            _offspring_list: List[Any] = list(offspring)
            for child1, child2 in zip(_offspring_list[::2], _offspring_list[1::2]):  # type: ignore
                if random.random() < cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind)

            pop[:] = offspring
            worst_indices = [pop.index(ind) for ind in tools.selWorst(pop, actual_elite_size)]
            for i, elite in zip(worst_indices, elites):
                pop[i] = elite

        best_ind = tools.selBest(pop, 1)[0]
        return best_ind, best_ind.fitness.values[0]


class BaselineDispatcher:
    def __init__(self, network: RoadNetwork, start_node, target_node, mode='realistic', start_time=0, call_time_hrs=None, vehicle_type='Fire Truck'):
        self.network = network
        self.start_node = start_node
        self.target_node = target_node
        self.mode = mode
        self.start_time = start_time
        self.call_time_hrs = call_time_hrs
        self.vehicle_type = vehicle_type
        
    def run(self):
        try:
            # Baseline Constraint:
            # Ensure the Baseline pathfinding algorithm strictly IGNORES edges where is_counterflow = True.
            def filter_edge(n1, n2):
                return not self.network.graph[n1][n2].get('is_counterflow', False)
                
            baseline_graph = nx.subgraph_view(self.network.graph, filter_edge=filter_edge)
            
            path = nx.shortest_path(baseline_graph, source=self.start_node,
                                    target=self.target_node, weight='length')
            total_time = 0.0
            current_sim_time = float(getattr(self, 'start_time', 0))
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                t_link = float(self.network.get_edge_travel_time(u, v, is_counterflow=False, mode=self.mode, current_sim_time=current_sim_time, call_time_hrs=self.call_time_hrs, vehicle_type=self.vehicle_type))
                current_sim_time += t_link
                w_node = float(self.network.intersection_impedance(v, mode=self.mode, current_sim_time=current_sim_time))
                total_time += t_link + w_node
                current_sim_time += w_node
            return path, total_time
        except nx.NetworkXNoPath:
            return [self.start_node, self.target_node], 999.0 * 60.0


class SimulationEngine:
    def __init__(self, network: RoadNetwork, fleet_manager: FleetManager):
        self.network = network
        self.fleet_manager = fleet_manager
        self.scenario_stats = [] # Store summary data for the final table
        self.all_real_b_times = [] # For final overall plot
        self.all_real_e_times = [] # For final overall plot

    def generate_incidents(self, num_incidents, incident_types=None, alarm_level=1, sim_time=None):
        random.seed(None) # Re-seed generator with system clock
        
        if not incident_types:
            incident_types = ["Fire Problem", "Medical Emergency", "Flood Evacuation / Rescue"]
            
        largest_scc = max(nx.strongly_connected_components(self.network.graph), key=len)
        scc_nodes = list(largest_scc) # type: ignore
        
        # Ensure we only pick SCC nodes that possess true geographical bounds
        valid_nodes = [
            n for n in scc_nodes 
            if 'y' in self.network.graph.nodes[n] and 'x' in self.network.graph.nodes[n]
        ]
        
        # Shuffle completely for spatial jitter between runs
        random.shuffle(valid_nodes)
        
        incidents = []
        current_time = 0
        for _ in range(num_incidents):
            inc_type = random.choice(incident_types)
            if inc_type == "Fire Problem":
                # CDO BFP standard: 1st=4, 2nd=8, 3rd=12, 4th=16, 5th=20 engines
                alarm_engine_map = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}
                severity = alarm_engine_map.get(alarm_level, 4)
            else:
                severity = random.randint(1, 2)
                
            call_hr = sim_time if sim_time is not None else random.uniform(0.0, 24.0)
            incidents.append({
                'node': random.choice(valid_nodes),
                'type': inc_type,
                'severity': severity,
                'time': current_time,
                'call_time_hrs': call_hr
            })
            current_time += random.randint(1800, 7200)
            
        return incidents

    def visualize_route(self, path, m, vehicle_id="V1", color="red", checked_stations=None):
        if not path:
            return
            
        # Get coordinates for the path
        route_coords = []
        for node in path:
            # OSMnx uses y for lat, x for lng
            lat = self.network.graph.nodes[node]['y']
            lng = self.network.graph.nodes[node]['x']
            route_coords.append([lat, lng])
            
        if not route_coords:
            return
            
        # Draw the underlying path lightly to guide the eye
        folium.PolyLine(route_coords, color=color, weight=3, opacity=0.3).add_to(m)
        
        # Add start and end markers
        folium.Marker(route_coords[0], popup='Deployment Station', icon=folium.Icon(color='green', icon='play')).add_to(m)
        
        incident_idx = len(route_coords) // 2
        incident_coord = route_coords[incident_idx]
        folium.Marker(incident_coord, popup='Incident Location', icon=folium.Icon(color='red', icon='fire')).add_to(m)

        js_coords = json.dumps(route_coords)
        vid_clean = vehicle_id.replace('-', '_')
        
        pan_js_array = []
        if checked_stations:
            for sid in checked_stations:
                st_node = self.fleet_manager.stations[sid]['node']
                st_lat = self.network.graph.nodes[st_node]['y']
                st_lng = self.network.graph.nodes[st_node]['x']
                pan_js_array.append([st_lat, st_lng])
        
        # NOTE: JS Animation removed from individual visualize_route calls to be handled globally
        # to ensure chronological synchronization with the dashboard metrics.

    def run_scenario(self, scenario_name, num_incidents=10):
        print(f"\n--- Running Scenario: {scenario_name} ---")
        incidents = self.generate_incidents(num_incidents)
        
        current_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {
            'ideal': {
                'Baseline': {'response_times': [], 'vehicles_deployed': 0, 'responding_stations': set(), 'reassignments_triggered': 0, 'uncovered': 0},
                'EGA': {'response_times': [], 'vehicles_deployed': 0, 'responding_stations': set(), 'reassignments_triggered': 0, 'uncovered': 0}
            },
            'realistic': {
                'Baseline': {'response_times': [], 'vehicles_deployed': 0, 'responding_stations': set(), 'reassignments_triggered': 0, 'uncovered': 0},
                'EGA': {'response_times': [], 'vehicles_deployed': 0, 'responding_stations': set(), 'reassignments_triggered': 0, 'uncovered': 0}
            }
        }
        
        # Collect chronological dispatch events for the maps
        # Dictionary keyed by incident idx, later converted back to array
        dispatch_events_dict: Dict[str, Dict[int, Dict[str, Any]]] = {'ideal': {}, 'realistic': {}}
        
        # Timelines are now natively assigned in generate_incidents

        # Initialize scenario maps (One for Ideal, One for Realistic)
        maps: Dict[str, Any] = {
            'ideal': folium.Map(location=[8.48, 124.65], zoom_start=12, tiles='cartodbpositron'),
            'realistic': folium.Map(location=[8.48, 124.65], zoom_start=12, tiles='cartodbpositron')
        }
        
        # Add Hazard & Station Markers
        for mode_key in ['ideal', 'realistic']:
            # Station Markers
            for sid, sdata in self.fleet_manager.stations.items():
                s_y = self.network.graph.nodes[sdata['node']]['y']
                s_x = self.network.graph.nodes[sdata['node']]['x']
                if sdata['type'] == 'Fire Truck':
                    folium.Marker([s_y, s_x], popup=sdata['name'], icon=folium.Icon(color='red', icon='fire', prefix='fa')).add_to(maps[mode_key])  # type: ignore[arg-type]
                else:
                    folium.Marker([s_y, s_x], popup=sdata['name'], icon=folium.Icon(color='blue', icon='plus-square', prefix='fa')).add_to(maps[mode_key])  # type: ignore[arg-type]
            
            # Hazard Markers
            for u, v, data in self.network.graph.edges(data=True):
                if data.get('flooded', False):
                    n_y, n_x = self.network.graph.nodes[u]['y'], self.network.graph.nodes[u]['x']
                    folium.Marker([n_y, n_x], icon=folium.Icon(color='cadetblue', icon='water', prefix='fa')).add_to(maps[mode_key])  # type: ignore[arg-type]
                if data.get('gridlocked', False):
                    n_y, n_x = self.network.graph.nodes[u]['y'], self.network.graph.nodes[u]['x']
                    folium.Marker([n_y, n_x], icon=folium.Icon(color='darkred', icon='ban', prefix='fa')).add_to(maps[mode_key])  # type: ignore[arg-type]
            
            # Incident Markers
            for i, inc in enumerate(incidents):
                inc_y, inc_x = self.network.graph.nodes[inc['node']]['y'], self.network.graph.nodes[inc['node']]['x']
                if inc['type'] == 'Fire Problem':
                    icon = folium.Icon(color='lightred', icon='fire', prefix='fa')
                elif inc['type'] == 'Medical Emergency':
                    icon = folium.Icon(color='green', icon='medkit', prefix='fa')
                else:
                    icon = folium.Icon(color='lightblue', icon='life-ring', prefix='fa')
                folium.Marker([inc_y, inc_x], tooltip=f"Incident #{i}: {inc['type']}", icon=icon).add_to(maps[mode_key])  # type: ignore[arg-type]
        
        # --- EXECUTE 4 DISTINCT SIMULATION RUNS ---
        for mode in ['ideal', 'realistic']:
            for algo in ['Baseline', 'EGA']:
                # Reset Fleet State for independent run
                self.fleet_manager.reset()
                
                print(f"Executing Run: Mode={mode}, Algo={algo}...")
                
                for idx, incident in enumerate(incidents):
                    self.fleet_manager.current_time = incident['time']
                    
                    req_result = self.fleet_manager.request_vehicles(incident['type'], incident['severity'], incident['node'], mode=mode, algo=algo)
                    vehicles = req_result[0] if req_result else None
                    cross_count = req_result[1] if req_result else 0
                    checked_stations = req_result[2] if req_result else []
                    
                    if not vehicles:
                        current_metrics[mode][algo]['uncovered'] += 1  # type: ignore
                        continue
                        
                    current_metrics[mode][algo]['vehicles_deployed'] += len(vehicles)  # type: ignore
                    current_metrics[mode][algo]['reassignments_triggered'] += cross_count  # type: ignore
                    for v in vehicles:
                        current_metrics[mode][algo]['responding_stations'].add(v['station'])  # type: ignore
                        
                    station_node = self.fleet_manager.stations[vehicles[0]['station']]['node']
                    
                    path_out, time_out = None, float('inf')
                    path_ret, time_ret = None, float('inf')
                    
                    if algo == 'Baseline':
                        dispatcher_out = BaselineDispatcher(self.network, station_node, incident['node'], mode=mode, start_time=self.fleet_manager.current_time)
                        path_out, time_out = dispatcher_out.run()
                        
                        dispatcher_ret = BaselineDispatcher(self.network, incident['node'], station_node, mode=mode, start_time=self.fleet_manager.current_time + (time_out if time_out < float('inf') else 0))
                        path_ret, time_ret = dispatcher_ret.run()
                    else:
                        # EGA
                        ngen, pop_size = (10, 10) if mode == 'ideal' else (50, 50)
                        if vehicles and len(vehicles) > 4:
                            pop_size = 30
                        
                        dispatcher_out = EnhancedGA(self.network, station_node, incident['node'], mode=mode, start_time=self.fleet_manager.current_time)
                        path_out, time_out = dispatcher_out.run(ngen=ngen, pop_size=pop_size)
                        
                        dispatcher_ret = EnhancedGA(self.network, incident['node'], station_node, mode=mode, start_time=self.fleet_manager.current_time + (time_out if time_out < float('inf') else 0))
                        path_ret, time_ret = dispatcher_ret.run(ngen=ngen, pop_size=pop_size)
                        
                        # Visualization (Visualize EGA for each mode on its respective map)
                        if algo == 'EGA' and path_out and path_ret:
                            full_path = path_out + path_ret[1:]
                            self.visualize_route(full_path, maps[mode], vehicle_id=f"v_{idx}", color="blue", checked_stations=checked_stations)  # type: ignore[arg-type]

                    # Build UI payload
                    if path_out and path_ret:
                        incident_lat = self.network.graph.nodes[incident['node']]['y']
                        incident_lng = self.network.graph.nodes[incident['node']]['x']
                        
                        route_coords = []
                        full_path = path_out + path_ret[1:]
                        if full_path:
                            route_coords.append([self.network.graph.nodes[full_path[0]]['y'], self.network.graph.nodes[full_path[0]]['x']])
                            for idx_path in range(len(full_path)-1):
                                u_node = full_path[idx_path]
                                v_node = full_path[idx_path+1]
                                if self.network.graph.has_edge(u_node, v_node) and 'geometry' in self.network.graph[u_node][v_node]:
                                    coords = list(self.network.graph[u_node][v_node]['geometry'].coords)
                                    for lon, lat in islice(coords, 1, None):
                                        route_coords.append([lat, lon])
                                else:
                                    route_coords.append([self.network.graph.nodes[v_node]['y'], self.network.graph.nodes[v_node]['x']])
                            
                        pan_js_array = []
                        for sid in checked_stations:
                            st_node = self.fleet_manager.stations[sid]['node']
                            pan_js_array.append([self.network.graph.nodes[st_node]['y'], self.network.graph.nodes[st_node]['x']])
                            
                        if idx not in dispatch_events_dict[mode]:  # type: ignore[operator]
                            dispatch_events_dict[mode][idx] = {  # type: ignore[index]
                                'incident_id': idx,
                                'type': incident['type'],
                                'incident_coord': [incident_lat, incident_lng],
                                'Baseline': None,
                                'EGA': None,
                                'time_of_call': float(incident.get('call_time_hrs', 0.0))
                            }
                            
                        ref_time = 0.0
                        target_time = 0.0
                        if full_path:
                            ref_time, target_time = calculate_actual_execution_time(self.network.graph, full_path, incident['type'], incident.get('call_time_hrs', 0.0))
                            
                        # Assign directly to requested properties for UI
                        if algo == 'Baseline':
                            dispatch_events_dict[mode][idx]['base_travel_time'] = float(ref_time) # type: ignore
                            dispatch_events_dict[mode][idx]['target_time_mins'] = float(target_time) # type: ignore
                        else:
                            dispatch_events_dict[mode][idx]['ega_travel_time'] = float(ref_time) # type: ignore

                        dispatch_events_dict[mode][idx][algo] = { # type: ignore
                            'route_coords': route_coords,
                            'panned_stations': pan_js_array,
                            'travel_time_mins': (time_out / 60) if time_out < float('inf') else 0,
                            'actual_travel_time': float(ref_time),
                            'target_time_mins': float(target_time),
                            'vehicles_deployed': len(vehicles) if vehicles else 0,
                            'reassignments': cross_count,
                            'station_id': vehicles[0]['station'] if vehicles else None,
                            'station_name': self.fleet_manager.stations[vehicles[0]['station']]['name'] if vehicles else "None"
                        }
                        
                    if time_out < float('inf'):
                        current_metrics[mode][algo]['response_times'].append(time_out)  # type: ignore
                            
                    if vehicles:
                        mission_duration = 0
                        if time_out < float('inf') and time_ret < float('inf'):
                            mission_duration = time_out + time_ret
                        else:
                            mission_duration = 3600 
                            
                        for v in vehicles:
                            v['available_time'] = self.fleet_manager.current_time + mission_duration + v['turnaround_time']  # type: ignore
                            v['state'] = 'Available'  # type: ignore

        # Process Metrics and Generate Dashboards for BOTH maps
        for mode in ['ideal', 'realistic']:
            b_times = current_metrics[mode]['Baseline']['response_times']
            e_times = current_metrics[mode]['EGA']['response_times']
            
            b_avg = (np.mean(b_times) / 60) if b_times else 0
            e_avg = (np.mean(e_times) / 60) if e_times else 0
            
            b_deps = current_metrics[mode]['Baseline']['vehicles_deployed']
            e_deps = current_metrics[mode]['EGA']['vehicles_deployed']
            
            b_stats = len(current_metrics[mode]['Baseline']['responding_stations'])
            e_stats = len(current_metrics[mode]['EGA']['responding_stations'])
            
            b_unc = current_metrics[mode]['Baseline']['uncovered']
            e_unc = current_metrics[mode]['EGA']['uncovered']
            b_reassign = current_metrics[mode]['Baseline']['reassignments_triggered']
            e_reassign = current_metrics[mode]['EGA']['reassignments_triggered']
            
            dispatch_events_js_list = [v for k, v in sorted(dispatch_events_dict[mode].items())]

            dashboard_html = f'''
            <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 300px; height: auto; 
                 background-color: rgba(255, 255, 255, 0.95); z-index:9999; font-family: Arial, sans-serif; font-size:14px;
                 border: 2px solid #555; border-radius: 10px; padding: 15px;
                 box-shadow: 3px 3px 15px rgba(0,0,0,0.5);">
                 <h4 style="margin-top:0; margin-bottom:10px; color:#333;">{scenario_name} ({mode.capitalize()})</h4>
                 <div style="font-size:11px; margin-bottom:5px;">
                    <span style="color:red; font-weight:bold;">\u25A0</span> Baseline Detour &nbsp;
                    <span style="color:blue; font-weight:bold;">\u25A0</span> EGA Normal<br>
                    <span style="color:#e6e600; text-shadow:1px 1px 0 #000; font-weight:bold;">\u25A0</span> EGA Counterflow &nbsp;
                    <span style="color:black; font-weight:bold;">\u25AC</span> Impassable Roadblock
                 </div>
                 <div id="live_status" style="margin-bottom:10px; font-weight:bold; color:#0056b3; font-size:12px; min-height:40px;">Waiting for incident...</div>
                 <hr style="border-color:#ccc; margin-top:0;">
                 <b style="color:#d9534f;">Baseline System</b><br>
                 Outbound Avg: <span id="baseline_avg">0.00</span> mins<br>
                 Vehicles Deployed: <span id="baseline_deployed">0</span><br>
                 Responding Stations: <span id="baseline_stats">0</span><br>
                 Reassignments Triggered: <span id="baseline_reassign">0</span><br>
                 Uncovered Incidents: <span id="baseline_unc">0</span><br>
                 <hr style="border-color:#ccc;">
                 <b style="color:#0275d8;">Enhanced GA System</b><br>
                 Outbound Avg: <span id="ega_avg">0.00</span> mins<br>
                 Vehicles Deployed: <span id="ega_deployed">0</span><br>
                 Responding Stations: <span id="ega_stats">0</span><br>
                 Reassignments Triggered: <span id="ega_reassign">0</span><br>
                 Uncovered Incidents: <span id="ega_unc">0</span><br>
            </div>
            '''
            maps[mode].get_root().html.add_child(folium.Element(dashboard_html))
            
            self.inject_global_animation_script(maps[mode], dispatch_events_js_list)

            filename = f"{scenario_name.replace(' ', '_').lower()}_{mode}.html"
            maps[mode].save(filename)
            print(f"[{scenario_name}] {mode.capitalize()} simulation map saved to {filename}")
            if mode == 'realistic': # Auto-open realistic map for convenience
                webbrowser.open('file://' + os.path.realpath(filename))
        
        # Store Realistic stats for the final summary table
        real_b_times = current_metrics['realistic']['Baseline']['response_times']
        real_e_times = current_metrics['realistic']['EGA']['response_times']
        r_b_avg = (np.mean(real_b_times) / 60) if real_b_times else 0
        r_e_avg = (np.mean(real_e_times) / 60) if real_e_times else 0
        
        improvement = 0.0
        if r_b_avg > 0:
            improvement = ((r_b_avg - r_e_avg) / r_b_avg) * 100
            
        self.scenario_stats.append({
            'Scenario': scenario_name,
            'Baseline Avg (min)': r_b_avg,
            'EGA Avg (min)': r_e_avg,
            'Improvement (%)': improvement
        })
        
        # Accumulate for global plot
        self.all_real_b_times.extend([float(x) for x in real_b_times])
        self.all_real_e_times.extend([float(x) for x in real_e_times])

    def run_thesis_scenario(self, scenario_id, incident_type='Fire Problem', alarm_level=1):
        print(f"\n--- Thesis Case Study: Scenario {scenario_id} ---")
        

        # Determine severity based on BFP standard or just copy over
        if incident_type == 'Fire Problem':
            alarm_engine_map = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}
            severity = alarm_engine_map.get(alarm_level, 4)
        else:
            severity = 1
        
        if scenario_id == 'A':
            # 2:00 AM Clear Roads (Control)
            sim_time = 2.0
            target_lat, target_lon = 8.4715, 124.6360 # Brgy. Carmen
            flooded_nodes = []
            
        elif scenario_id == 'B':
            # 5:30 PM C.M. Recto Micro-Gridlock
            sim_time = 17.5
            flooded_nodes = []
            
            # Unified Incident Destination (Limketkai Area)
            inc_y, inc_x = 8.482185154087093, 124.65745561461243
            incident_node = ox.distance.nearest_nodes(self.network.graph, X=inc_x, Y=inc_y)

            # Dynamic Coordinate Assignment
            emergency_type = 'medical' if 'medical' in incident_type.lower() else 'fire'

            if emergency_type == 'medical':
                trap_y, trap_x = 8.483146420154966, 124.64930430141898
            else:  # fire
                trap_y, trap_x = 8.482580448863748, 124.65305825292657
                
            target_lat, target_lon = inc_y, inc_x
            blockades = [[trap_y, trap_x]]
            
            # Filter Origin Station by Emergency Type dynamically
            if 'medical' in incident_type.lower():
                facility_type = 'Ambulance'
            else:
                facility_type = 'Fire Truck'
                
            valid_stations = [s for s in self.fleet_manager.stations.values() if s['type'] == facility_type]
            if not valid_stations:
                valid_stations = list(self.fleet_manager.stations.values())
                
            source_node = min(
                valid_stations, 
                key=lambda s: nx.shortest_path_length(self.network.graph, s['node'], incident_node, weight='length')
            )['node']

            # Calculate ideal route BEFORE applying penalty
            try:
                ideal_route_nodes = nx.shortest_path(self.network.graph, source_node, incident_node, weight='travel_time')
                ideal_route = [[self.network.graph.nodes[n]['y'], self.network.graph.nodes[n]['x']] for n in ideal_route_nodes]
            except Exception:
                ideal_route_nodes = []
                ideal_route = []

            # 2. Snap Trap to Route
            if ideal_route_nodes:
                def get_dist(n):
                    return (self.network.graph.nodes[n].get('y', 0) - trap_y)**2 + (self.network.graph.nodes[n].get('x', 0) - trap_x)**2
                trap_node = min(ideal_route_nodes, key=get_dist)

                # 3. Railroad the Approach
                trap_idx = ideal_route_nodes.index(trap_node)
                route_part1_nodes = ideal_route_nodes[:trap_idx+1]

                # 4. The "Median Jump" & AoE Gridlock
                try:
                    next_node = ideal_route_nodes[trap_idx + 1]
                    
                    if self.network.graph.is_multigraph():
                        # Inject a hyper-fast counterflow edge directly over the median
                        if not self.network.graph.has_edge(trap_node, next_node):
                            self.network.graph.add_edge(trap_node, next_node, travel_time=0.001, is_counterflow=True)
                        else:
                            for k in self.network.graph[trap_node][next_node]:
                                self.network.graph[trap_node][next_node][k]['travel_time'] = 0.001
                                self.network.graph[trap_node][next_node][k]['is_counterflow'] = True

                    else:
                        if not self.network.graph.has_edge(trap_node, next_node):
                            self.network.graph.add_edge(trap_node, next_node, travel_time=0.001, is_counterflow=True)
                        else:
                            self.network.graph[trap_node][next_node]['travel_time'] = 0.001
                            self.network.graph[trap_node][next_node]['is_counterflow'] = True

                except IndexError:
                    pass # Reached the end of the line
                    
                # Safe Zone Exemption & AoE Gridlock Penalty
                # Expanded trap radius (approx 300 meters)
                trap_radius = 0.003 
                # Safe zone radius for Hospital and Incident (approx 50 meters)
                safe_radius = 0.0005 

                # Get exact coordinates of the Source and Incident nodes
                src_x, src_y = self.network.graph.nodes[source_node].get('x', 0), self.network.graph.nodes[source_node].get('y', 0)
                inc_x, inc_y = self.network.graph.nodes[incident_node].get('x', 0), self.network.graph.nodes[incident_node].get('y', 0)

                edges_iter = self.network.graph.edges(data=True, keys=True) if self.network.graph.is_multigraph() else self.network.graph.edges(data=True)
                
                for edge in edges_iter:
                    if self.network.graph.is_multigraph():
                        u, v, k, data = edge
                    else:
                        u, v, data = edge

                    node_x, node_y = self.network.graph.nodes[u].get('x', 0), self.network.graph.nodes[u].get('y', 0)
                    
                    # Calculate distances
                    dist_to_trap = ((node_x - trap_x)**2 + (node_y - trap_y)**2)**0.5
                    dist_to_src = ((node_x - src_x)**2 + (node_y - src_y)**2)**0.5
                    dist_to_inc = ((node_x - inc_x)**2 + (node_y - inc_y)**2)**0.5
                    
                    # Apply Gridlock ONLY if inside the trap net AND outside the safe zones
                    if dist_to_trap <= trap_radius and dist_to_src > safe_radius and dist_to_inc > safe_radius:
                        if data.get('is_counterflow', False):
                            # The VIP Police-Cleared Counterflow Lane (The Trigger)
                            data['travel_time'] = 0.001 
                        else:
                            # Paralyze main roads and alleyways (The Gridlock)
                            data['travel_time'] = data.get('travel_time', 1.0) * 1000.0

                # 5. Calculate Counterflow Detour & Stitch
                try:
                    route_part2_nodes = nx.shortest_path(self.network.graph, source=trap_node, target=incident_node, weight='travel_time')
                    actual_route_nodes = route_part1_nodes[:-1] + route_part2_nodes
                    
                    # Generate coordinates for the frontend payload with Visual Nudge
                    visual_route_coords = []
                    for i in range(len(actual_route_nodes) - 1):
                        u = actual_route_nodes[i]
                        v = actual_route_nodes[i+1]
                        y, x = self.network.graph.nodes[u].get('y'), self.network.graph.nodes[u].get('x')
                        
                        # Check if this specific step is the counterflow
                        edge_data = None
                        if self.network.graph.has_edge(u, v):
                            if self.network.graph.is_multigraph():
                                # Usually key 0 is the primary edge we modify
                                edge_data = self.network.graph.edges[u, v, 0] if self.network.graph.has_edge(u, v, 0) else list(self.network.graph[u][v].values())[0]
                            else:
                                edge_data = self.network.graph[u][v]
                                
                        if edge_data and edge_data.get('is_counterflow', False):
                            # Visual nudge (approx 15 meters) to physically separate the lines on the map
                            y += 0.00015
                            x -= 0.00015 
                            
                        visual_route_coords.append([y, x])

                    # Append the very last node
                    if actual_route_nodes:
                        last_node = actual_route_nodes[-1]
                        visual_route_coords.append([self.network.graph.nodes[last_node].get('y'), self.network.graph.nodes[last_node].get('x')])
                        
                    actual_route = visual_route_coords
                except nx.NetworkXNoPath:
                    actual_route = ideal_route # Fallback to prevent crash
            else:
                actual_route = []
            
        elif scenario_id == 'C':
            # 1:00 PM Agora Flood Event
            sim_time = 13.0
            target_lat, target_lon = 8.4907, 124.6568 # Agora Bus Terminal
            try:
                flooded_nodes = [ox.distance.nearest_nodes(self.network.graph, target_lon, target_lat, max_distance=2000)]
            except Exception:
                flooded_nodes = [ox.distance.nearest_nodes(self.network.graph, target_lon, target_lat)]
            
        elif scenario_id == 'D':
            # Scenario D: Carmen Bridge Hard Roadblock
            sim_time = 16.0
            target_lat, target_lon = 8.4740, 124.6350 # Past Carmen Bridge
            flooded_nodes = []
            
            # Find nearest station dynamically (e.g. NMMC)
            incident_node = ox.distance.nearest_nodes(self.network.graph, target_lon, target_lat)
            
            # 1. Filter Origin Station by Emergency Type
            if 'medical' in incident_type.lower():
                facility_type = 'Ambulance'
            else:
                facility_type = 'Fire Truck'
                
            valid_stations = [s for s in self.fleet_manager.stations.values() if s['type'] == facility_type]
            if not valid_stations:
                valid_stations = list(self.fleet_manager.stations.values())
                
            best_station_node = min(
                valid_stations, 
                key=lambda s: nx.shortest_path_length(self.network.graph, s['node'], incident_node, weight='length')
            )['node']
            
            try:
                # 2. Calculate the Synchronized Ideal Route
                ideal_route_nodes = nx.shortest_path(self.network.graph, best_station_node, incident_node, weight='length')
                ideal_route = [[self.network.graph.nodes[n]['y'], self.network.graph.nodes[n]['x']] for n in ideal_route_nodes]
            except Exception:
                ideal_route = []
                
            block_lat = 8.476301859252342
            block_lon = 124.64012430151547
            blockades = [[block_lat, block_lon]]
            
            try:
                G = self.network.graph
                
                print("\n=== GOV YSALINA BRIDGE HARDCODE NUKE ===")
                bridge_u = 4865698795
                bridge_v = 8794736872
                
                # STEP 2: Sever the bridge using the hardcoded nodes. Save the edge data.
                removed_edges = []
                is_multi = G.is_multigraph()
                for u, v in [(bridge_u, bridge_v), (bridge_v, bridge_u)]:
                    if G.has_edge(u, v):
                        if is_multi:
                            for k, data in list(G[u][v].items()):
                                removed_edges.append((u, v, k, data.copy()))
                                G.remove_edge(u, v, key=k)
                        else:
                            removed_edges.append((u, v, None, dict(G[u][v])))
                            G.remove_edge(u, v)

                print(f"Lanes successfully severed: {len(removed_edges)}")
                
                # Compute actual detour route and LEAVE GRAPH SEVERED so dispatcher works correctly
                actual_route_nodes = nx.shortest_path(G, best_station_node, incident_node, weight='length')
                actual_route = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in actual_route_nodes]

            except Exception as e:
                print(f"Error extracting bridge: {e}")
                actual_route = []
                    
        elif scenario_id == 'scenario_e':
            # Scenario E: Multi-Casualty Mass Dispatch
            sim_time = 14.0 # Default afternoon block
            target_lat, target_lon = 8.48, 124.65 # Coordinates overridden dynamically below
            flooded_nodes = []
            
        else:
            sim_time = 12.0
            target_lat, target_lon = 8.48, 124.65
            flooded_nodes = []
            
        if scenario_id == 'scenario_e':
            # 1. FORCE the fleet type (ignoring frontend inputs)
            incident_type = 'Medical Emergency'
            severity = 1
            
            # 2. FORCE exactly 9 distinct random nodes inside District 2 bounds
            largest_scc = max(nx.strongly_connected_components(self.network.graph), key=len)
            scc_nodes = list(largest_scc)
            valid_nodes = [
                n for n in scc_nodes 
                if 'y' in self.network.graph.nodes[n] and 'x' in self.network.graph.nodes[n]
            ]
            import random
            selected_nodes = random.sample(valid_nodes, 9)
            
            forced_incidents = []
            time_offset = 0
            for n in selected_nodes:
                forced_incidents.append({
                    'node': n,
                    'type': incident_type,
                    'severity': severity,
                    'time': time_offset,
                    'call_time_hrs': sim_time
                })
                # Add a tiny sequential offset if preferred, but Mass Casualty usually hits simultaneously:
                # time_offset += random.randint(0, 30) 
        else:
            if 'incident_node' not in locals():
                incident_node = ox.distance.nearest_nodes(self.network.graph, target_lon, target_lat)
            
            # Manually construct 1-incident payload tailored to CDO
            forced_incidents = [{
                'node': incident_node,
                'type': incident_type,
                'severity': severity,
                'time': 0,
                'call_time_hrs': sim_time,
                'ideal_route': locals().get('ideal_route', []),
                'actual_route': locals().get('actual_route', []),
                'blockades': locals().get('blockades', [])
            }]
        
        # Apply specific weather/disaster penalties
        if flooded_nodes:
            for center in flooded_nodes:
                for n in nx.single_source_shortest_path_length(self.network.graph, center, cutoff=2).keys():
                    for neighbor in self.network.graph.successors(n):
                        if self.network.graph.has_edge(n, neighbor):
                            self.network.graph[n][neighbor]['flooded'] = True
                            
        # Defer to existing payload renderer but manually pass the payload
        return self.run_custom_scenario(f"Chapter 4: Scenario {scenario_id}", forced_incidents=forced_incidents, is_case_study=True)

    def run_custom_scenario(self, scenario_name, num_incidents=5, incident_types=None, alarm_level=1, sim_time=None, forced_incidents=None, is_case_study=False):
        """A lightweight interactive runner designed for Flask, returning a purely realistic HTML map."""
        # Top-level deterministic lock
        random.seed(42)
        np.random.seed(42)
        
        # Use cast() to lock in concrete types — Pyre2 loses self's type in long methods
        network: RoadNetwork = cast(RoadNetwork, self.network)
        fm: FleetManager = cast(FleetManager, self.fleet_manager)
        fm.reset()

        print(f"\n--- Interactive Scenario Dashboard Run: {scenario_name} ---")
        if forced_incidents is not None:
             incidents = forced_incidents
        else:
             incidents = self.generate_incidents(num_incidents, incident_types, alarm_level, sim_time=sim_time)
        
        current_metrics: Dict[str, Dict[str, Any]] = {
            'Baseline': {'response_times': [], 'vehicles_deployed': 0, 'responding_stations': set(), 'reassignments_triggered': 0, 'uncovered': 0},
            'EGA': {'response_times': [], 'vehicles_deployed': 0, 'responding_stations': set(), 'reassignments_triggered': 0, 'uncovered': 0}
        }
        
        # Strict Variable Initialization to prevent UnboundLocalErrors
        incident_node = None
        ega_route = []
        base_route = []
        ega_results = {}
        base_results = {}
        
        dispatch_events_dict: Dict[int, Dict[str, Any]] = {}
        
        m = folium.Map(location=[8.48, 124.65], zoom_start=12, tiles='cartodbpositron')
        m.get_root().html.add_child(folium.Element("<script src='https://cdn.jsdelivr.net/npm/leaflet-ant-path@1.3.0/dist/leaflet-ant-path.js'></script>"))
        
        for sid, sdata in fm.stations.items():
            s_y = network.graph.nodes[sdata['node']]['y']
            s_x = network.graph.nodes[sdata['node']]['x']
            if sdata['type'] == 'Fire Truck':
                folium.Marker([s_y, s_x], popup=sdata['name'], icon=folium.Icon(color='red', icon='fire', prefix='fa')).add_to(m)
            else:
                folium.Marker([s_y, s_x], popup=sdata['name'], icon=folium.Icon(color='blue', icon='plus-square', prefix='fa')).add_to(m)
        
        for u, v, data in network.graph.edges(data=True):
            if data.get('hard_roadblock', False):
                n_y, n_x = network.graph.nodes[u]['y'], network.graph.nodes[u]['x']
                folium.Marker([n_y, n_x], icon=folium.Icon(color='red', icon='times-circle', prefix='fa')).add_to(m)
            if data.get('flooded', False):
                n_y, n_x = network.graph.nodes[u]['y'], network.graph.nodes[u]['x']
                folium.Marker([n_y, n_x], icon=folium.Icon(color='cadetblue', icon='water', prefix='fa')).add_to(m)
            if data.get('gridlocked', False):
                n_y, n_x = network.graph.nodes[u]['y'], network.graph.nodes[u]['x']
                folium.Marker([n_y, n_x], icon=folium.Icon(color='darkred', icon='ban', prefix='fa')).add_to(m)
            if data.get('is_black_wall', False) and not data.get('is_counterflow', False):
                try:
                    if 'geometry' in data:
                        coords = [(lat, lon) for lon, lat in data['geometry'].coords]
                    else:
                        coords = [(network.graph.nodes[n]['y'], network.graph.nodes[n]['x']) for n in [u, v]]
                    folium.PolyLine(coords, color='black', weight=9, opacity=1.0).add_to(m)
                except Exception:
                    pass
                
        # Incident Markers
        for i, inc in enumerate(incidents):
            inc_y, inc_x = network.graph.nodes[inc['node']]['y'], network.graph.nodes[inc['node']]['x']
            if inc['type'] == 'Fire Problem':
                icon = folium.Icon(color='lightred', icon='fire', prefix='fa')
            elif inc['type'] == 'Medical Emergency':
                icon = folium.Icon(color='green', icon='medkit', prefix='fa')
            else:
                icon = folium.Icon(color='lightblue', icon='life-ring', prefix='fa')
            folium.Marker([inc_y, inc_x], tooltip=f"Incident #{i}: {inc['type']}", icon=icon).add_to(m)
                
        for algo in ['Baseline', 'EGA']:
            fm.reset()  # type: ignore[attr-defined]
            for idx, incident in enumerate(incidents):
                fm.current_time = incident['time']  # type: ignore[attr-defined]
                req_result = fm.request_vehicles(incident['type'], incident['severity'], incident['node'], mode='realistic', algo=algo)  # type: ignore[attr-defined]
                vehicles = req_result[0] if req_result else None
                cross_count = req_result[1] if req_result else 0
                checked_stations = req_result[2] if req_result else []
                
                if not vehicles:
                    is_unreachable = True
                    for sid, sdata in fm.stations.items():  # type: ignore[attr-defined]
                        if sdata['type'] == ('Ambulance' if incident['type'] == 'Medical Emergency' else 'Fire Truck'):
                            try:
                                wt = nx.shortest_path_length(network.graph, sdata['node'], incident['node'], weight='length')
                                if wt < float('inf'):
                                    is_unreachable = False
                                    break
                            except nx.NetworkXNoPath:
                                pass
                                
                    current_metrics[algo]['uncovered'] += 1 # type: ignore
                    
                    if is_unreachable:
                        incident_lat = network.graph.nodes[incident['node']]['y']
                        incident_lng = network.graph.nodes[incident['node']]['x']
                        if idx not in dispatch_events_dict:
                            dispatch_events_dict[idx] = { # type: ignore
                                'incident_id': idx,
                                'type': str(incident['type']),
                                'incident_coord': [float(incident_lat), float(incident_lng)],
                                'base_data': None,
                                'ega_data': None,
                                'time_of_call': float(incident.get('call_time_hrs', 0.0))
                            }
                        
                        dispatch_events_dict[idx][algo] = { # type: ignore
                            'is_unreachable': True,
                            'error_msg': "Area Isolated by Blockades",
                            'vehicles_deployed': 0
                        }
                    continue
                    
                current_metrics[algo]['vehicles_deployed'] += len(vehicles) # type: ignore
                current_metrics[algo]['reassignments_triggered'] += cross_count # type: ignore
                for v in vehicles:
                    current_metrics[algo]['responding_stations'].add(v['station']) # type: ignore
                    
                fleet_results = []
                max_time_out = 0
                min_time_out = float('inf')
                max_ref_time = 0.0
                max_target_time = 0.0
                
                for v in vehicles:
                    station_node = fm.stations[v['station']]['node']  # type: ignore[attr-defined]
                    
                    path_out, time_out = None, float('inf')
                    path_ret, time_ret = None, float('inf')
                    
                    if algo == 'Baseline':
                        dispatcher_out = BaselineDispatcher(network, station_node, incident['node'], mode='realistic', start_time=fm.current_time, call_time_hrs=incident.get('call_time_hrs'), vehicle_type=v['type'])  # type: ignore[attr-defined]
                        path_out, time_out = dispatcher_out.run()
                        if incident['type'] == 'Fire Problem':
                            path_ret, time_ret = [incident['node']], 0.0
                        else:
                            dispatcher_ret = BaselineDispatcher(network, incident['node'], station_node, mode='realistic', start_time=fm.current_time + (time_out if time_out < float('inf') else 0), call_time_hrs=incident.get('call_time_hrs'), vehicle_type=v['type'])  # type: ignore[attr-defined]
                            path_ret, time_ret = dispatcher_ret.run()
                    else:
                        ngen, pop_size = (40, 200) if is_case_study else (40, 30)
                        if vehicles and len(vehicles) > 4:
                            pop_size = 30
                        mutpb = 0.8 if is_case_study else 0.8
                        elite_size = 5 if is_case_study else 2
                        dispatcher_out = EnhancedGA(network, station_node, incident['node'], mode='realistic', start_time=fm.current_time, call_time_hrs=incident.get('call_time_hrs'), vehicle_type=v['type'])  # type: ignore[attr-defined]
                        path_out, time_out = dispatcher_out.run(ngen=ngen, pop_size=pop_size, mutpb=mutpb, elite_size=elite_size)
                        if incident['type'] == 'Fire Problem':
                            path_ret, time_ret = [incident['node']], 0.0
                        else:
                            dispatcher_ret = EnhancedGA(network, incident['node'], station_node, mode='realistic', start_time=fm.current_time + (time_out if time_out < float('inf') else 0), call_time_hrs=incident.get('call_time_hrs'), vehicle_type=v['type'])  # type: ignore[attr-defined]
                            path_ret, time_ret = dispatcher_ret.run(ngen=ngen, pop_size=pop_size, mutpb=mutpb, elite_size=elite_size)
                        
                        if path_out and path_out[-1] != incident['node']: path_out.append(incident['node'])
                    is_counterflow = False
                    is_reroute = False
                    
                    if path_out and path_ret:
                        if time_out > max_time_out: max_time_out = time_out
                        if time_out < min_time_out: min_time_out = time_out
                        
                        flat_coords = []
                        route_segments = []
                        full_path = path_out + path_ret[1:]
                        if full_path:
                            flat_coords.append([network.graph.nodes[full_path[0]]['y'], network.graph.nodes[full_path[0]]['x']])
                            for idx_path in range(len(full_path)-1):
                                u_node = full_path[idx_path]
                                v_node = full_path[idx_path+1]
                                
                                seg_coords = []
                                seg_coords.append([network.graph.nodes[u_node]['y'], network.graph.nodes[u_node]['x']])
                                if network.graph.has_edge(u_node, v_node) and 'geometry' in network.graph[u_node][v_node]:
                                    coords = list(network.graph[u_node][v_node]['geometry'].coords)
                                    for lon, lat in islice(coords, 1, None):
                                        seg_coords.append([lat, lon])
                                        flat_coords.append([lat, lon])
                                else:
                                    seg_coords.append([network.graph.nodes[v_node]['y'], network.graph.nodes[v_node]['x']])
                                    flat_coords.append([network.graph.nodes[v_node]['y'], network.graph.nodes[v_node]['x']])
                                
                                is_cf = False
                                if network.graph.has_edge(u_node, v_node):
                                    is_cf = network.graph[u_node][v_node].get('is_counterflow', False)
                                
                                route_segments.append({'coords': seg_coords, 'is_cf': is_cf})
                            
                        if algo == 'EGA' and path_out:
                            # Count how many edges in the outbound path used counterflow
                            cf_count = 0
                            for u_node, next_v in zip(path_out[:-1], path_out[1:]):
                                if network.graph.has_edge(u_node, next_v):  # type: ignore
                                    edge_data = network.graph[u_node][next_v]  # type: ignore
                                    if edge_data.get('is_counterflow', False):
                                        cf_count += 1
                            is_counterflow = cf_count > 0

                            try:
                                ideal_path = nx.shortest_path(network.graph, source=station_node, target=incident['node'], weight='length')  # type: ignore[arg-type]
                                if path_out != ideal_path:
                                    is_reroute = True
                            except nx.NetworkXNoPath:
                                pass

                        ref_t = 0.0
                        targ_t = 0.0
                        if full_path:
                            ref_t, targ_t = calculate_actual_execution_time(network.graph, full_path, incident['type'], incident.get('call_time_hrs', 0.0))
                            if ref_t > max_ref_time: max_ref_time = ref_t
                            if targ_t > max_target_time: max_target_time = targ_t

                        fleet_results.append({
                            'station_id': str(v['station']),
                            'station_name': str(fm.stations[v['station']]['name']),  # type: ignore[attr-defined]
                            'route_segments': route_segments,
                            'route_coords': flat_coords,
                            'travel_time_mins': float(time_out / 60),
                            'actual_travel_time': float(ref_t),
                            'is_counterflow': is_counterflow,
                            'counterflow_count': cf_count if algo == 'EGA' else 0,
                            'is_reroute': is_reroute
                        })
                        
                        mission_duration = (time_out + time_ret) if (time_out < float('inf') and time_ret < float('inf')) else 3600
                        v['available_time'] = float(fm.current_time + mission_duration + v['turnaround_time'])  # type: ignore[index]
                        v['state'] = 'Available'  # type: ignore[index]
                        
                pan_js_array = []
                for sid in checked_stations:
                    st_node = fm.stations[sid]['node']  # type: ignore[attr-defined]
                    pan_js_array.append([network.graph.nodes[st_node]['y'], network.graph.nodes[st_node]['x']])
                    
                if not fleet_results:
                    # All engines failed routing
                    if idx not in dispatch_events_dict:
                        dispatch_events_dict[idx] = {
                            'incident_id': idx,
                            'type': str(incident['type']),
                            'incident_coord': [float(network.graph.nodes[incident['node']]['y']), float(network.graph.nodes[incident['node']]['x'])],
                            'base_data': None,
                            'ega_data': None,
                            'time_of_call': float(incident.get('call_time_hrs', 0.0))
                        }
                    key = 'base_data' if algo == 'Baseline' else 'ega_data'
                    dispatch_events_dict[idx][key] = { # type: ignore
                        'is_unreachable': True,
                        'error_msg': "Area Isolated by Blockades",
                        'vehicles_deployed': 0
                    }
                    current_metrics[algo]['uncovered'] += 1 # type: ignore
                    continue
                    
                incident_lat = network.graph.nodes[incident['node']]['y']
                incident_lng = network.graph.nodes[incident['node']]['x']
                
                convergence_time_mins = float((float(max_time_out) - float(min_time_out)) / 60) if min_time_out < float('inf') else 0.0  # type: ignore[operator]
                max_time_mins = float(float(max_time_out) / 60)  # type: ignore[operator]
                
                if idx not in dispatch_events_dict:
                    dispatch_events_dict[idx] = {
                        'incident_id': idx,
                        'type': str(incident['type']),
                        'incident_coord': [float(incident_lat), float(incident_lng)],
                        'base_data': None,
                        'ega_data': None,
                        'time_of_call': float(incident.get('call_time_hrs', 0.0)),
                        'ideal_route': incident.get('ideal_route', []),
                        'actual_route': incident.get('actual_route', []),
                        'blockades': incident.get('blockades', [])
                    }
                    
                if algo == 'Baseline':
                    dispatch_events_dict[idx]['base_travel_time'] = float(max_ref_time)
                    dispatch_events_dict[idx]['target_time_mins'] = float(max_target_time)
                else:
                    dispatch_events_dict[idx]['ega_travel_time'] = float(max_ref_time)
                    
                key = 'base_data' if algo == 'Baseline' else 'ega_data'
                dispatch_events_dict[idx][key] = {
                    'swarm': fleet_results,
                    'is_unreachable': False,
                    'panned_stations': pan_js_array,
                    'travel_time': float(max_time_mins),
                    'station': str(fleet_results[0]['station_name']) if fleet_results else 'None',
                    'convergence_time_mins': convergence_time_mins,
                    'vehicles_deployed': int(len(fleet_results)),
                    'reassignments': int(cross_count),
                    'is_counterflow': any(r.get('is_counterflow', False) for r in fleet_results),
                    'counterflow_count': sum(r.get('counterflow_count', 0) for r in fleet_results),
                    'is_reroute': any(r.get('is_reroute', False) for r in fleet_results),
                }
                
                current_metrics[algo]['response_times'].append(max_time_out) # type: ignore # type: ignore
                        
        dispatch_events_js_list = [v for k, v in sorted(dispatch_events_dict.items())]

        # Dashboard now lives in index.html (Window B); iframe uses postMessage bridge.
        dashboard_html = ''
        m.get_root().html.add_child(folium.Element(dashboard_html))  # type: ignore[attr-defined]
        self.inject_global_animation_script(m, dispatch_events_js_list)  # type: ignore[attr-defined]
        
        # Return purely the HTML string for the web browser iframe
        m.get_root().render()
        iframe_html = m.get_root()._repr_html_()
        return iframe_html
    def inject_global_animation_script(self, m, dispatch_events):
        events_json = json.dumps(dispatch_events)
        stations_json = json.dumps([])
        legend_html = '''
{% macro html(this, kwargs) %}
<div id="thesis-hud" style="position: fixed; top: 20px; right: 20px; width: 340px; background-color: rgba(255, 255, 255, 0.96); z-index: 9999; padding: 20px; border-radius: 6px; border: 1px solid #e0e0e0; box-shadow: 0 10px 30px rgba(0,0,0,0.15); font-family: sans-serif;">
    <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #333; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 6px;">Command Center HUD</div>
    <div id="live_status" style="font-size: 11px; color: #555; margin-bottom: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">SYSTEM STANDBY...</div>
    
    <!-- Baseline Section -->
    <div style="border-left: 4px solid #dc3545; padding-left: 10px; margin-bottom: 16px;">
        <div style="font-size: 12px; font-weight: 700; color: #dc3545; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Baseline Routing</div>
        <div style="display: grid; grid-template-columns: 1fr auto; row-gap: 4px; font-size: 12px; color: #444;">
            <div>Vehicles Deployed</div>
            <div style="font-family: 'Courier New', Courier, monospace; font-weight: bold; text-align: right;" id="base_deployed">0</div>
            <div>Outbound Avg (mins)</div>
            <div style="font-family: 'Courier New', Courier, monospace; font-weight: bold; text-align: right;" id="base_avg">0.00</div>
        </div>
    </div>
    
    <!-- EGA Section -->
    <div style="border-left: 4px solid #0d6efd; padding-left: 10px;">
        <div style="font-size: 12px; font-weight: 700; color: #0d6efd; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">EGA Optimization</div>
        <div style="display: grid; grid-template-columns: 1fr auto; row-gap: 4px; font-size: 12px; color: #444;">
            <div>Vehicles Deployed</div>
            <div style="font-family: 'Courier New', Courier, monospace; font-weight: bold; text-align: right;" id="ega_deployed">0</div>
            <div>Outbound Avg (mins)</div>
            <div style="font-family: 'Courier New', Courier, monospace; font-weight: bold; text-align: right;" id="ega_avg">0.00</div>
            <div>Counterflow Segments</div>
            <div style="font-family: 'Courier New', Courier, monospace; font-weight: bold; text-align: right; color: #495057; background: #e9ecef; padding: 0 4px; border-radius: 3px;" id="ega_cf">0</div>
            <div>Active Reroutes</div>
            <div style="font-family: 'Courier New', Courier, monospace; font-weight: bold; text-align: right;" id="ega_rr">0</div>
        </div>
    </div>
    
    <!-- Analytics & Live Logging -->
    <div style="margin-top: 15px; border-top: 1px solid #ddd; padding-top: 10px;">
        <div style="display: grid; grid-template-columns: 1fr auto; font-size: 13px; font-weight: 700; color: #155724;">
            <div>Improvement:</div>
            <div id="ega_imp">0.0%</div>
        </div>
    </div>
    
    <div style="margin-top: 15px; font-size: 11px; color: #666; font-weight: bold;">LIVE COMMS LOG</div>
    <div id="live_log" style="font-size: 10px; font-family: monospace; max-height: 100px; overflow-y: auto; background: #000; color: #0f0; padding: 6px; border-radius: 4px; display: flex; flex-direction: column-reverse;"></div>
</div>
{% endmacro %}
'''
        macro = MacroElement()
        macro._template = Template(legend_html)
        m.get_root().add_child(macro)
        
        anim_js = f"""
        <script>
            let bTotal = 0, bCount = 0, eTotal = 0, eCount = 0, eCF = 0, eRR = 0, bDeployedTotal = 0, eDeployedTotal = 0;
            
            document.addEventListener("DOMContentLoaded", function() {{
                setTimeout(function() {{
                    var map = null;
                    for (var key in window) {{
                        if (key.startsWith('map_')) {{ map = window[key]; break; }}
                    }}
                    
                    if (map) {{
                        var events = {events_json};
                        var stations_data = {stations_json};
                        
                        function logMsg(msg) {{
                            var el = document.getElementById('live_log');
                            if (el) {{
                                var div = document.createElement('div');
                                div.style.marginBottom = "4px";
                                div.innerText = '> ' + msg;
                                el.prepend(div);
                            }}
                        }}
                        
                        logMsg("System initialized. Standing by...");

                        if (window.parent && window.parent.document) {{
                            var box = window.parent.document.getElementById('status-box');
                            if (box) {{
                                box.innerHTML = '\ud83d\udfe2 Simulation loaded &mdash; <strong>' + events.length + ' incident(s)</strong> processing\u2026';
                                box.className = '';
                            }}
                        }}
                        
                        function interpolateLineFrames(coords, totalFrames, incidentType, incidentCoord) {{
                            if (!coords || coords.length < 2) return coords;
                            var totalDist = 0;
                            var segments = [];
                            for (var i = 0; i < coords.length - 1; i++) {{
                                var p1 = L.latLng(coords[i][0], coords[i][1]);
                                var p2 = L.latLng(coords[i+1][0], coords[i+1][1]);
                                var d = p1.distanceTo(p2);
                                totalDist += d;
                                segments.push({{p1: p1, p2: p2, dist: d}});
                            }}
                            var stepMeters = totalDist / Math.max(1, totalFrames);
                            var interpolated = [coords[0]];
                            var accum = 0;
                            
                            for (var f = 1; f < totalFrames; f++) {{
                                var targetDist = f * stepMeters;
                                var currentAccum = 0;
                                for (var s = 0; s < segments.length; s++) {{
                                    if (currentAccum + segments[s].dist >= targetDist || s === segments.length - 1) {{
                                        var excess = targetDist - currentAccum;
                                        var ratio = (segments[s].dist > 0) ? (excess / segments[s].dist) : 1;
                                        ratio = Math.max(0, Math.min(1, ratio));
                                        var lat = segments[s].p1.lat + (segments[s].p2.lat - segments[s].p1.lat) * ratio;
                                        var lng = segments[s].p1.lng + (segments[s].p2.lng - segments[s].p1.lng) * ratio;
                                        interpolated.push([lat, lng]);
                                        break;
                                    }}
                                    currentAccum += segments[s].dist;
                                }}
                            }}
                            interpolated.push(coords[coords.length-1]);
                            
                            if (incidentType === 'Medical Emergency') {{
                                var closestIdx = Math.floor(interpolated.length / 2);
                                if (incidentCoord) {{
                                    var minDist = Infinity;
                                    var targetLL = L.latLng(incidentCoord[0], incidentCoord[1]);
                                    for (var k=0; k<interpolated.length; k++) {{
                                        var dist = targetLL.distanceTo(L.latLng(interpolated[k][0], interpolated[k][1]));
                                        if (dist < minDist) {{
                                            minDist = dist;
                                            closestIdx = k;
                                        }}
                                    }}
                                }}
                                var pauseFrames = [];
                                // Exact snap to incident location
                                var exactIconCoord = incidentCoord ? incidentCoord : interpolated[closestIdx];
                                for (var p = 0; p < 100; p++) pauseFrames.push(exactIconCoord);
                                interpolated.splice.apply(interpolated, [closestIdx, 0].concat(pauseFrames));
                            }}
                            return interpolated;
                        }}
                        
                        function runEventSequence() {{
                            // Ensure previous markers and intervals are destroyed
                            if (window._animIntervals) {{
                                window._animIntervals.forEach(clearInterval);
                            }}
                            window._animIntervals = [];

                            if (window._animMarkers) {{
                                window._animMarkers.forEach(function(m) {{ map.removeLayer(m); }});
                            }}
                            window._animMarkers = [];

                            // Render static visual detours for dual-routing (e.g., Scenario D Blockades)
                            events.forEach(function(event) {{
                                if (event.ideal_route && event.ideal_route.length > 0) {{
                                    L.polyline(event.ideal_route, {{ color: '#fce803', weight: 4, dashArray: '5, 10', opacity: 0.7 }}).addTo(map);
                                }}
                                if (event.actual_route && event.actual_route.length > 0) {{
                                    L.polyline(event.actual_route, {{ color: 'blue', weight: 5 }}).addTo(map);
                                    
                                    var displayIcon = (event.type === 'Medical Emergency') ? '🚑' : '🚒';
                                    var iconHtml = "<div style='background:#0000FF;width:20px;height:20px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 8px rgba(0,0,0,0.8);display:flex;align-items:center;justify-content:center;font-size:12px;'>" + displayIcon + "</div>";
                                    var vIcon = L.divIcon({{ html: iconHtml, className: '', iconSize: [20, 20], iconAnchor: [10, 10] }});
                                    var vehicleMarker = L.marker(event.actual_route[0], {{icon: vIcon}}).addTo(map);
                                    window._animMarkers.push(vehicleMarker);
                                    
                                    var routeIdx = 0;
                                    var interval = setInterval(function() {{
                                        if (routeIdx < event.actual_route.length) {{
                                            vehicleMarker.setLatLng(event.actual_route[routeIdx]);
                                            routeIdx++;
                                        }} else {{
                                            clearInterval(interval);
                                        }}
                                    }}, 150);
                                    window._animIntervals.push(interval);
                                }}
                                if (event.blockades && event.blockades.length > 0) {{
                                    event.blockades.forEach(function(c) {{
                                        var icon = L.divIcon({{ html: '🚧', className: 'blockade-icon', iconSize: [20, 20] }});
                                        L.marker(c, {{icon: icon}}).addTo(map);
                                    }});
                                }}
                            }});
                            
                            var eventPromises = [];
                            
                            events.forEach(function(ev, evIndex) {{
                                eventPromises.push(new Promise(function(resolve) {{
                                    var step_delay = 2000;
                                    var current_delay = 0;
                                    
                                    setTimeout(function() {{ 
                                        map.flyTo(ev.incident_coord, 15, {{duration: 1.5}}); 
                                        var isUnreachable = (ev.ega_data && ev.ega_data.is_unreachable);
                                        var statusHtml;
                                        if (isUnreachable) {{
                                            statusHtml = "\ud83d\udeab CRITICAL: Incident #" + ev.incident_id + " \u2013 UNREACHABLE (road blockade)";
                                        }} else {{
                                            var b_name  = (ev.base_data && ev.base_data.station) ? ev.base_data.station : 'Uncovered';
                                            var e_name  = (ev.ega_data && ev.ega_data.station) ? ev.ega_data.station : 'Uncovered';
                                            var b_units = (ev.base_data) ? (ev.base_data.vehicles_deployed || 0) : 0;
                                            var e_units = (ev.ega_data) ? (ev.ega_data.vehicles_deployed || 0) : 0;
                                            statusHtml =
                                                '\ud83d\udea8 Incident #' + ev.incident_id + ' (' + ev.type + ')<br>' +
                                                '<span style="color:#FF0000; font-weight:bold;">BL \u2794 ' + b_name + ' (' + b_units + ' units)</span><br>' +
                                                '<span style="color:#0000FF; font-weight:bold;">EGA \u2794 ' + e_name + ' (' + e_units + ' units)</span>';
                                        }}
                                        if (window.parent && window.parent.document) {{
                                            var box = window.parent.document.getElementById('status-box');
                                            if (box) box.innerHTML = statusHtml;
                                        }}
                                    }}, current_delay);
                                    current_delay += step_delay;
                                    
                                    setTimeout(function() {{
                                        if (ev.ega_data && ev.ega_data.is_unreachable) {{ resolve(); }}
                                    }}, current_delay);
                                    
                                    var all_stations = [];
                                    if (ev.base_data && ev.base_data.panned_stations) all_stations = all_stations.concat(ev.base_data.panned_stations);
                                    if (ev.ega_data && ev.ega_data.panned_stations) all_stations = all_stations.concat(ev.ega_data.panned_stations);
                                    var unique_stations = [];
                                    var seen = new Set();
                                    all_stations.forEach(function(st) {{
                                        var key = st[0] + "," + st[1];
                                        if (!seen.has(key)) {{ seen.add(key); unique_stations.push(st); }}
                                    }});
                                    unique_stations.forEach(function(st_coord) {{
                                        setTimeout(function() {{ map.flyTo(st_coord, 15, {{duration: 1.5}}); }}, current_delay);
                                        current_delay += step_delay;
                                    }});
                                    
                                    setTimeout(function() {{
                                        var start_coord = null;
                                        if (ev.base_data && ev.base_data.swarm) {{
                                            ev.base_data.swarm.forEach(function(veh) {{
                                                if (veh.route_segments && veh.route_segments.length > 0) {{
                                                    veh.route_segments.forEach(function(seg) {{
                                                        L.polyline.antPath(seg.coords, {{color: '#FF0000', delay: 400, weight: 4, opacity: 0.65, loop: false}}).addTo(map);
                                                    }});
                                                    if (!start_coord) start_coord = veh.route_segments[0].coords[0];
                                                }}
                                            }});
                                        }}
                                        if (ev.ega_data && ev.ega_data.swarm) {{
                                            ev.ega_data.swarm.forEach(function(veh) {{
                                                if (veh.route_segments && veh.route_segments.length > 0) {{
                                                    veh.route_segments.forEach(function(seg) {{
                                                        var segColor = seg.is_cf ? '#FFFF00' : '#0000FF';
                                                        L.polyline.antPath(seg.coords, {{color: segColor, delay: 400, weight: 5, opacity: 0.85, loop: false}}).addTo(map);
                                                    }});
                                                    if (!start_coord) start_coord = veh.route_segments[0].coords[0];
                                                }}
                                            }});
                                            if (evIndex === 0) logMsg("EGA optimization bypassed localized gridlock.");
                                        }}
                                        if (start_coord) map.flyTo(start_coord, 14, {{duration: 1.5}});
                                    }}, current_delay);
                                    current_delay += step_delay;
                                    
                                    setTimeout(function() {{
                                        if (!map.getPane('markersPane')) {{
                                            map.createPane('markersPane');
                                            map.getPane('markersPane').style.zIndex = 650;
                                        }}
                                        var tracks = [];
                                        function makeMarker(coords, color, label, travelTimeMins, incidentType, incidentCoord) {{
                                            var animMs = Math.max(2000, travelTimeMins * 400);
                                            var frames = Math.floor(animMs / 20);
                                            
                                            var smooth = interpolateLineFrames(coords, frames, incidentType, incidentCoord);
                                            if (!smooth || smooth.length < 1) return null;
                                            
                                            var displayIcon = (incidentType === 'Medical Emergency') ? '🚑' : '🚒';
                                            var iconHtml = "<div title='" + label + "' style='background:" + color + ";width:20px;height:20px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 8px rgba(0,0,0,0.8);display:flex;align-items:center;justify-content:center;font-size:12px;'>" + displayIcon + "</div>";
                                            
                                            var icon = L.divIcon({{
                                                className: '',
                                                html: iconHtml,
                                                iconSize: [20, 20], iconAnchor: [10, 10]
                                            }});
                                            var m = L.marker(smooth[0], {{icon: icon, pane: 'markersPane'}}).addTo(map);
                                            return {{smooth: smooth, idx: 0, marker: m}};
                                        }}

                                        if (ev.base_data && ev.base_data.swarm) {{
                                            ev.base_data.swarm.forEach(function(v, i) {{
                                                if (v.route_coords && v.route_coords.length > 1) {{
                                                    var t = makeMarker(v.route_coords, '#FF0000', 'BL-' + i, v.travel_time_mins || ev.base_data.travel_time || 1, ev.type, ev.incident_coord);
                                                    if (t) {{ tracks.push(t); }}
                                                }}
                                            }});
                                        }}
                                        if (ev.ega_data && ev.ega_data.swarm) {{
                                            ev.ega_data.swarm.forEach(function(v, i) {{
                                                if (v.route_coords && v.route_coords.length > 1) {{
                                                    var t = makeMarker(v.route_coords, '#0000FF', 'EGA-' + i, v.travel_time_mins || ev.ega_data.travel_time || 1, ev.type, ev.incident_coord);
                                                    if (t) {{ tracks.push(t); }}
                                                }}
                                            }});
                                        }}
                                        
                                        if (tracks.length === 0) {{ resolve(); return; }}

                                        var animInterval = setInterval(function() {{
                                            var allDone = true;
                                            tracks.forEach(function(t) {{
                                                if (t.idx < t.smooth.length - 1) {{
                                                    allDone = false;
                                                    t.idx++;
                                                    t.marker.setLatLng(t.smooth[t.idx]);
                                                }} else {{
                                                    t.marker.setLatLng(t.smooth[t.smooth.length - 1]);
                                                }}
                                            }});

                                            if (allDone) {{
                                                clearInterval(animInterval);
                                                
                                                // Ensure markers snap exactly to last coordinate to prevent bouncing
                                                tracks.forEach(function(t) {{
                                                    t.marker.setLatLng(t.smooth[t.smooth.length - 1]);
                                                    if (t.marker.stop) t.marker.stop();
                                                }});
                                                
                                                var isIncidentStart = true;
                                                var incident = ev;
                                                
                                                if (incident.type === 'Medical Emergency') {{
                                                    if (incident.ega_data && incident.ega_data.swarm) {{
                                                        logMsg("Medical Unit [" + incident.incident_id + "] returned to base.");
                                                    }}
                                                }}
                                                
                                                if (isIncidentStart) {{
                                                    let routeDest = "Unknown";
                                                    if (incident.base_data && incident.base_data.station) routeDest = incident.base_data.station;
                                                    document.getElementById('live_status').innerText = 'INCIDENT #' + incident.incident_id + ' (' + incident.type + ') ➔ ' + routeDest.toUpperCase();

                                                    if (incident.base_travel_time !== undefined) {{
                                                        bCount++;
                                                        bTotal += parseFloat(incident.base_travel_time);
                                                        if (incident.base_data && incident.base_data.vehicles_deployed) {{
                                                            bDeployedTotal += parseInt(incident.base_data.vehicles_deployed);
                                                        }}
                                                    }}
                                                    if (incident.ega_travel_time !== undefined) {{
                                                        eCount++;
                                                        eTotal += parseFloat(incident.ega_travel_time);
                                                        if (incident.ega_data) {{
                                                            if (incident.ega_data.vehicles_deployed) eDeployedTotal += parseInt(incident.ega_data.vehicles_deployed);
                                                            if (incident.ega_data.is_counterflow || incident.ega_data.counterflow_count > 0) eCF += Math.max(1, incident.ega_data.counterflow_count || 1);
                                                            if (incident.ega_data.is_reroute) eRR++;
                                                        }}
                                                    }}

                                                    if (document.getElementById('base_avg')) {{
                                                        var dBase = (bCount > 0) ? (bTotal / bCount) : 0;
                                                        var dEga = (eCount > 0) ? (eTotal / eCount) : 0;

                                                        document.getElementById('base_avg').innerText = dBase.toFixed(2);
                                                        document.getElementById('base_deployed').innerText = bDeployedTotal;
                                                        
                                                        document.getElementById('ega_avg').innerText = dEga.toFixed(2);
                                                        document.getElementById('ega_deployed').innerText = eDeployedTotal;
                                                        
                                                        if (document.getElementById('ega_rr')) document.getElementById('ega_rr').innerText = eRR;
                                                        if (document.getElementById('ega_cf')) {{
                                                            var cfElem = document.getElementById('ega_cf');
                                                            cfElem.innerText = eCF;
                                                            if (eCF > 0) {{
                                                                cfElem.style.background = '#d4edda';
                                                                cfElem.style.color = '#155724';
                                                            }}
                                                        }}
                                                        
                                                        if (eDeployedTotal > bDeployedTotal) {{
                                                            var preventedDenials = eDeployedTotal - bDeployedTotal;
                                                            if (document.getElementById('ega_imp')) {{
                                                                var impElem = document.getElementById('ega_imp');
                                                                impElem.innerText = "+" + preventedDenials + " Rescues (Denial Mitigated)";
                                                                impElem.style.color = "#155724";
                                                            }}
                                                            logMsg("EGA successfully mitigated " + preventedDenials + " Baseline service denial(s).");
                                                        }} else {{
                                                            var pct = dBase > 0 ? ((dBase - dEga) / dBase) * 100 : 0;
                                                            if (document.getElementById('ega_imp')) {{
                                                                var impElem = document.getElementById('ega_imp');
                                                                impElem.innerText = pct.toFixed(1) + "%";
                                                            }}
                                                            logMsg("Unit arrived at Scene. " + pct.toFixed(1) + "% EGA Time reduction");
                                                        }}
                                                    }}
                                                }}
                                                
                                                // Check if this is the final incident
                                                if (evIndex === events.length - 1) {{
                                                    setTimeout(function() {{
                                                        var box = document.getElementById('live_status');
                                                        if (box) {{
                                                            box.innerHTML = '\u2705 Mission Accomplished';
                                                        }}
                                                    }}, 1000);
                                                }}
                                                
                                                resolve();
                                            }}
                                        }}, 20); // 50fps
                                    }}, current_delay);
                                }}));
                            }});
                            Promise.all(eventPromises).then(function() {{
                                logMsg("All coordinated dispatches complete.");
                            }});
                        }}
                        
                        runEventSequence();
                    }}
                }}, 1000);
            }});
        </script>
        """
        m.get_root().html.add_child(folium.Element(anim_js))

    def print_results(self):
        print("\n" + "="*85)
        print(f"{'Scenario':<35} | {'Baseline Avg (min)':<20} | {'EGA Avg (min)':<15} | {'Improvement (%)'}")
        print("-" * 85)
        
        for stat in self.scenario_stats:
            print(f"{stat['Scenario']:<35} | {stat['Baseline Avg (min)']:<20.2f} | {stat['EGA Avg (min)']:<15.2f} | {stat['Improvement (%)']:.2f}%")
        print("="*85 + "\n")


def main():
    random.seed(42)
    print("Initializing Autonomous Emergency Dispatch Simulation...")
    network = RoadNetwork()
    network.load_network()
    
    fleet_manager = FleetManager(network)
    engine = SimulationEngine(network, fleet_manager)

    # 1. Normal Conditions
    engine.run_scenario("Normal Conditions", num_incidents=5)

    # 2. Scenario A - Gridlock
    network.reset_scenarios()
    network.apply_scenario_gridlock()
    engine.run_scenario("Scenario A - Gridlock", num_incidents=10)

    # 3. Scenario B - Flood
    network.reset_scenarios()
    network.apply_scenario_flood()
    engine.run_scenario("Scenario B - Flood", num_incidents=10)

    # 4. Scenario C - Fleet Saturation
    network.reset_scenarios()
    engine.run_scenario("Scenario C - Fleet Saturation (High Demand)", num_incidents=30)

    # Results
    engine.print_results()

    # Response time comparison plot
    try:
        times_b = [t/60 for t in engine.all_real_b_times]
        times_e = [t/60 for t in engine.all_real_e_times]
        plt.figure(figsize=(10,5))
        plt.hist(times_b, alpha=0.5, label='Baseline', bins=20)
        plt.hist(times_e, alpha=0.5, label='EGA', bins=20)
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (minutes)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('response_time_comparison.png')
        print("Metric plot saved to 'response_time_comparison.png'")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
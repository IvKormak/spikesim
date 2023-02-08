import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import trange
from dataclasses import dataclass
import csv

class SpikeNetworkSim:
    def __init__(self, inputs_l = 1, labels = None,dt = 1):
        
        if labels is None:
            labels = range(inputs_l)
            
        self.labels_dict = dict(zip(labels, range(inputs_l)))
        self.values = pd.DataFrame(columns=range(inputs_l))
        
        input_nodes = [
            {
                "type": "input",
                "priority": 0,
                "listening": None,
                "broadcasting": None,
                "layer": -1
            } for _ in range(inputs_l)]
        
        self.nodes = pd.DataFrame(data=input_nodes, columns=["type", "priority", "listening", "broadcasting"])
        self.weights = pd.DataFrame(columns=["weights", "inhibited"])
        self.layers = pd.DataFrame(columns=["layer"])
        self.dt = dt
        
        self.layer_params = {
            "tau_refractory": [],
            "tau_inhibitory": [],
            "tau_ltp": [],
            "tau_leak": [],
            "thres": [],
            "ainc": [],
            "adec": [],
            "wmin": [],
            "wmax": [],
            "learning": [],
            "wta": [],
            "desensibilization": []
        }
        
        self.defaults = {
            "tau_inhibitory": 3, 
            "tau_refractory": 5, 
            "tau_leak": 10, 
            "tau_ltp": 5, 
            "thres": 200,
            "ainc": 30, 
            "adec": -15, 
            "wmax": 255,
            "wmin": 1,
            "learning": True,
            "wta": False,
            "desensibilization": None
        }
        
    def new_layer(self, width, weights=None, labels=None, **layer_params):
        #print(f"{inputs_l=},{labels=},{dt=},{tau_inhibitory=},{tau_refractory=},{tau_leak=},{tau_ltp=},{thres=},{ainc=},{adec=},{wmax=},{wmin=}")
        for param in layer_params.keys():
            if param in layer_params:
                self.layer_params[param].append(layer_params[param])
            else:
                raise Exception(f"Parameter {param} doesn't exist")
        for dparam in self.defaults.keys():
            if dparam not in layer_params:
                self.layer_params[dparam].append(self.defaults[dparam])
                
                
        nnodes = []
        nweights = []
        nlayers = []
        priority = self.nodes["priority"].max()+1
        layer = self.nodes["layer"].max()+1
        
        if layer == 0: #первый слой
            inputs = np.array(self.nodes.query("priority==0").index.tolist())
        else: #нужно пропустить потенцирующие ноды
            inputs = np.array(self.nodes.query("priority==@priority-2").index.tolist())
        if weights is None or weights.shape[0]==0:
            weights = np.random.randint(self.layer_params["wmin"][-1], self.layer_params["wmax"][-1], (width, inputs.shape[0]))
        elif weights.shape[1] > inputs.shape[0] or weights.shape[0] > width:
            raise Exception(f"Требуется массив (1...{width},{inputs.shape[0]}), получено {weights.shape}")
        elif weights.shape[0] < width:
            weights = np.concatenate((weights, np.random.randint(self.layer_params["wmin"][-1], self.layer_params["wmax"][-1], (width-weights.shape[0], inputs.shape[0]))))
            
        node_id = self.nodes.index.size
        presynaptic_id = node_id+inputs.shape[0]
        postsynaptic_id = node_id+inputs.shape[0]+1
        
        layer_ltp_range = np.arange(node_id, node_id+inputs.shape[0])
        layer_presynaptic_range = np.arange(width)*(3)+presynaptic_id
        layer_postsynaptic_range = np.arange(width)*(3)+postsynaptic_id
        
        for i in inputs:
            nnodes.append(
                {
                    "type": "ltp",
                    "listening": i,
                    "broadcasting": None,
                    "priority": priority,
                    "layer": layer
                }
            )
                
        
        if labels is None:
            labels = layer_postsynaptic_range
        self.labels_dict.update(dict(zip(layer_postsynaptic_range, labels)))
        
        for w in weights:
            nnodes.append(
                {
                    "type": "presynaptic",
                    "listening": inputs,
                    "broadcasting": postsynaptic_id,
                    "priority": priority+1,
                    "layer": layer
                }
            )
            nweights.append(
                {
                    "node": presynaptic_id,
                    "weights": w,
                    "inhibited": -1
                }
            )
            
            nnodes.append(
                {
                    "type": "postsynaptic",
                    "listening": presynaptic_id,
                    "broadcasting": layer_presynaptic_range[layer_presynaptic_range != presynaptic_id],
                    "priority": priority+2,
                    "layer": layer
                }
            )
            nnodes.append(
                {
                    "type": "potentiating",
                    "listening": layer_ltp_range,
                    "broadcasting": presynaptic_id,
                    "priority": priority+3,
                    "layer": layer
                }
            )
            
            presynaptic_id += 3
            postsynaptic_id += 3
        nlayers = [{"layer": layer} for _ in range(node_id, presynaptic_id)]
        self.nodes = pd.concat((self.nodes, pd.DataFrame(nnodes))).reset_index(drop=True)
        self.weights = pd.concat((self.weights, pd.DataFrame(nweights).set_index("node", drop=True)))
        self.layers = pd.concat((self.layers, pd.DataFrame(nlayers))).reset_index(drop=True)
    
    def make_recurrent(self):
        max_priority = self.nodes["priority"].max()
        last_layer_output = np.array(self.nodes.query("priority==@max_priority-1").index.tolist())
        first_layer_summators = np.array(self.nodes.query("priority==2").index.tolist())
        nlayer = [{"layer": 0} for _ in last_layer_output]
        nnodes = [
            {
                "type": "recurrent",
                "listening": o,
                "broadcasting": None,
                "priority": max_priority+1,
                "layer": 0
            } for o in last_layer_output
        ]
        recurrent_presynaptic_indexes = np.arange(self.nodes.index.size, self.nodes.index.size+last_layer_output.shape[0])
        nweights = self.weights.to_dict()
        nnodes = pd.concat((self.nodes, pd.DataFrame(nnodes))).reset_index(drop=True).to_dict()
        for s in first_layer_summators:
            nweights["weights"][s] = np.concatenate((nweights["weights"][s], np.random.randint(self.wmin, self.wmax, last_layer_output.shape[0])))
            nnodes["listening"][s] = np.concatenate((nnodes["listening"][s], recurrent_presynaptic_indexes))
            nnodes["listening"][s+2] = np.concatenate((nnodes["listening"][s+2], recurrent_presynaptic_indexes))
        self.nodes = pd.DataFrame(nnodes)
        self.weights = pd.DataFrame(nweights)
        self.layers = pd.concat((self.layers, pd.DataFrame(nlayers))).reset_index(drop=True)
    
    def stepwise_generator(self, data):
        vals_z = np.zeros(data.shape[1])
        nodes_sorted = self.nodes.sort_values("priority")
        nodes_sorted, index_sorted = nodes_sorted.values, nodes_sorted.index
        
        for t, vals in enumerate(data):
            layer = None
            for (node_type, _, listen, cast, _layer), node in zip(nodes_sorted, index_sorted):
                if node_type == "input":
                    continue
                if _layer != layer:
                    layer = _layer
                    params = {k: self.layer_params[k][layer] for k in self.layer_params.keys()}
                    params["leak"] = np.exp(-self.dt/params["tau_leak"])
                    
                n_val = vals[node]
                if node_type == "recurrent":
                    n_val = vals[listen]
                if node_type == "ltp":
                    if vals[listen]:
                        n_val = 0
                    else:
                        n_val = vals_z[node]+1
                elif node_type == "presynaptic":
                    if self.weights.at[node, "inhibited"] < t:
                        n_val = (vals[listen]*self.weights.at[node, "weights"]).sum()+vals_z[node]*params["leak"]
                elif node_type == "postsynaptic":
                    n_val = int(vals[listen]>params["thres"])
                    if n_val:
                        self.weights.at[listen, "inhibited"] = t+params["tau_refractory"]
                        for b in cast:
                            if params["wta"]:
                                vals[b] = 0
                            self.weights.at[b, "inhibited"] = max(t+params["tau_inhibitory"], self.weights.at[b, "inhibited"]+params["tau_inhibitory"])
                elif node_type == "potentiating":
                    if vals[node-1] and params["learning"]:
                        nw = self.weights.at[cast, "weights"] + np.where(vals[listen]<params["tau_ltp"], params["ainc"], params["adec"])
                        nw = np.where(nw>params["wmax"], params["wmax"], nw)
                        self.weights.at[cast, "weights"] = np.where(nw<params["wmin"], params["wmin"], nw)

                vals[node] = n_val
            vals_z = vals
            yield dict(zip(self.nodes.index, vals))
                
    def feed_csv(self, data_csv, out_csv, data_timestep=1):
        self.weights["inhibited"].values[:] = 0
        time_scale_factor = data_timestep//self.dt
        with open(data_csv, "r", newline='') as f:
            datareader = csv.DictReader(f, delimiter='\t')
            #строки будут повторяться чтобы привести данные к временному шагу расчёта
            data = pd.DataFrame([{int(k): float(v) for k,v in row.items()} for row in datareader for dt in np.arange(time_scale_factor)], columns=self.nodes.index).fillna(0)
        s = self.stepwise_generator(data)
        out = [u.to_dict() for u in s]
        self.values = pd.DataFrame(out)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.values.columns)
            writer.writeheader()
            for row in out:
                writer.writerow(row)
        return pd.DataFrame(out)
    
    def feed_raw(self, data_raw, out_csv=None):
        self.weights["inhibited"].values[:] = 0
        data = pd.DataFrame(data_raw, columns=self.nodes.index).fillna(0).values
        s = self.stepwise_generator(data)
        out = [u for u in s]
        self.values = pd.DataFrame(out)
        if not out_csv is None:
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.values.columns)
                writer.writeheader()
                for row in out:
                    writer.writerow(row)
        return pd.DataFrame(out)        
    
    def error(self, answers_vector, pos_weight=1, neg_weight=1):
        if not isinstance(answers_vector, np.ndarray):
            answers_vector = np.array(answers_vector)
        output_nodes = self.nodes.query("type == 'postsynaptic'").index
        outputs = self.values.loc[:, output_nodes].values
        categories = list(set(answers_vector))
        score = []
        for category in categories:
            answers_cat_pos = np.where(answers_vector == category, 1, 0)
            answers_cat_neg = np.where(answers_vector != category, 1, 0)
            sp = np.einsum("i,ij->j", answers_cat_pos, outputs)*pos_weight
            sn = np.einsum("i,ij->j", answers_cat_neg, outputs)*neg_weight
            score.append(((answers_cat_pos.sum()-sp)+sn/(len(categories)-1))/answers_cat_pos.sum())
        return pd.DataFrame(score, columns=output_nodes, index=categories)
            
            
    
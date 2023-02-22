import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import trange
from dataclasses import dataclass
import csv

class SpikeNetworkSim:
    def __init__(self, inputs_l = None, labels = None,dt = 1):
        
        if labels is None:
            labels = range(inputs_l)
        if inputs_l is None:
            inputs_l = len(labels)
            
        self.labels_dict = dict(zip(labels, np.arange(inputs_l)))
        self.values = pd.DataFrame(columns=np.arange(inputs_l))
        
        input_nodes = [
            {
                "type": "input",
                "priority": 0,
                "listening": None,
                "broadcasting": None
            } for _ in range(inputs_l)]
        
        self.nodes = pd.DataFrame(data=input_nodes, columns=["type", "priority", "listening", "broadcasting"])
        self.weights = pd.DataFrame(columns=["weights", "inhibited"])
        self.layers = pd.DataFrame(data=[{"layer":-1} for _ in np.arange(inputs_l)], columns=["layer"])
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
            "layer_type": []
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
            "layer_type": "default"
        }
        
        self.processors = {
            "default": self.process,
            "tempotron": self.ttron_process
        }
        
    def new_dendritic_layer(self, dendrite_connections, dendrite_weights=None, labels=None, **layer_params):
        """
        connections: описание каждого дендрита, список словарей. Каждый ключ - индекс входа из input_list, значение - задержка сигнала в дельтах
        """
        for param in layer_params.keys():
            if param in layer_params:
                self.layer_params[param].append(layer_params[param])
            else:
                raise Exception(f"Parameter {param} doesn't exist")
        for dparam in self.defaults.keys():
            if dparam not in layer_params:
                self.layer_params[dparam].append(self.defaults[dparam])
        
        first_node = self.nodes.index.size-1
        last_priority = self.nodes.priority.max()
        ltp_priority = last_priority+2  
        layer = self.layers.layer.max()+1
        
        if dendrite_weights is None:
            dendrite_weights = [{k: [np.random.randint(1, 256) for _ in c] for k, c in connection.items()} for connection in dendrite_connections]
        
        delay_map = {}
        delay_ltp_nodes = {}
        
        dendrite_inputs_map = []
        weights_map = []
        
        nnodes = []
        nweights = []
        
        for dendrite, dweights in zip(dendrite_connections, dendrite_weights):
            current_priority = last_priority+1
            dendrite_inputs = []
            weights = []
            for synaptic_via, delay in dendrite.items():
                if not synaptic_via in delay_map:
                    delay_map[synaptic_via] = {0: synaptic_via}
                if not isinstance(delay, int):
                    for _d in delay:
                        dendrite_inputs.append((synaptic_via, _d))
                    delay = max(delay)
                else:
                    dendrite_inputs.append((synaptic_via, delay))
                if delay > 0 and delay not in delay_map[synaptic_via]:
                    for d in np.arange(delay, 1, -1):
                        nnodes.append(
                            {
                                "type": "buffer", 
                                "listening": first_node+len(nnodes)+2, 
                                "broadcasting": None,
                                "priority": current_priority
                            }
                        )
                        delay_map[synaptic_via][d] = first_node+len(nnodes)
                        current_priority += 1
                    nnodes.append(
                        {
                            "type": "buffer", 
                            "listening": synaptic_via, 
                            "broadcasting": None,
                            "priority": current_priority
                        }
                    )
                    delay_map[synaptic_via][1] = first_node+len(nnodes)
            weights = [dweights[s][d] for s,d in dendrite_inputs]
            weights_map.append(weights)
            dendrite_inputs = [delay_map[s][d] for s,d in dendrite_inputs]
            dendrite_inputs_map.append(dendrite_inputs)
        for inputs in dendrite_inputs_map: 
            for i in inputs:
                if i not in delay_ltp_nodes:
                    nnodes.append(
                        {
                            "type": "ltp",
                            "listening": i,
                            "broadcasting": None,
                            "priority": ltp_priority
                        }
                    )
                    delay_ltp_nodes[i] = first_node+len(nnodes)
                
        presynaptic_range = np.arange(len(dendrite_connections))*(3)+first_node+len(nnodes)+1
        postsynaptic_range = presynaptic_range+1
        
        for dendrite, inputs, w in zip(dendrite_connections, dendrite_inputs_map, weights_map):
                
            presynaptic_priority = ltp_priority+1
            postsynaptic_priority = presynaptic_priority+1
            potentiating_priority = postsynaptic_priority+1

            presynaptic_id = first_node+len(nnodes)+1
            postsynaptic_id = presynaptic_id+1
            potentiating_id = postsynaptic_id+1
            
            
            nnodes.append(
                {
                    "type": "presynaptic", 
                    "listening": inputs, 
                    "broadcasting": postsynaptic_id, 
                    "priority": presynaptic_priority
                }
            )
            nnodes.append(
                {
                    "type": "postsynaptic", 
                    "listening": presynaptic_id, 
                    "broadcasting": presynaptic_range[presynaptic_range != presynaptic_id], 
                    "priority": postsynaptic_priority}
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
                    "type": "potentiating", 
                    "listening": [delay_ltp_nodes[n] for n in inputs], 
                    "broadcasting": presynaptic_id, 
                    "priority": potentiating_priority}
            )
            
        nlabels = np.arange(first_node+1, first_node+1+len(nnodes), dtype="object")
        if labels is not None:
            nlabels[postsynaptic_range-first_node-1] = labels
            
        self.labels_dict.update(dict(zip(np.arange(first_node+1, first_node+1+len(nnodes), dtype="object"), nlabels)))
        nlayers = [{"layer": layer} for _ in range(self.nodes.index.size, potentiating_id+1)]
        self.nodes = pd.concat((self.nodes, pd.DataFrame(nnodes))).reset_index(drop=True)
        self.weights = pd.concat((self.weights, pd.DataFrame(nweights).set_index("node", drop=True)))
        self.layers = pd.concat((self.layers, pd.DataFrame(nlayers))).reset_index(drop=True)
                          
        
    def new_layer(self, width, weights=None, labels=None, passed_inputs=None, **layer_params):
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
        layer = self.layers["layer"].max()+1
        
        if layer == 0: #первый слой
            inputs = np.array(self.nodes.query("priority==0").index.tolist())
        else: #нужно пропустить потенцирующие ноды
            inputs = np.array(self.nodes.query("priority==@priority-2").index.tolist())
        if passed_inputs is not None:
            inputs = np.concatenate((inputs, passed_inputs))
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
                    "priority": priority
                }
            )
                
        
        nlabels = np.arange(first_node+1, first_node+1+len(nnodes), dtype="object")
        if labels is not None:
            nlabels[postsynaptic_rangefirst_node-1] = labels
        self.labels_dict.update(dict(zip(np.arange(first_node+1, first_node+1+len(nnodes)), nlabels)))
        
        for w in weights:
            nnodes.append(
                {
                    "type": "presynaptic",
                    "listening": inputs,
                    "broadcasting": postsynaptic_id,
                    "priority": priority+1
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
                    "priority": priority+2
                }
            )
            nnodes.append(
                {
                    "type": "potentiating",
                    "listening": layer_ltp_range,
                    "broadcasting": presynaptic_id,
                    "priority": priority+3
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
                "priority": max_priority+1
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
        
        netmap = nodes_sorted.join(self.layers)
        net_it = list(zip(netmap.values, netmap.index))
        
        layer_params = self.layer_params
        
        status = self.weights.to_dict()
        
        for t, vals in enumerate(data):
            layer = None
            for (node_type, _, listen, cast, _layer), node in net_it:
                n_val = 0
                if node_type == "input":
                    continue
                if _layer != layer:
                    layer = _layer
                    params = {k: layer_params[k][layer] for k in layer_params.keys()}
                    params["leak"] = np.exp(-self.dt/params["tau_leak"])
                    processor = self.processors[params["layer_type"]]
                vals = processor(node_type, _, listen, cast, status, vals, vals_z, params, node, t)       
                        
            vals_z = vals
            yield dict(zip(self.nodes.index, vals))
        old_weights = self.weights.to_dict()
        old_weights.update(status)
        self.weights = pd.DataFrame(old_weights)
    
    def process(self, node_type, _, listen, cast, status, vals, vals_z, params, node, t):
        n_val = vals[node]
        match node_type:
            case "ltp":
                if vals[listen]:
                    n_val = 1
                else:
                    n_val = vals_z[node]+1
            case "buffer":
                n_val = vals_z[listen]
            case "presynaptic":
                if status["inhibited"][node] < t:
                    n_val = (vals[listen]*status["weights"][node]).sum()+vals_z[node]*params["leak"]
            case "postsynaptic":
                n_val = int(vals[listen]>params["thres"])
                if n_val:
                    status["inhibited"][listen] = t+params["tau_refractory"]
                    for b in cast:
                        if params["wta"]:
                            vals[b] = 0
                        status["inhibited"][b] = max(t+params["tau_inhibitory"], status["inhibited"][b]+params["tau_inhibitory"])
            case "potentiating":
                if vals[node-1] and params["learning"]:
                    nw = status["weights"][cast] + np.where(vals[listen]<params["tau_ltp"], params["ainc"], params["adec"])
                    nw = np.where(nw>params["wmax"], params["wmax"], nw)
                    status["weights"][cast] = np.where(nw<params["wmin"], params["wmin"], nw)
        vals[node] = n_val
        return vals
    
    def ttron_process(self, node_type, _, listen, cast, status, vals, vals_z, params, node, t):
        n_val = vals[node]
        if node_type == "ltp":
            if vals[listen]:
                n_val = 1
            else:
                n_val = vals_z[node]+1
        elif node_type == "presynaptic":
            n_val = (vals[listen]*status["weights"][node]).sum()+vals_z[node]*params["leak"]
        elif node_type == "postsynaptic":
            n_val = int((1-2*(status["inhibited"][listen] > t))*vals[listen]>params["thres"])
            if n_val != 0:
                vals[listen] = -1*params["tau_refractory"]/params["tau_ltp"]*params["thres"]
                if n_val == 1:
                    for b in cast:
                        if params["wta"]:
                            vals[b] = 0
                        status["inhibited"][b] = max(t+params["tau_inhibitory"], status["inhibited"][b]+params["tau_inhibitory"])
        elif node_type == "potentiating":
            if vals[node-1] != 0 and params["learning"]:
                dw = (params["ainc"] if vals[node-1] > 0 else params["adec"])*(1-np.exp(-params["tau_ltp"]/vals[listen]))
                nw = status["weights"][cast] + dw
                nw = np.where(nw>params["wmax"], params["wmax"], nw)
                status["weights"][cast] = np.where(nw<params["wmin"], params["wmin"], nw)
        vals[node] = n_val
        return vals
    
    def dendritic_process(self, node_type, _, listen, cast, status, vals, vals_z, params, node, t):
        n_val = vals[node]
        if node_type == "buffer":
            n_val = vals[listen]
        if node_type == "ltp":
            if vals[listen]:
                n_val = 1
            else:
                n_val = vals_z[node]+1
        elif node_type == "presynaptic":
            if status["inhibited"][node] < t:
                n_val = (vals[listen]*status["weights"][node]).sum()+vals_z[node]*params["leak"]
        elif node_type == "postsynaptic":
            n_val = int(vals[listen]>params["thres"])
            if n_val:
                status["inhibited"][listen] = t+params["tau_refractory"]
                for b in cast:
                    if params["wta"]:
                        vals[b] = 0
                    status["inhibited"][b] = max(t+params["tau_inhibitory"], status["inhibited"][b]+params["tau_inhibitory"])
        elif node_type == "potentiating":
            if vals[node-1] and params["learning"]:
                nw = status["weights"][cast] + np.where(vals[listen]<params["tau_ltp"], params["ainc"], params["adec"])
                nw = np.where(nw>params["wmax"], params["wmax"], nw)
                status["weights"][cast] = np.where(nw<params["wmin"], params["wmin"], nw)
        vals[node] = n_val
        return vals
    
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
            
            
    
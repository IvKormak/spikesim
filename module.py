import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import trange
from dataclasses import dataclass
import csv

class Network:
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
        
        self.nodes = pd.DataFrame(data=input_nodes, columns=["type", "priority", "listening", "broadcasting", "layer"])
        self.weights = pd.DataFrame(columns=["weights", "inhibited"])
        self.dt = dt
        
        self.tau_refractory = []
        self.tau_inhibitory = []
        self.tau_ltp = []
        self.leak = []
        self.thres = []
        self.ainc = []
        self.adec = []
        self.wmin = []
        self.wmax = []
        self.learning = []
        
    def new_layer(self, width, weights=None, labels=None, *, 
                    tau_inhibitory = 3, 
                    tau_refractory = 5, 
                    tau_leak = 10, 
                    tau_ltp = 5, 
                    thres = 200,
                    ainc = 30, 
                    adec = -15, 
                    wmax = 255,
                    wmin = 1,
                    learning = True
                 ):
        #print(f"{inputs_l=},{labels=},{dt=},{tau_inhibitory=},{tau_refractory=},{tau_leak=},{tau_ltp=},{thres=},{ainc=},{adec=},{wmax=},{wmin=}")
        self.tau_refractory.append(tau_refractory)
        self.tau_inhibitory.append(tau_inhibitory)
        self.tau_ltp.append(tau_ltp)
        self.leak.append(np.exp(-self.dt/tau_leak))
        self.thres.append(thres)
        self.ainc.append(ainc)
        self.adec.append(adec)
        self.wmin.append(wmin)
        self.wmax.append(wmax)
        self.learning.append(learning)
        
        nnodes = []
        nweights = []
        priority = self.nodes["priority"].max()+1
        layer = self.nodes["layer"].max()+1
        
        if layer == 0: #первый слой
            inputs = np.array(self.nodes.loc[self.nodes["priority"]==0].index.tolist())
        else: #нужно пропустить потенцирующие ноды
            inputs = np.array(self.nodes.loc[self.nodes["priority"]==priority-2].index.tolist())
        
            
        if weights is None:
            weights = np.array([np.random.randint(self.wmin, self.wmax, inputs.shape[0]) for _ in range(width)])
            
        if weights.shape != (width, inputs.shape[0]):
            raise Exception(f"Требуется массив {(width, inputs.shape[0])}, получено {weights.shape}")
            
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
            
        self.nodes = pd.concat((self.nodes, pd.DataFrame(nnodes))).reset_index(drop=True)
        self.weights = pd.concat((self.weights, pd.DataFrame(nweights).set_index("node", drop=True)))
    
    def make_recurrent(self):
        max_priority = self.nodes["priority"].max()
        last_layer_output = np.array(self.nodes.loc[self.nodes["priority"]==max_priority-1].index.tolist())
        first_layer_summators = np.array(self.nodes.loc[self.nodes["priority"]==2].index.tolist())
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
    
    def stepwise_generator(self, data):
        vals = pd.Series(np.zeros(data.shape[1]))
        for t in data.index:
            vals_z = vals
            vals = data.iloc[t].copy()
            layer = None
            for node in self.nodes.sort_values("priority").index.tolist():
                node_type, _, listen, cast, _layer = self.nodes.loc[node, :]
                if node_type == "input":
                    continue
                if _layer != layer:
                    layer = _layer
                    (
                        leak,
                        thres,
                        tau_refractory, 
                        tau_inhibitory,
                        tau_ltp,
                        ainc, 
                        adec,
                        wmin,
                        wmax,
                        learning
                    ) = (
                        self.leak[layer],
                        self.thres[layer],
                        self.tau_refractory[layer], 
                        self.tau_inhibitory[layer],
                        self.tau_ltp[layer],
                        self.ainc[layer], 
                        self.adec[layer],
                        self.wmin[layer],
                        self.wmax[layer],
                        self.learning[layer]
                    )
                    
                n_val = vals.at[node]
                if node_type == "recurrent":
                    n_val = vals.at[listen]
                if node_type == "ltp":
                    if vals.at[listen]:
                        n_val = 0
                    else:
                        n_val += self.dt
                elif node_type == "presynaptic":
                    if self.weights.at[node, "inhibited"] < t:
                        n_val = (vals[listen].values*self.weights.at[node, "weights"]).sum()+vals_z[node]*leak
                elif node_type == "postsynaptic":
                    n_val = int(vals.at[listen]>thres)
                    if n_val:
                        self.weights.at[listen, "inhibited"] = t+tau_refractory
                        for b in cast:
                            self.weights.at[b, "inhibited"] = max(t+tau_inhibitory, self.weights.at[b, "inhibited"]+tau_inhibitory)
                elif node_type == "potentiating":
                    if vals.at[node-1] and learning:
                        nw = self.weights.at[cast, "weights"] + np.where(vals[listen]>tau_ltp, ainc, adec)
                        nw = np.where(nw>wmax, wmax, nw)
                        self.weights.at[cast, "weights"] = np.where(nw<wmin, wmin, nw)

                vals.at[node] = n_val
            yield vals
                
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
    
    def feed_raw(self, data_raw, out_csv):
        self.weights["inhibited"].values[:] = 0
        data = pd.DataFrame(data_raw, columns=self.nodes.index).fillna(0)
        s = self.stepwise_generator(data)
        out = [u.to_dict() for u in s]
        self.values = pd.DataFrame(out)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.values.columns)
            writer.writeheader()
            for row in out:
                writer.writerow(row)
        return pd.DataFrame(out)
    
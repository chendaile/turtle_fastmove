import os
import numpy as np
import json


class optimized_para:
    def __init__(self, route_name, contine_bool=False):
        self.best_param_path = "output/best_params_" + route_name + ".json"
        if not contine_bool:
            with open("config/init_arg.json", "r") as f:
                config = json.load(f)
            self.params = config["params"]
        else:
            with open(self.best_param_path, "r") as f:
                config = json.load(f)
            self.params = config["best_params"]

        with open("config/init_arg.json", "r") as f:
            config = json.load(f)
        self.params_range = config["params_range"]
        self.current_paramList = np.array(list(self.params.values()))

    def save_best_params(self, lap_time):
        save_data = {"best_time": lap_time, "best_params": self.params.copy()}
        try:
            with open(self.best_param_path, "r") as f:
                old_data = json.load(f)
            if lap_time >= old_data["best_time"]:
                return False
        except:
            pass

        os.makedirs("output", exist_ok=True)
        with open(self.best_param_path, "w") as f:
            json.dump(save_data, f, indent=2)
        return True

    def update_params(self, param_updates):
        old_params = self.params.copy()

        if isinstance(param_updates, dict):
            for param_name, update in param_updates.items():
                min_val, max_val = self.params_range[param_name]
                self.params[param_name] = np.clip(update, min_val, max_val)
        elif isinstance(param_updates, np.ndarray):
            for i, param_name in enumerate(self.params.keys()):
                min_val, max_val = self.params_range[param_name]
                self.params[param_name] = np.clip(
                    param_updates[i], min_val, max_val)

        self.current_paramList = np.array(list(self.params.values()))

        return old_params

    def generate_candidate_params(self):
        current = self.current_paramList
        noise = np.random.normal(0, 0.1, size=current.shape)
        candidate = current + noise

        param_names = list(self.params.keys())
        for i, name in enumerate(param_names):
            min_val, max_val = self.params_range[name]
            candidate[i] = np.clip(candidate[i], min_val, max_val)

        return candidate

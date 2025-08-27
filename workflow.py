import json
import torch
import uuid
import copy
import os
from enum import Enum
import numpy as np
import hashlib
from torchvision import transforms
import comfy.model_management
from PIL import Image
from nodes import SaveImage
import gc
import folder_paths
from server import PromptServer
from execution import PromptExecutor


class ExecutionResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


script_list_path = os.path.join(folder_paths.user_directory, "default", "workflows")


def recursive_delete(workflow, to_delete):
    # workflow_copy = copy.deepcopy(workflow)
    new_delete = []
    for node_id in to_delete:
        for node_id2, node in workflow.items():
            # Add validation to ensure node is a dictionary and has inputs
            if not isinstance(node, dict):
                print(f"Warning: Node {node_id2} is not a dictionary, got {type(node).__name__}: {node}")
                continue
            if "inputs" not in node:
                print(f"Warning: Node {node_id2} does not have 'inputs' key")
                continue
            if not isinstance(node["inputs"], dict):
                print(f"Warning: Node {node_id2} inputs is not a dictionary, got {type(node['inputs']).__name__}: {node['inputs']}")
                continue
                
            for input_name, input_value in node["inputs"].items():
                if type(input_value) == list:
                    if len(input_value) > 0:
                        if input_value[0] == node_id:
                            new_delete.append(node_id2)
        if node_id in workflow:
            del workflow[node_id]
    if len(new_delete) > 0:
        workflow = recursive_delete(workflow, new_delete)
    return workflow


class Workflow(SaveImage):
    def __init__(self):
        self.ws = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflows": ("COMBO", {"values": []}),
                "workflow": ("STRING", {"default": ""})
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = (
        AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"),
        AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"), AnyType("*"),
    )

    FUNCTION = "generate"
    CATEGORY = "FlowChain ⛓️"

    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(s, workflows, workflow, **kwargs):
        m = hashlib.sha256()
        m.update(workflows.encode())

        # Ajouter le contenu du workflow au hash pour détecter les changements de structure
        if workflow:
            workflow_data = json.loads(workflow)

            # Extraire les nœuds de sortie avec leurs positions/types/connexions
            outputs = {}
            for k, v in workflow_data.items():
                if v.get('class_type') == 'WorkflowOutput':
                    # Capturer le nom, type et la source de données (connexions entrantes)
                    output_info = {
                        'name': v['inputs']['Name'],
                        'type': v['inputs']['type'],
                        'position': v.get('_meta', {}).get('position', [0, 0]),
                    }

                    # Ajouter les connexions d'entrée pour tracer la provenance des données
                    for input_name, input_value in v['inputs'].items():
                        if isinstance(input_value, list) and len(input_value) > 0:
                            # Stocker les IDs des nœuds connectés à cette sortie
                            output_info[input_name + '_source'] = input_value

                    outputs[k] = output_info

            # Être sûr de préserver l'ordre des sorties dans le hash
            # en les triant par position verticale
            sorted_outputs = dict(sorted(
                outputs.items(),
                key=lambda item: item[1].get('position', [0, 0])[1]
            ))

            # Ajouter l'information des sorties au hash
            m.update(json.dumps(sorted_outputs, sort_keys=True).encode())

        return m.digest().hex()

    async def generate(self, workflows, workflow, unique_id, **kwargs):
        
        # Aggressive VRAM cleanup at the very start
        def aggressive_cleanup():
            try:
                # Log memory before cleanup if available
                if torch.cuda.is_available():
                    try:
                        memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                        print(f"VRAM before cleanup: {memory_before:.2f} GB")
                    except:
                        pass
                
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                if hasattr(comfy.model_management, 'cleanup_models'):
                    comfy.model_management.cleanup_models()
                if hasattr(comfy.model_management, 'current_loaded_models'):
                    comfy.model_management.current_loaded_models.clear()
                
                # Force multiple garbage collections
                for _ in range(3):
                    gc.collect()
                
                # Clear torch cache aggressively
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Force another empty_cache after sync
                    torch.cuda.empty_cache()
                    
                    # Log memory after cleanup
                    try:
                        memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                        print(f"VRAM after cleanup: {memory_after:.2f} GB")
                    except:
                        pass
                        
            except Exception as e:
                print(f"Warning: Error during aggressive cleanup: {e}")
        
        # Initial cleanup before starting
        aggressive_cleanup()

        def populate_inputs(workflow, inputs, kwargs_values):
            workflow_inputs = {k: v for k, v in workflow.items() if  "class_type" in v and v["class_type"] == "WorkflowInput"}
            for key, value in workflow_inputs.items():
                # Add defensive check for Name field
                if "inputs" not in value or "Name" not in value["inputs"]:
                    print(f"Warning: WorkflowInput node {key} missing 'Name' field in inputs. Available inputs: {list(value.get('inputs', {}).keys())}")
                    continue
                    
                if value["inputs"]["Name"] in inputs:
                    if type(inputs[value["inputs"]["Name"]]) == list:
                        if value["inputs"]["Name"] in kwargs_values:
                            workflow[key]["inputs"]["default"] = kwargs_values[value["inputs"]["Name"]]
                    else:
                        workflow[key]["inputs"]["default"] = inputs[value["inputs"]["Name"]]

            workflow_inputs_images = {k: v for k, v in workflow.items() if "class_type" in v and
                                      v["class_type"] == "WorkflowInput" and "inputs" in v and "type" in v["inputs"] and v["inputs"]["type"] == "IMAGE"}
            for key, value in workflow_inputs_images.items():
                if "default" not in value["inputs"]:
                    workflow[key]["inputs"]["default"] = torch.tensor([])
                else:
                    if isinstance(value["inputs"]["default"], list):
                        # Si c'est une liste, on la laisse telle quelle
                        workflow[key]["inputs"]["default"] = torch.tensor([])
                    else:
                        # Si c'est un tensor, on vérifie s'il est vide
                        if value["inputs"]["default"].numel() == 0:
                            workflow[key]["inputs"]["default"] = torch.tensor([])
                    
            return workflow

        def treat_switch(workflow):
            to_delete = []
            #do_net_delete = []
            switch_to_delete = [-1]
            while len(switch_to_delete) > 0:
                switch_nodes = {k: v for k, v in workflow.items() if "class_type" in v and
                                v["class_type"].startswith("Switch") and v["class_type"].endswith("[Crystools]")}
                # order switch nodes by inputs.boolean value
                switch_to_delete = []
                switch_nodes_copy = copy.deepcopy(switch_nodes)
                for switch_id, switch_node in switch_nodes.items():
                    # create list of inputs who have switch in their inputs

                    inputs_from_switch = []
                    for node_ids, node in workflow.items():
                        # Add validation to ensure node is a dictionary and has inputs
                        if not isinstance(node, dict):
                            print(f"Warning: Node {node_ids} is not a dictionary, got {type(node).__name__}: {node}")
                            continue
                        if "inputs" not in node:
                            print(f"Warning: Node {node_ids} does not have 'inputs' key")
                            continue
                        if not isinstance(node["inputs"], dict):
                            print(f"Warning: Node {node_ids} inputs is not a dictionary, got {type(node['inputs']).__name__}: {node['inputs']}")
                            continue
                            
                        for input_name, input_value in node["inputs"].items():
                            if type(input_value) == list:
                                if len(input_value) > 0:
                                    if input_value[0] == switch_id:
                                        inputs_from_switch.append({node_ids: input_name})
                    # convert to dictionary
                    inputs_from_switch = {k: v for d in inputs_from_switch for k, v in d.items()}
                    switch = switch_nodes_copy[switch_id]
                    for node_id, input_name in inputs_from_switch.items():
                        if type(switch["inputs"]["boolean"]) == list:
                            switch_boolean_value = workflow[switch["inputs"]["boolean"][0]]["inputs"]

                            other_input_name = None
                            if "default" in switch_boolean_value:
                                other_input_name = "default"
                            elif "boolean" in switch_boolean_value:
                                other_input_name = "boolean"

                            if other_input_name is not None:
                                if switch_boolean_value[other_input_name] == True:
                                    if type(switch["inputs"]["on_true"]) == list:
                                        workflow[node_id]["inputs"][input_name] = switch["inputs"]["on_true"]
                                        if node_id in switch_nodes_copy:
                                            switch_nodes_copy[node_id]["inputs"][input_name] = switch["inputs"]["on_true"]
                                    else:
                                        to_delete.append(node_id)
                                else:
                                    if type(switch["inputs"]["on_false"]) == list:
                                        workflow[node_id]["inputs"][input_name] = switch["inputs"]["on_false"]
                                        if node_id in switch_nodes_copy:
                                            switch_nodes_copy[node_id]["inputs"][input_name] = switch["inputs"]["on_false"]
                                    else:
                                        to_delete.append(node_id)
                                switch_to_delete.append(switch_id)
                        else:
                            if switch["inputs"]["boolean"] == True:
                                if type(switch["inputs"]["on_true"]) == list:
                                    workflow[node_id]["inputs"][input_name] = switch["inputs"]["on_true"]
                                    if node_id in switch_nodes_copy:
                                        switch_nodes_copy[node_id]["inputs"][input_name] = switch["inputs"]["on_true"]
                                else:
                                    to_delete.append(node_id)
                            else:
                                if type(switch["inputs"]["on_false"]) == list:
                                    workflow[node_id]["inputs"][input_name] = switch["inputs"]["on_false"]
                                    if node_id in switch_nodes_copy:
                                        switch_nodes_copy[node_id]["inputs"][input_name] = switch["inputs"]["on_false"]
                                else:
                                    to_delete.append(node_id)
                            switch_to_delete.append(switch_id)
                print(switch_to_delete)
                workflow = {k: v for k, v in workflow.items() if
                            not ("class_type" in v and v["class_type"].startswith("Switch") and v["class_type"].endswith(
                                "[Crystools]") and k in switch_to_delete)}

            return workflow, to_delete

        def treat_continue(workflow):
            to_delete = []
            continue_nodes = {k: v for k, v in workflow.items() if "class_type" in v and
                              v["class_type"].startswith("WorkflowContinue")}
            do_net_delete = []
            for continue_node_id, continue_node in continue_nodes.items():
                for node_id, node in workflow.items():
                    # Add validation to ensure node is a dictionary and has inputs
                    if not isinstance(node, dict):
                        print(f"Warning: Node {node_id} is not a dictionary, got {type(node).__name__}: {node}")
                        continue
                    if "inputs" not in node:
                        print(f"Warning: Node {node_id} does not have 'inputs' key")
                        continue
                    if not isinstance(node["inputs"], dict):
                        print(f"Warning: Node {node_id} inputs is not a dictionary, got {type(node['inputs']).__name__}: {node['inputs']}")
                        continue
                        
                    for input_name, input_value in node["inputs"].items():
                        if type(input_value) == list:
                            if len(input_value) > 0:
                                if input_value[0] == continue_node_id:
                                    if type(continue_node["inputs"]["continue_workflow"]) == list:
                                        input_other_node = \
                                            workflow[continue_node["inputs"]["continue_workflow"][0]][
                                                "inputs"]
                                        other_input_name = None
                                        if "default" in input_other_node:
                                            other_input_name = "default"
                                        elif "boolean" in input_other_node:
                                            other_input_name = "boolean"

                                        if other_input_name is not None:
                                            if input_other_node[other_input_name]:
                                                workflow[node_id]["inputs"][input_name] = continue_node["inputs"]["input"]
                                            else:
                                                to_delete.append(node_id)
                                        else:
                                            do_net_delete.append(continue_node_id)
                                    else:
                                        if continue_node["inputs"]["continue_workflow"]:
                                            workflow[node_id]["inputs"][input_name] = continue_node["inputs"]["input"]
                                        else:
                                            to_delete.append(node_id)

            workflow = {k: v for k, v in workflow.items() if
                                    not ("class_type" in v and v["class_type"].startswith("WorkflowContinue") and k not in do_net_delete)}
            return workflow, to_delete

        def redefine_id(subworkflow, max_id):
            new_sub_workflow = {}

            for k, v in subworkflow.items():
                max_id += 1
                new_sub_workflow[str(max_id)] = v
                # replace old id by new id items in inputs of workflow
                for node_id, node in subworkflow.items():
                    # Add validation to ensure node is a dictionary and has inputs
                    if not isinstance(node, dict):
                        print(f"Warning: Node {node_id} is not a dictionary, got {type(node).__name__}: {node}")
                        continue
                    if "inputs" not in node:
                        print(f"Warning: Node {node_id} does not have 'inputs' key")
                        continue
                    if not isinstance(node["inputs"], dict):
                        print(f"Warning: Node {node_id} inputs is not a dictionary, got {type(node['inputs']).__name__}: {node['inputs']}")
                        continue
                        
                    for input_name, input_value in node["inputs"].items():
                        if type(input_value) == list:
                            if len(input_value) > 0:
                                if input_value[0] == k:
                                    subworkflow[node_id]["inputs"][input_name][0] = str(max_id)
                for node_id, node in new_sub_workflow.items():
                    # Add validation to ensure node is a dictionary and has inputs
                    if not isinstance(node, dict):
                        print(f"Warning: New workflow node {node_id} is not a dictionary, got {type(node).__name__}: {node}")
                        continue
                    if "inputs" not in node:
                        print(f"Warning: New workflow node {node_id} does not have 'inputs' key")
                        continue
                    if not isinstance(node["inputs"], dict):
                        print(f"Warning: New workflow node {node_id} inputs is not a dictionary, got {type(node['inputs']).__name__}: {node['inputs']}")
                        continue
                        
                    for input_name, input_value in node["inputs"].items():
                        if type(input_value) == list:
                            if len(input_value) > 0:
                                if input_value[0] == k:
                                    new_sub_workflow[node_id]["inputs"][input_name][0] = str(max_id)
            return new_sub_workflow, max_id

        def change_subnode(subworkflow, node_id_to_find, value):
            for node_id, node in subworkflow.items():
                # Add validation to ensure node is a dictionary and has inputs
                if not isinstance(node, dict):
                    print(f"Warning: Node {node_id} is not a dictionary, got {type(node).__name__}: {node}")
                    continue
                if "inputs" not in node:
                    print(f"Warning: Node {node_id} does not have 'inputs' key")
                    continue
                if not isinstance(node["inputs"], dict):
                    print(f"Warning: Node {node_id} inputs is not a dictionary, got {type(node['inputs']).__name__}: {node['inputs']}")
                    continue
                    
                for input_name, input_value in node["inputs"].items():
                    if type(input_value) == list:
                        if len(input_value) > 0:
                            if input_value[0] == node_id_to_find:
                                subworkflow[node_id]["inputs"][input_name] = value

            return subworkflow

        def merge_inputs_outputs(workflow, workflow_name, subworkflow, workflow_outputs):
            # get max workflow id
            # coinvert workflow_outputs to list
            workflow_outputs = list(workflow_outputs.values())
            # prendre le premier workflow
            workflow_node = [{"id":k, **v} for k, v in workflow.items() if "class_type" in v and v["class_type"] == "Workflow" and v["inputs"]["workflows"] == workflow_name][0]
            sub_input_nodes = {k: v for k, v in subworkflow.items() if "class_type" in v and v["class_type"] == "WorkflowInput"}
            do_not_delete = []
            for sub_id, sub_node in sub_input_nodes.items():
                # Add defensive check for Name field
                if "inputs" not in sub_node or "Name" not in sub_node["inputs"]:
                    print(f"Warning: WorkflowInput sub-node {sub_id} missing 'Name' field. Skipping.")
                    continue
                    
                if sub_node["inputs"]["Name"] in workflow_node["inputs"]:
                    value = workflow_node["inputs"][sub_node["inputs"]["Name"]]
                    if type(value) == list:
                        subworkflow = change_subnode(subworkflow, sub_id, value)
                    else:
                        subworkflow[sub_id]["inputs"]["default"] = value
                        do_not_delete.append(sub_id)

            # remove input node
            subworkflow = {k: v for k, v in subworkflow.items() if not ("class_type" in v and v["class_type"] == "WorkflowInput" and k not in do_not_delete)}

            # get sub workflow file path
            sub_workflow_file_path = os.path.join(folder_paths.user_directory, "default", "workflows", workflow_name)
            sub_original_positions = {}
            
            if os.path.exists(sub_workflow_file_path):
                try:
                    with open(sub_workflow_file_path, "r", encoding="utf-8") as f:
                        sub_original_workflow = json.load(f)
                        
                    if "nodes" in sub_original_workflow:
                        for node in sub_original_workflow["nodes"]:
                            if node.get("type") == "WorkflowOutput":
                                node_id = str(node.get("id", "unknown"))
                                pos_y = node.get("pos", [0, 0])[1]
                                w_values = node.get("widgets_values", "")
                                if "Name" in w_values:
                                    node_name = w_values["Name"]["value"]
                                else:
                                    node_name = w_values[0]
                                sub_original_positions[node_name] = pos_y
                except Exception as e:
                    print(f"Error reading sub-workflow file: {str(e)}")

            sub_output_nodes = {k: v for k, v in subworkflow.items() if "class_type" in v and v["class_type"] == "WorkflowOutput"}
            
            # sort sub workflow output nodes
            sub_outputs_with_position = []
            for k, v in sub_output_nodes.items():
                # Add defensive check for Name field
                if "inputs" not in v or "Name" not in v["inputs"]:
                    print(f"Warning: WorkflowOutput sub-node {k} missing 'Name' field. Using default name.")
                    output_name = f"output_{k}"
                else:
                    output_name = v["inputs"]["Name"]
                y_position = sub_original_positions.get(output_name, 999999)
                sub_outputs_with_position.append((k, y_position))

            sub_outputs_with_position.sort(key=lambda x: x[1])
            sub_output_nodes = {k: sub_output_nodes[k] for k, _ in sub_outputs_with_position}

                
            workflow_copy = copy.deepcopy(workflow)
            for node_id, node in workflow_copy.items():
                # Add validation to ensure node is a dictionary and has inputs
                if not isinstance(node, dict):
                    print(f"Warning: Node {node_id} is not a dictionary, got {type(node).__name__}: {node}")
                    continue
                if "inputs" not in node:
                    print(f"Warning: Node {node_id} does not have 'inputs' key")
                    continue
                if not isinstance(node["inputs"], dict):
                    print(f"Warning: Node {node_id} inputs is not a dictionary, got {type(node['inputs']).__name__}: {node['inputs']}")
                    continue
                    
                for input_name, input_value in node["inputs"].items():
                    if type(input_value) == list:
                        if len(input_value) > 0:
                            if input_value[0] == workflow_node["id"]:
                                for sub_output_id, sub_output_node in sub_output_nodes.items():
                                    # Add defensive checks for Name fields
                                    if ("inputs" in sub_output_node and "Name" in sub_output_node["inputs"] and
                                        input_value[1] < len(workflow_outputs) and
                                        "inputs" in workflow_outputs[input_value[1]] and 
                                        "Name" in workflow_outputs[input_value[1]]["inputs"]):
                                        
                                        if sub_output_node["inputs"]["Name"] == workflow_outputs[input_value[1]]["inputs"]["Name"]:
                                            workflow[node_id]["inputs"][input_name] = sub_output_node["inputs"]["default"]

            # remove output node
            subworkflow = {k: v for k, v in subworkflow.items() if not ("class_type" in v and v["class_type"] == "WorkflowOutput")}

            return workflow, subworkflow

        def clean_workflow(workflow, inputs=None, kwargs_values=None):
            if kwargs_values is None:
                kwargs_values = {}
            if inputs is None:
                inputs = {}
            if inputs is not None:
                workflow = populate_inputs(workflow, inputs, kwargs_values)

            # Filter out unknown node types that ComfyUI doesn't recognize
            import nodes
            valid_workflow = {}
            filtered_nodes = []
            
            for node_id, node in workflow.items():
                if not isinstance(node, dict) or "class_type" not in node:
                    print(f"Warning: Skipping invalid node {node_id}")
                    filtered_nodes.append(node_id)
                    continue
                    
                class_type = node["class_type"]
                
                # Check if this is a known node type
                if class_type in nodes.NODE_CLASS_MAPPINGS:
                    valid_workflow[node_id] = node
                else:
                    print(f"Warning: Filtering out unknown node type '{class_type}' (node {node_id})")
                    filtered_nodes.append(node_id)
            
            # Update workflow to only include valid nodes
            workflow = valid_workflow
            
            # Remove connections to filtered nodes
            for node_id, node in workflow.items():
                if "inputs" in node and isinstance(node["inputs"], dict):
                    inputs_to_update = {}
                    for input_name, input_value in node["inputs"].items():
                        if isinstance(input_value, list) and len(input_value) > 0:
                            if input_value[0] in filtered_nodes:
                                print(f"Warning: Removing connection from filtered node {input_value[0]} to {node_id}.{input_name}")
                                # Don't include this input
                                continue
                        inputs_to_update[input_name] = input_value
                    node["inputs"] = inputs_to_update

            workflow_outputs = {k: v for k, v in workflow.items() if "class_type" in v and v["class_type"] == "WorkflowOutput"}

            for output_id, output_node in workflow_outputs.items():
                workflow[output_id]["inputs"]["ui"] = False

            workflow, switch_to_delete = treat_switch(workflow)
            workflow, continue_to_delete = treat_continue(workflow)
            workflow = recursive_delete(workflow, switch_to_delete + continue_to_delete)
            return workflow, workflow_outputs

        def convert_comfyui_workflow_to_internal_format(workflow_data):
            """Convert ComfyUI export format to internal workflow format"""
            if not isinstance(workflow_data, dict):
                return workflow_data
                
            # Check if this is a ComfyUI export format (has 'nodes' array)
            if "nodes" not in workflow_data or not isinstance(workflow_data["nodes"], list):
                # Check if it's already in internal format (all keys should be node IDs)
                # Internal format has node IDs as keys and node data as values
                is_internal_format = True
                for key, value in workflow_data.items():
                    if not isinstance(value, dict) or "class_type" not in value:
                        is_internal_format = False
                        break
                
                if is_internal_format:
                    return workflow_data
                else:
                    # Unknown format, try to handle gracefully
                    print(f"Warning: Unknown workflow format. Root keys: {list(workflow_data.keys())}")
                    return {}
                
            print(f"Converting ComfyUI export format with {len(workflow_data['nodes'])} nodes")
            internal_workflow = {}
            
            # First pass: create a mapping of link_id -> (source_node_id, source_slot)
            link_mapping = {}
            if "links" in workflow_data and isinstance(workflow_data["links"], list):
                for link in workflow_data["links"]:
                    if isinstance(link, list) and len(link) >= 6:
                        # ComfyUI link format: [link_id, source_node_id, source_slot, target_node_id, target_slot, type]
                        link_id = link[0]
                        source_node_id = str(link[1])
                        source_slot = link[2]
                        link_mapping[link_id] = (source_node_id, source_slot)
            
            for node in workflow_data["nodes"]:
                if not isinstance(node, dict) or "id" not in node:
                    print(f"Skipping invalid node: {node}")
                    continue
                
                node_type = node.get("type", "")
                
                # Skip note nodes and other non-executable node types
                if node_type in ["Note", "Reroute", "PrimitiveNode"] or node_type.startswith("workflow/"):
                    print(f"Skipping non-executable node type: {node_type}")
                    continue
                    
                node_id = str(node["id"])
                
                # Convert node structure to expected internal format
                converted_node = {
                    "class_type": node.get("type", ""),
                    "inputs": {}
                }
                
                # Copy over any properties from the node to inputs
                if "properties" in node and isinstance(node["properties"], dict):
                    converted_node["inputs"].update(node["properties"])
                    
                # Handle widgets_values (node parameters)
                if "widgets_values" in node:
                    if isinstance(node["widgets_values"], dict):
                        # For dict format, preserve the structure
                        converted_node["inputs"].update(node["widgets_values"])
                    elif isinstance(node["widgets_values"], list):
                        # For list format, map to parameter names using the inputs definition
                        widgets = node["widgets_values"]
                        
                        # Handle specific FlowChain node types first
                        if node_type == "WorkflowInput" and len(widgets) >= 1:
                            converted_node["inputs"]["Name"] = widgets[0] if len(widgets) > 0 else ""
                            if len(widgets) > 1:
                                converted_node["inputs"]["default"] = widgets[1]
                        elif node_type == "WorkflowOutput" and len(widgets) >= 1:
                            converted_node["inputs"]["Name"] = widgets[0] if len(widgets) > 0 else ""
                            if "outputs" in node and isinstance(node["outputs"], list):
                                for output in node["outputs"]:
                                    if isinstance(output, dict) and "type" in output:
                                        converted_node["inputs"]["type"] = output["type"]
                                        break
                            else:
                                converted_node["inputs"]["type"] = "IMAGE"
                        elif node_type == "Workflow" and len(widgets) >= 2:
                            converted_node["inputs"]["workflows"] = widgets[0] if len(widgets) > 0 else ""
                            converted_node["inputs"]["workflow"] = widgets[1] if len(widgets) > 1 else ""
                        elif node_type == "WanVideoSampler":
                            # Special handling for WanVideoSampler due to widget mapping complexity
                            # Based on analysis of vidFlow.json widgets_values: [4, 1.0, 8.0, 1057359483639419, 'increment', False, 'lcm', 0, 1, '', 'comfy', 0, -1, False]
                            if len(widgets) >= 14:
                                converted_node["inputs"]["steps"] = widgets[0]  # 4
                                converted_node["inputs"]["cfg"] = widgets[1]    # 1.0
                                converted_node["inputs"]["shift"] = widgets[2]  # 8.0
                                converted_node["inputs"]["seed"] = widgets[3]   # 1057359483639419
                                # Skip widgets[4] = 'increment' - this seems to be misplaced or extra
                                converted_node["inputs"]["force_offload"] = widgets[5]  # False
                                converted_node["inputs"]["scheduler"] = widgets[6]      # 'lcm' - this is the correct scheduler
                                converted_node["inputs"]["riflex_freq_index"] = widgets[7]  # 0
                                converted_node["inputs"]["denoise_strength"] = widgets[8]   # 1
                                converted_node["inputs"]["batched_cfg"] = widgets[9]        # ''
                                converted_node["inputs"]["rope_function"] = widgets[10]     # 'comfy'
                                converted_node["inputs"]["start_step"] = widgets[11]        # 0
                                converted_node["inputs"]["end_step"] = widgets[12]          # -1
                                converted_node["inputs"]["add_noise_to_samples"] = widgets[13]  # False
                            else:
                                # Fallback to original mapping if widgets length is unexpected
                                converted_node["widgets_values"] = widgets
                        else:
                            # For all other nodes, map widgets to their corresponding widget-enabled inputs
                            # This mimics ComfyUI's internal widget mapping behavior
                            if "inputs" in node and isinstance(node["inputs"], list):
                                # Find inputs that have widgets (these get values from widgets_values)
                                widget_inputs = []
                                for input_def in node["inputs"]:
                                    if isinstance(input_def, dict) and "widget" in input_def:
                                        # This input expects a widget value
                                        widget_inputs.append(input_def["name"])
                                
                                # Map widgets_values to widget-enabled inputs in order
                                for i, widget_value in enumerate(widgets):
                                    if i < len(widget_inputs):
                                        input_name = widget_inputs[i]
                                        converted_node["inputs"][input_name] = widget_value
                                    else:
                                        # If we have more widgets than expected widget inputs, store with generic names
                                        converted_node["inputs"][f"widget_{i}"] = widget_value
                            else:
                                # Fallback: preserve widgets_values if we can't determine the mapping
                                converted_node["widgets_values"] = widgets
                
                # Handle input connections from other nodes using proper link resolution
                if "inputs" in node and isinstance(node["inputs"], list):
                    for input_conn in node["inputs"]:
                        if isinstance(input_conn, dict):
                            input_name = input_conn.get("name", "")
                            link_id = input_conn.get("link", None)
                            
                            if link_id is not None and link_id in link_mapping:
                                source_node_id, source_slot = link_mapping[link_id]
                                # Create proper connection reference [source_node_id, source_slot]
                                # Only set this if we haven't already set this input from widgets
                                if input_name not in converted_node["inputs"]:
                                    # Special handling for problematic connections that cause tensor/boolean issues
                                    if (node_type == "WanVideoImageToVideoEncode" and input_name == "end_image"):
                                        # For end_image connections, add validation metadata
                                        print(f"Warning: Setting up end_image connection from node {source_node_id} to {node_type}")
                                        print(f"  This connection may cause tensor/boolean evaluation errors")
                                        # Still create the connection but add a flag for special handling
                                        converted_node["inputs"][input_name] = [source_node_id, source_slot]
                                        # Add metadata to help with debugging
                                        if "_connection_metadata" not in converted_node:
                                            converted_node["_connection_metadata"] = {}
                                        converted_node["_connection_metadata"][input_name] = {
                                            "source_node": source_node_id,
                                            "source_slot": source_slot,
                                            "validation_needed": True,
                                            "known_issue": "tensor_boolean_ambiguity"
                                        }
                                    else:
                                        converted_node["inputs"][input_name] = [source_node_id, source_slot]
                            elif input_name:
                                # If there's no link and no widget value already set, 
                                # this input should remain unconnected (None)
                                if input_name not in converted_node["inputs"]:
                                    pass  # Leave it as None/unset
                
                internal_workflow[node_id] = converted_node
                
                # Validate that inputs don't contain unexpected tensor data
                if "inputs" in converted_node:
                    inputs_to_remove = []
                    for input_name, input_value in converted_node["inputs"].items():
                        if hasattr(input_value, 'shape'):  # Check if it's a tensor
                            print(f"Warning: Node {node_id} ({node_type}) input '{input_name}' contains tensor data, this may cause issues")
                            print(f"  Tensor shape: {input_value.shape}, removing to prevent errors")
                            inputs_to_remove.append(input_name)
                    
                    # Remove tensor inputs to prevent runtime errors
                    for input_name in inputs_to_remove:
                        del converted_node["inputs"][input_name]
                    
                    # Additional validation for known problematic combinations
                    if node_type == "WanVideoImageToVideoEncode" and "end_image" in converted_node["inputs"]:
                        end_image_value = converted_node["inputs"]["end_image"]
                        if isinstance(end_image_value, list) and len(end_image_value) == 2:
                            # This is a connection reference, check if the source is a WorkflowInput
                            source_node_id = str(end_image_value[0])
                            print(f"Warning: end_image connected from node {source_node_id}")
                            print(f"  This may cause 'Boolean value of Tensor' errors in WanVideoWrapper")
                            print(f"  Removing this connection to prevent runtime error")
                            print(f"  Note: end_image will be None instead of connected image data")
                            
                            # Remove the problematic connection to prevent the runtime error
                            # This means end_image will be None, which should work fine for optional parameters
                            del converted_node["inputs"]["end_image"]
                            
                            if "_validation_warnings" not in converted_node:
                                converted_node["_validation_warnings"] = []
                            converted_node["_validation_warnings"].append({
                                "input": "end_image",
                                "issue": "connection_removed_to_prevent_tensor_boolean_error",
                                "source_node": source_node_id,
                                "action": "removed_connection",
                                "description": "Removed end_image connection to prevent 'Boolean value of Tensor' error in WanVideoWrapper"
                            })
                
            print(f"Converted to internal format with {len(internal_workflow)} nodes")
            return internal_workflow

        def get_recursive_workflow(workflow_name, workflows, max_id=0):
            # if workflows[-5:] == ".json":
            #    workflow = get_workflow(workflows)
            # else:
            try:
                if workflows == "{}":
                    raise ValueError("Empty workflow.")
                
                # If workflows is empty or None, try to load from file
                if not workflows or workflows.strip() == "":
                    workflow_file_path = os.path.join(folder_paths.user_directory, "default", "workflows", workflow_name)
                    if os.path.exists(workflow_file_path):
                        with open(workflow_file_path, "r", encoding="utf-8") as f:
                            workflows = f.read()
                    else:
                        raise FileNotFoundError(f"Workflow file not found: {workflow_file_path}")
                
                loaded_data = json.loads(workflows)
                
                # Convert ComfyUI export format to internal format if needed
                workflow = convert_comfyui_workflow_to_internal_format(loaded_data)
                
                # Validate that the processed workflow has the expected structure
                if not isinstance(workflow, dict):
                    raise ValueError(f"Invalid workflow structure after processing: expected dictionary, got {type(workflow).__name__}")
                
                # Validate that all nodes have the expected structure
                for node_id, node in workflow.items():
                    if not isinstance(node, dict):
                        raise ValueError(f"Invalid node structure at '{node_id}': expected dictionary, got {type(node).__name__}")
                    if "inputs" in node and not isinstance(node["inputs"], dict):
                        raise ValueError(f"Invalid inputs structure at node '{node_id}': expected dictionary, got {type(node['inputs']).__name__}")
                    
            except Exception as e:
                print(f"JSON parsing error for workflow '{workflow_name}':")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Workflow content length: {len(workflows) if workflows else 0}")
                print(f"First 200 chars: {workflows[:200] if workflows else 'None'}")
                print(f"Last 200 chars: {workflows[-200:] if workflows and len(workflows) > 200 else 'N/A'}")
                
                # Try to get more specific information about the structure
                try:
                    test_parse = json.loads(workflows)
                    print(f"JSON parsed successfully. Root keys: {list(test_parse.keys()) if isinstance(test_parse, dict) else 'Not a dict'}")
                    if isinstance(test_parse, dict) and "nodes" in test_parse:
                        print(f"Found 'nodes' key with {len(test_parse['nodes'])} items")
                        if test_parse["nodes"] and isinstance(test_parse["nodes"], list):
                            first_node = test_parse["nodes"][0]
                            print(f"First node type: {type(first_node).__name__}, keys: {list(first_node.keys()) if isinstance(first_node, dict) else 'Not a dict'}")
                except Exception as parse_e:
                    print(f"Could not re-parse JSON: {str(parse_e)}")
                
                raise RuntimeError(f"Error while loading workflow: {workflow_name}. JSON parsing failed: {str(e)}. See <a href='https://github.com/numz/Comfyui-FlowChain'> for more information.")

            workflow, max_id = redefine_id(workflow, max_id)
            sub_workflows = {k: v for k, v in workflow.items() if "class_type" in v and v["class_type"] == "Workflow"}
            for key, sub_workflow_node in sub_workflows.items():
                # Cleanup before processing each sub-workflow
                aggressive_cleanup()
                
                workflow_json = sub_workflow_node["inputs"]["workflow"]
                sub_workflow_name = sub_workflow_node["inputs"]["workflows"]
                subworkflow, max_id = get_recursive_workflow(sub_workflow_name, workflow_json, max_id)

                # Cleanup after processing each sub-workflow
                aggressive_cleanup()

                # get sub workflow file path
                sub_workflow_file_path = os.path.join(folder_paths.user_directory, "default", "workflows", sub_workflow_name)
                sub_original_positions = {}
                
                if os.path.exists(sub_workflow_file_path):
                    try:
                        with open(sub_workflow_file_path, "r", encoding="utf-8") as f:
                            sub_original_workflow = json.load(f)
                            
                        if "nodes" in sub_original_workflow:
                            for node in sub_original_workflow["nodes"]:
                                if node.get("type") == "WorkflowOutput":
                                    node_id = str(node.get("id", "unknown"))
                                    pos_y = node.get("pos", [0, 0])[1]
                                    w_values = node.get("widgets_values", "")
                                    if "Name" in w_values:
                                        node_name = w_values["Name"]["value"]
                                    else:
                                        node_name = w_values[0]
                                    sub_original_positions[node_name] = pos_y
                    except Exception as e:
                        print(f"Error reading sub-workflow file: {str(e)}")

                workflow_outputs_sub = {k: v for k, v in subworkflow.items() if "class_type" in v and v["class_type"] == "WorkflowOutput"}
                
                # sort sub workflow output nodes
                sub_outputs_with_position = []
                for k, v in workflow_outputs_sub.items():
                    # Add defensive check for Name field
                    if "inputs" not in v or "Name" not in v["inputs"]:
                        print(f"Warning: WorkflowOutput sub-node {k} missing 'Name' field. Using default name.")
                        output_name = f"output_{k}"
                    else:
                        output_name = v["inputs"]["Name"]
                    y_position = sub_original_positions.get(output_name, 999999)
                    sub_outputs_with_position.append((k, y_position))

                sub_outputs_with_position.sort(key=lambda x: x[1])
                workflow_outputs_sub = {k: workflow_outputs_sub[k] for k, _ in sub_outputs_with_position}

                workflow, subworkflow = merge_inputs_outputs(workflow, sub_workflow_name, subworkflow, workflow_outputs_sub)
                workflow = {k: v for k, v in workflow.items() if k != key}
                # add subworkflow to workflow
                workflow.update(subworkflow)
            return workflow, max_id
        
        server_instance = PromptServer.instance
        client_id = server_instance.client_id
        if server_instance and hasattr(server_instance, 'prompt_queue'):
            current_queue = server_instance.prompt_queue.get_current_queue()
            queue_info = {
                'queue_running': current_queue[0],
                'queue_pending': current_queue[1]
            }
            
            # Now you can access the original inputs as before
            queue_to_use = queue_info["queue_running"]
            original_inputs = [v["inputs"] for k, v in queue_to_use[0][2].items() if k == unique_id][0]

        else:
            # Fallback to empty inputs if server instance not available
            original_inputs = {}
        
        # Clear VRAM at the start to ensure clean execution environment
        aggressive_cleanup()
        
        workflow, _ = get_recursive_workflow(workflows, workflow, 5000)
        print(f"DEBUG: Calling get_recursive_workflow with workflow_name='{workflows}', workflow_content_length={len(workflow) if workflow else 0}")
        
        # Aggressive cleanup after workflow processing but before execution
        aggressive_cleanup()
        
        workflow, workflow_outputs = clean_workflow(workflow, original_inputs, kwargs)
        
        # Accéder au fichier JSON original pour obtenir les positions correctes
        workflow_file_path = os.path.join(folder_paths.user_directory, "default", "workflows", workflows)
        original_positions = {}
        
        # Récupérer les positions des noeuds de sortie depuis le fichier original
        
        if os.path.exists(workflow_file_path):
            try:
                with open(workflow_file_path, "r", encoding="utf-8") as f:
                    original_workflow = json.load(f)
                    
                # Créer un mapping node_id -> position pour les noeuds WorkflowOutput
                if "nodes" in original_workflow:
                    for node in original_workflow["nodes"]:
                        if node.get("type") == "WorkflowOutput":
                            # node_id = str(node.get("id", "unknown"))
                            pos_y = node.get("pos", [0, 0])[1]
                            
                            w_values = node.get("widgets_values", "")
                            if "Name" in w_values:
                                node_name = w_values["Name"]["value"]
                            else:
                                node_name = w_values[0]

                            original_positions[node_name] = pos_y
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier workflow original: {str(e)}")
        
        # Récupérer les nœuds de sortie et les trier par position Y
        workflow_outputs_with_position = []
        for k, v in workflow_outputs.items():
            # Add defensive check for Name field
            if "inputs" not in v or "Name" not in v["inputs"]:
                print(f"Warning: WorkflowOutput node {k} missing 'Name' field. Using default name.")
                output_name = f"output_{k}"
            else:
                output_name = v["inputs"]["Name"]
            # Utiliser la position du fichier original si disponible, sinon utiliser une position par défaut
            y_position = original_positions.get(output_name, 999999)
            workflow_outputs_with_position.append((k, y_position))
        
        # Trier par position Y croissante
        workflow_outputs_with_position.sort(key=lambda x: x[1])
        
        # Extraire seulement les IDs dans l'ordre trié
        workflow_outputs_id = [k for k, _ in workflow_outputs_with_position]

        prompt_id = str(uuid.uuid4())

        class SimpleServer:
            def __init__(self):
                self.client_id = client_id
                self.last_node_id = None
                self.last_prompt_id = prompt_id

            def send_sync(self, *args, **kwargs):
                pass  # No-op implementation
            
        simple_server = SimpleServer()
        executor = PromptExecutor(simple_server)
        
        try:
            # Aggressive VRAM cleanup before execution
            aggressive_cleanup()
            
            # Validate workflow before execution
            if not workflow or not isinstance(workflow, dict):
                raise ValueError(f"Invalid workflow structure for execution: {type(workflow)}")
            
            # Check that we have at least some valid nodes
            if len(workflow) == 0:
                raise ValueError("No valid nodes found in workflow after cleaning")
            
            print(f"Executing workflow with {len(workflow)} nodes")
            
            await executor.execute_async(workflow, prompt_id, {"client_id": client_id}, workflow_outputs_id)

            history_result = executor.history_result
            
            # Immediate cleanup after execution completes
            aggressive_cleanup()
            
        except Exception as e:
            # Ensure aggressive cleanup even if execution fails
            aggressive_cleanup()
            print(f"Error during workflow execution: {type(e).__name__}: {str(e)}")
            raise e
        finally:
            # Final comprehensive cleanup
            aggressive_cleanup()
            
            # Additional cleanup for any remaining references
            try:
                # Clear any potential remaining model references
                if hasattr(comfy.model_management, 'models'):
                    comfy.model_management.models.clear()
                if hasattr(comfy.model_management, 'model_cache'):
                    comfy.model_management.model_cache.clear()
            except Exception:
                pass

        # Remplacer la boucle de génération d'output qui ne respecte pas l'ordre
        output = []
        for id_node in workflow_outputs_id:  # Utiliser l'ordre trié des IDs
            if id_node in history_result["outputs"]:
                result_value = history_result["outputs"][id_node]["default"]
                # Apply formatting based on the expected output type
                output.append(result_value[0])
            else:
                node = workflow_outputs[id_node]  # Récupérer le nœud correspondant à l'ID
                if node["inputs"]["type"] == "IMAGE" or node["inputs"]["type"] == "MASK":
                    black_image_np = np.zeros((255, 255, 3), dtype=np.uint8)
                    black_image_pil = Image.fromarray(black_image_np)
                    transform = transforms.ToTensor()
                    image_tensor = transform(black_image_pil)
                    image_tensor = image_tensor.permute(1, 2, 0)
                    image_tensor = image_tensor.unsqueeze(0)
                    output.append(image_tensor)
                else:
                    output.append(None)

        # Final cleanup before returning results
        try:
            del history_result
            del executor
            del workflow
            del workflow_outputs
        except:
            pass
        
        # One final aggressive cleanup before return
        aggressive_cleanup()

        return tuple(output)
        # return tuple(queue[uid]["outputs"])


NODE_CLASS_MAPPINGS_WORKFLOW = {
    "Workflow": Workflow,
}

NODE_DISPLAY_NAME_MAPPINGS_WORKFLOW = {
    "Workflow": "Workflow (FlowChain ⛓️)",
}

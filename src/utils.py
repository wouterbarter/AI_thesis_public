import pandas as pd

from tqdm.auto import tqdm
import math
import hashlib
from typing import List, Dict, Any

import torch
import torch.nn.functional as F

import ipywidgets as widgets
from IPython.display import display, clear_output

from src.prompt_manager import PreparedPrompt
from .modeler import Modeler, ModelOutput # Use relative import to access the Modeler class
import itertools
from collections.abc import Iterator



def generate_content_id(row, columns_to_hash):
    """Creates a unique SHA-256 hash from a row's content."""
    # Combine the content of the specified columns into a single string
    combined_string = "".join(str(row[col]) for col in columns_to_hash)

    # Create a hash and return the first 16 characters for a manageable ID
    return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()[:16]


def generate_in_batches(modeler: Modeler, 
                        conversations: list[PreparedPrompt], 
                        batch_size: int, 
                        **kwargs) -> list[ModelOutput]:
    """
    A wrapper function that runs generation in batches and returns processed output objects.
    """
    model_output_list = []

    n_batches = math.ceil(len(conversations) / batch_size)
    print(f"Processing {len(conversations)} prompts in {n_batches} batches of size {batch_size}...")

    for i in tqdm(range(0, len(conversations), batch_size), desc="Generating Batches"):
        batch_prompts = conversations[i:i + batch_size]

        # Returns: list[ModelOutput]
        batch_results = modeler.generate_chat(batch_prompts, **kwargs)
        model_output_list.extend(batch_results)

    return model_output_list


def chunked_generator(iterable, size):
    """Helper to slice a generator into batches."""
    iterator = iter(iterable)
    for first in iterator:
        # Chain the first item back with the next (size-1) items
        yield list(itertools.chain([first], itertools.islice(iterator, size - 1)))

def generate_stream(modeler: Modeler, prompt_iterator: Iterator[PreparedPrompt], batch_size: int, **kwargs):
    """
    Yields batches of ModelOutput objects as they are computed.
    """
    batch_gen = chunked_generator(prompt_iterator, batch_size)
    
    for batch_prompts in batch_gen:
        # This is where the work happens
        batch_results = modeler.generate_chat(batch_prompts, **kwargs)
        
        # We yield the whole list of results for this batch
        yield batch_results




def create_interactive_viewer(
    data_dict: Dict[tuple, pd.DataFrame], 
    group_keys: List[str],
    title: str = "Interactive DataFrame Viewer"
):
    """
    Creates an interactive viewer with multiple dropdowns that
    correctly displays the rich, interactive pandas DataFrame.
    
    This works by using the "clear_output(wait=True)" method,
    which forces a full, clean re-render of the DataFrame.

    Args:
        data_dict:  The dictionary where the key is a tuple of group values.
        group_keys: A list of strings defining what each item in the
                    tuple key represents, in the *exact same order*
                    as your groupby.
                    e.g., ['model_name', 'mode_rating', 'prompt_id']
    """
    
    all_keys_tuples = list(data_dict.keys())
    if not all_keys_tuples:
        print("Warning: The provided dictionary is empty.")
        return
        
    if len(all_keys_tuples[0]) != len(group_keys):
        print(f"Error: Mismatch between key length ({len(all_keys_tuples[0])}) "
              f"and group_keys length ({len(group_keys)}).")
        return

    # --- 1. Create a dictionary of dropdowns dynamically ---
    dropdowns = {}
    for i, key_name in enumerate(group_keys):
        unique_vals = sorted(list(set(k[i] for k in all_keys_tuples)))
        dropdowns[key_name] = widgets.Dropdown(
            options=unique_vals,
            description=f"{key_name}:",
            style={'description_width': 'initial'},
            layout={'width': 'auto'}
        )
        
    # --- 2. Create the HBox for controls ---
    controls_box = widgets.HBox(list(dropdowns.values()))

    # --- 3. Define the update function (based on your working logic) ---
    def update_display(change): # 'change' is unused, but required by .observe
        
        # A. Build the key from all dropdowns
        current_key_tuple = tuple(
            dd.value for dd in dropdowns.values()
        )
        
        # B. Get the dataframe
        df_to_show = data_dict.get(current_key_tuple)

        # C. Manually clear and redraw *everything*
        # This is the "brute force" method from your working function.
        clear_output(wait=True)
        
        # D. Re-display the title and controls
        print(f"--- {title} ---")
        display(controls_box)
        
        # E. Display the new dataframe
        if df_to_show is not None:
            # This call is now at the "top level" of the cell
            # and should render the full, interactive DataFrame.
            display(df_to_show) 
        else:
            print(f"No data for this combination: {current_key_tuple}")

    # --- 4. Link observers ---
    # Make a change to *any* dropdown trigger the update
    for dd in dropdowns.values():
        dd.observe(update_display, names='value')

    # --- 5. Initial Display ---
    # Manually draw the title and controls just once to start
    print(f"--- {title} ---")
    display(controls_box)
    
    # Manually call the update function to draw the *first* dataframe
    update_display(None)


def view_statsmodels_summaries(results_dict: dict, title: str = "Interactive Model Summary Viewer"):
    """
    Creates an interactive dropdown viewer in Jupyter
    for a dictionary of statsmodels results.

    Args:
        results_dict: The dictionary (e.g., regression_results) 
                      containing your fitted models.
        title: An optional title for the viewer.
    """

    # --- 1. Prepare the Dropdown Options ---
    # Create a mapping from a readable string to the original key
    try:
        option_map = {str(k): k for k in results_dict.keys()}
    except Exception as e:
        print(f"Error: Could not create options from dictionary keys. {e}")
        return

    # --- 2. Create the Widgets ---
    dropdown = widgets.Dropdown(
        options=option_map.keys(),
        description='Select Model Group:',
        style={'description_width': 'initial'},
        layout={'width': '80%'}
    )

    # The output area where the summary will be printed
    out = widgets.Output()

    # --- 3. Define the "on_change" Function (Nested) ---
    # This function is nested so it has access to the
    # widgets and data (option_map, results_dict, out)
    def on_dropdown_change(change):
        # 'change' is a dict, 'new' holds the selected value
        selected_string_key = change['new']

        # Get the original tuple key from our map
        tuple_key = option_map[selected_string_key]

        # Get the model from the dictionary
        model = results_dict[tuple_key]

        # Clear the output area and print the new summary
        with out:
            clear_output(wait=True)
            print(model.summary())

    # --- 4. Link the Dropdown to the Function ---
    dropdown.observe(on_dropdown_change, names='value')

    # --- 5. Display the Widgets ---
    print(f"--- {title} ---")
    display(dropdown, out)

    # --- 6. (Optional) Trigger the first display ---
    # This loads the summary for the very first model
    if option_map:
        first_key = list(option_map.keys())[0]
        on_dropdown_change({'new': first_key})


# def interactive_dataframe_selector(data_dict, description="Select option:"):
#     """
#     Creates an interactive dropdown to select and display a DataFrame.

#     Parameters:
#     - data_dict: dict
#         Dictionary where keys are dropdown labels and values are pandas DataFrames.
#     - description: str
#         Description label for the dropdown.
#     """

#     # Create dropdown from dictionary keys
#     dropdown = widgets.Dropdown(
#         options=list(data_dict.keys()),
#         description=description,
#         style={'description_width': 'initial'},
#         layout=widgets.Layout(width='300px')
#     )

#     # Callback function
#     def update_table(change):
#         clear_output(wait=True)
#         display(dropdown)
#         display(data_dict[change['new']])

#     # Attach callback
#     dropdown.observe(update_table, names='value')

#     # Initial display
#     display(dropdown)
#     display(data_dict[dropdown.value])


def interactive_dataframe_selector(data_dict, description="Select option:"):
    # label -> value pairs; value is the tuple key
    options = [(f"rating {k[0]} | prompt {k[1]}", k) for k in data_dict.keys()]
    dropdown = widgets.Dropdown(
        options=options,
        description=description,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )

    def update_table(change):
        clear_output(wait=True)
        display(dropdown)
        display(data_dict[change.new])

    dropdown.observe(update_table, names='value')
    display(dropdown)
    display(data_dict[dropdown.value])


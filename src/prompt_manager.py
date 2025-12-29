import yaml
import hashlib
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import hashlib
import yaml
from pathlib import Path
from typing import List, Optional, Dict
from collections.abc import Iterator
import torch

import yaml
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Dict, Any


@dataclass
class PreparedPrompt:
    """A simple data object to hold the final, rendered prompt data."""
    conversation: list[dict]
    id: str  # ID of the PromptTemplate that generated me
    dimension_name: str
    assistant_prefix: str
    input_id: str
    token_constraints: Optional[list[str]] = None
    constraint_ids: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    def to_analysis_record(self) -> dict:
        """
        Flattens the object into a single dictionary for the ResultsContainer.
        This keeps the logic of 'what defines me' inside the class itself.
        """
        # 1. Start with the Explicit Structure
        record = {
            'prompt_id': self.id,
            'input_id': self.input_id,
            'dimension_name': self.dimension_name,
            'assistant_prefix': self.assistant_prefix
        }

        # 2. Merge with Generic Metadata
        # (Explicit fields take precedence, or update as you prefer)
        if self.metadata:
            record.update(self.metadata)

        return record


@dataclass
class PromptTemplate:
    id: str
    name: str
    dimension_name: str
    description: str
    # TODO: throw error when len constrained_output does not match nr of output tokens
    token_constraints: list[str]
    tags: list[str]
    system_message: str
    user_message_template: str
    # assistant_prefix: str = "" #TODO: check if I need to pass this in this class, or whether PreparedPrompt is sufficient

    # Runtime cache for IDs (Not saved to YAML)
    _cached_constraint_ids: Optional[torch.Tensor] = field(
        default=None, init=False)

    @classmethod
    def from_file(cls, path: Path):
        """Loads a chat-style prompt from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Unpack the nested chat template
        chat_template = data.pop("template_chat")

        return cls(
            id=data["id"],
            name=data["name"],
            dimension_name=data["dimension_name"],
            description=data["description"],
            token_constraints=data["constrained_output"],
            tags=data["tags"],
            system_message=chat_template["system"],
            user_message_template=chat_template["user"]
        )

    @classmethod
    def from_dict(cls, data: dict):
        # Create a copy of the input dictionary.
        # This prevents the original 'data' object from being modified by pop().
        temp_data = data.copy()

        # Now, 'pop' operates on the copy, leaving the original 'data' intact.
        chat_template = temp_data.pop('template_chat')

        if temp_data.get('id') is None:
            # Hashes only the dictionary, without the ID field
            id = cls.hash_prompt(data)
        else:
            id = temp_data['id']

        return cls(
            id=id,
            name=temp_data["name"],
            dimension_name=temp_data["dimension_name"],
            description=temp_data["description"],
            token_constraints=temp_data["token_constraints"],
            tags=temp_data["tags"],
            system_message=chat_template["system"],
            user_message_template=chat_template["user"]
        )

    @staticmethod
    def hash_prompt(prompt_dict):
        # create a stable hash from sorted keys + template
        payload_dict = prompt_dict.copy()
        payload_dict.pop('id', None)
        payload = yaml.safe_dump(payload_dict, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def compute_constraint_ids(self, tokenizer) -> None:
        """
        Pre-computes and caches the token IDs for the constraints.
        """
        if not self.token_constraints:
            self._cached_constraint_ids = None
            return

        # Convert
        ids = tokenizer.convert_tokens_to_ids(self.token_constraints)

        # Sort for deterministic behavior (Critical for the Data Contract)
        ids = sorted(list(set(ids)))

        # Store as Tensor immediately to save time later
        self._cached_constraint_ids = torch.tensor(ids)

    def to_dict(self) -> dict:
        """
        Serializes the template. 
        NOTE: Excludes 'assistant_prefix' because it is a runtime experimental variable,
        not a fixed property of the prompt definition.
        """
        return {
            "id": self.id,
            "name": self.name,
            "dimension_name": self.dimension_name,
            "description": self.description,
            "token_constraints": self.token_constraints,
            "tags": self.tags,
            # We structure the chat templates nested, matching your from_dict logic
            "template_chat": {
                "system": self.system_message,
                "user": self.user_message_template
            }
        }

    def render_conversation(self,
                            row: dict | pd.Series,
                            few_shot_examples: pd.DataFrame | None = None,
                            rating_col: str | None = None,
                            assistant_prefix: str = "") -> list[Dict[str, str]]:
        """
        Renders a single prompt instance into a conversation list.

        Args:
            row: A dictionary with data to format the user message template.

        Returns:
            Returns: A rendered conversation (used to create PreparedPrompt)
        """
        # self.assistant_prefix = assistant_prefix #

        # Create the conversation structure
        conversation = []
        if self.system_message:
            conversation.append(
                {"role": "system", "content": self.system_message})
        # if few_shot_examples is not None and not few_shot_examples.empty: #TODO correct implementation if I actually need this (currently, I dont)
        #     few_shot_dict = few_shot_examples.to_dict(orient='records')
        #     for example in few_shot_dict:
        #         conversation.append({"role": "user",
        #                             "content": self.user_message_template.format(**example)}) #TODO FIX: now only works for two-step templates (might break)
        #         conversation.append({
        #             "role": "assistant",
        #             "content": f"{self.assistant_prefix}{example[rating_col]}"
        #         })

        # Format the user message with the specific data for this instance
        formatted_user_message = self.user_message_template.format(**row)
        conversation.append(
            {"role": "user", "content": formatted_user_message})

        return conversation


@dataclass
class PromptSuite:
    id: str
    # One or multiple templates belonging to one Latent Variable
    templates: Dict[str, PromptTemplate]

    @property
    def dimensions(self) -> list[str]:
        """Dynamically returns dimensions so there is no duplicate state."""
        return list(self.templates.keys())

    @property
    def tags(self) -> set[str]:
        """Dynamically returns tags (used to generate dynamic assistant prefix)"""
        if not self.templates:
            return set()  # Return empty set if no templates exist to prevent crash

        list_of_sets = [set(vals.tags) for vals in self.templates.values()]
        return set.intersection(*list_of_sets)

    @classmethod
    def from_list(cls, template_list: list[PromptTemplate]):
        templates = {
            template.dimension_name: template for template in template_list}
        suite_id = cls.generate_suite_id(templates)
        return cls(suite_id, templates)

    @classmethod
    def from_dict(cls, template_dict: Dict[str, Any]):
        templates = {}

        # 1. Structural Heuristic: Does it have a 'dimensions' block?
        if 'dimensions' in template_dict:
            # --- SUITE PATH ---
            suite_id = template_dict.get('id')

            for dim, pt_dict in template_dict['dimensions'].items():
                # Make a shallow copy to avoid mutating the input dictionary
                # (Good practice when modifying keys like 'dimension_name')
                pt_data = pt_dict.copy()
                pt_data['dimension_name'] = dim

                pt = PromptTemplate.from_dict(pt_data)
                templates[dim] = pt

            if suite_id is None:
                suite_id = PromptSuite.generate_suite_id(templates)

        elif 'template_chat' in template_dict:
            # --- ATOMIC PATH ---
            # It is a raw dictionary (e.g. from create_chat_prompt_dict)
            pt = PromptTemplate.from_dict(template_dict)

            # Use the prompt's dimension name
            dim = pt.dimension_name
            templates[dim] = pt

            # For atomic suites, the Suite ID is the Prompt ID
            suite_id = pt.id

        else:
            raise ValueError(
                "Dictionary matches neither Suite nor Template schema.")

        return cls(suite_id, templates)

    @staticmethod
    def generate_suite_id(templates: Dict[str, PromptTemplate]) -> str:
        """
        Generates a stable hash for the suite.
        Hash is based on: Sorted Dimension Names + Template Content Hashes.
        """
        # We create a dictionary of {dimension: template_id}
        # This ensures that if the template content changes, the Suite ID changes.
        # If the dimension name changes, the Suite ID changes.
        suite_signature = {
            dim: tmpl.id for dim, tmpl in templates.items()
        }

        # JSON dump with sort_keys=True guarantees deterministic ordering
        # e.g., {"clarity": "hashA", "persuasiveness": "hashB"}
        # is treated identical to {"persuasiveness": "hashB", "clarity": "hashA"}
        payload = json.dumps(suite_signature, sort_keys=True)

        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]

    def to_dict(self) -> dict:
        """
        Serializes the suite. 
        Structure: ID at root, followed by dimensions map.
        """
        dimensions_data = {}

        for dim_name, tmpl in self.templates.items():
            # 1. Get the base dict
            t_data = tmpl.to_dict()

            # 2. Cleanup Redundancy
            # The dimension_name is the KEY in the YAML, so remove it from value
            t_data.pop('dimension_name', None)

            # The ID is intrinsic to the content (hash), so we don't strictly need to save it
            # if we trust re-hashing on load. But keeping it for sanity check is fine.
            # If you want a cleaner YAML, remove it:
            t_data.pop('id', None)

            dimensions_data[dim_name] = t_data

        return {
            "id": self.id,
            "dimensions": dimensions_data
        }

    def render(
        self,
        row: dict | pd.Series,
        input_id: str,
        assistant_prefix_template: str
    ) -> list[PreparedPrompt]:
        """
        Renders a single conversation for a PreparedPrompt
        """
        prepared_prompts = []
        for dim_name, tmpl in self.templates.items():
            # Formats when it contains {variable}, otherwise same str as template
            final_prefix = assistant_prefix_template.format(dim_name=dim_name)
            # final_prefix = f"Based on the rubric, the {dim_name} rating (1-4) is:"
            p_prompt = PreparedPrompt(
                conversation=tmpl.render_conversation(row),
                id=self.id,
                dimension_name=dim_name,
                assistant_prefix=final_prefix,  # TODO called twice from different source
                input_id=input_id,
                token_constraints=tmpl.token_constraints,
                constraint_ids=tmpl._cached_constraint_ids
            )
            prepared_prompts.append(p_prompt)

        return prepared_prompts

    # def render_many(self,
    #                 rows: list[dict] | pd.DataFrame,
    #                 id_col: str,
    #                 assistant_prefix: str) -> list[PreparedPrompt]:
    #     """
    #     Eagerly renders all prompts.
    #     Note: If rows is a list[dict], strictly speaking we can't easily extract id_col
    #     unless keys match. Assuming DataFrame for safety or list of dicts with id_col.
    #     """
    #     #TODO this is just copied from Gemini, check and fix (seems okay)
    #     if isinstance(rows, pd.DataFrame):
    #         rows = rows.to_dict(orient="records")

    #     # We cannot use a simple list comprehension [self.render(row) for row in rows]
    #     # because self.render returns a LIST of prompts (one per dimension).

    #     flat_results = []
    #     for row in rows:
    #         # Fallback if list[dict] doesn't have the col?
    #         # Ideally enforce id_col presence.
    #         current_id = row.get(id_col, "unknown_id")

    #         # Pass input_id
    #         prompts = self.render(row, input_id=str(current_id), assistant_prefix=assistant_prefix)
    #         flat_results.extend(prompts)

    #     return flat_results

    def stream_render(self,
                      df: pd.DataFrame,
                      id_col: str,
                      assistant_prefix: str) -> Iterator[PreparedPrompt]:
        """
        Yields prepared prompts one by one. 
        This uses almost zero memory regardless of dataframe size.
        """
        # Safety Check
        if id_col not in df.columns:
            raise ValueError(f"ID Column '{id_col}' not found in DataFrame.")

        # Convert to dict iterator to avoid pandas overhead in loop
        records = df.to_dict(orient="records")

        for row in records:
            # render returns a list of constituents (e.g., [Clarity, Relevance])
            # 'yield from' flattens this list into the stream
            yield from self.render(row, str(row[id_col]), assistant_prefix)

    def precompute_constraints(self, tokenizer):
        """
        Iterates through all templates and pre-calculates their token IDs
        using the provided tokenizer. Call this once before generation.
        """
        print(f"Pre-computing constraint IDs for Suite {self.id}...")
        for dim_name, tmpl in self.templates.items():
            tmpl.compute_constraint_ids(tokenizer)

    def save(self, directory: Path, filename: Optional[str] = None):
        """
        Saves the suite to the specified directory.
        """
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Determine Filename
        if filename is None:
            filename = f"suite_{self.id}"
        else:
            filename = f"{filename}_suite_{self.id}"

        # Ensure extension
        if not filename.endswith('.yml'):
            filename += ".yml"

        # 2. Serialize
        data = self.to_dict()
        output_path = target_dir / filename

        # 3. Write
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False,
                               default_flow_style=False)
            print(f"✅ Saved suite to {output_path}")
            return output_path
        except IOError as e:
            print(f"❌ Error saving suite: {e}")
            raise


class PromptManager:
    def __init__(self, folder: Optional[Path] = None):
        self.folder: Optional[Path] = folder

        # 2. Only perform folder operations if a Path is provided
        if self.folder:
            # Ensure the folder exists if we're using file system operations
            self.folder.mkdir(exist_ok=True)
            print(f"PromptManager initialized with folder: {self.folder}")
        else:
            print("PromptManager initialized in memory-only (sandbox) mode.")

        self.suites: Dict[str, PromptSuite] = {}

    def load_all(self, tags_to_skip: set = set(), required_tags: set = set()) -> Dict[str, PromptSuite]:
        if not self.folder:
            print("Skipping file load: Manager is in memory-only mode.")
            return self.suites

        self.suites = {}
        files = self.list_folder_files()
        print(f"Scanning {len(files)} suites from {self.folder}...")

        for path in files:
            try:
                with open(path, "r", encoding='utf-8') as f:
                    prompt_dict = yaml.safe_load(f)
                ps = PromptSuite.from_dict(prompt_dict)
                if not required_tags.issubset(ps.tags):
                    print(f"Skipping {ps.id}")
                    continue

                if len(set.intersection(tags_to_skip, ps.tags)) > 0:
                    print(f"Skipping {ps.id}")
                    continue

                if ps.id in self.suites:
                    print(
                        f"⚠️ Warning: Duplicate Suite ID '{ps.id}' found in {path.name}. Overwriting previous entry.")
                self.suites[ps.id] = ps

            except yaml.YAMLError as e:
                print(f"❌ YAML Syntax Error in {path.name}: {e}")
            except KeyError as e:
                print(f"❌ Schema Error in {path.name}: Missing key {e}")
            except Exception as e:
                # Catch-all that actually tells you what happened
                print(f"❌ Failed to load {path.name}: {e}")

        print(f"Loaded {len(self.suites.items())} PromptSuites")
        return self.suites

    def list_folder_files(self):
        # Only proceed if a folder exists
        if self.folder:
            return list(self.folder.glob("*.yml"))
        return []  # Return an empty list if no folder is set

    def save_prompt_suite(self, prompt_suite: PromptSuite):
        if self.folder:
            prompt_suite.save(self.folder)
        else:
            print("No folder specified in PromptManager. Quitting.")


class PromptManager_old:
    #  Make 'folder' optional and set the default to None
    def __init__(self, folder: Optional[Path] = None):

        self.folder: Optional[Path] = folder

        # 2. Only perform folder operations if a Path is provided
        if self.folder:
            # Ensure the folder exists if we're using file system operations
            self.folder.mkdir(exist_ok=True)
            print(f"PromptManager initialized with folder: {self.folder}")
        else:
            print("PromptManager initialized in memory-only (sandbox) mode.")

        self.prompt_suites: PromptSuite

    @staticmethod
    def hash_prompt(prompt_dict):
        # create a stable hash from sorted keys + template
        payload = yaml.safe_dump(prompt_dict, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def load_from_dict(self, prompt_dict: dict) -> PromptTemplate:
        """
        Loads a single PromptTemplate object from a dictionary definition
        and updates the internal state.

        Args:
            prompt_dict (dict): A dictionary representing the prompt data,
                                typically including the nested 'template_chat'.

        Returns:
            The newly created and loaded PromptTemplate object.
        """
        # 1. Compute ID if not present (essential for manager state)
        if prompt_dict["id"] is None:
            prompt_dict["id"] = self.hash_prompt(prompt_dict)

        # 2. Create the PromptTemplate object
        # Note: from_dict() modifies the input dict (pops 'template_chat'), so
        # we pass a copy if the original dict might be reused.
        # For simplicity, assuming the original dict is safe to modify by from_dict
        try:
            # Create a copy as from_dict mutates the input by popping 'template_chat'
            p = PromptTemplate.from_dict(prompt_dict.copy())
        except Exception as e:
            print(f"Error: Failed to create PromptTemplate from dict: {e}")
            raise

        # 3. Update the internal state
        if not p.id:
            # This should not happen if hash_prompt is used, but is a safe check
            raise ValueError("Prompt loaded from dict has no 'id' field.")

        if p.id in self.prompt_templates:
            print(f"Warning: Overwriting existing prompt with ID: {p.id}")

        self.prompt_templates[p.id] = p

        return p

    def list_prompt_files(self):
        # Only proceed if a folder exists
        if self.folder:
            return list(self.folder.glob("*.yml"))
        return []  # Return an empty list if no folder is set

    def load_all(self) -> Dict[str, PromptTemplate]:
        """
        Loads ALL prompts from the folder, refreshing the internal state.

        Returns:
            A dictionary of all loaded {prompt_id: Prompt_object}.
        """
        if not self.folder:
            print("Skipping file load: Manager is in memory-only mode.")
            return self.prompt_templates

        self.prompt_templates = {}  # Clear existing state
        for path in self.list_prompt_files():
            try:
                p = PromptTemplate.from_file(path)
                if not p.id:
                    print(
                        f"Warning: Prompt {path} has no 'id' field. Skipping.")
                    continue
                self.prompt_templates[p.id] = p
            except Exception as e:
                print(
                    f"Warning: Failed to load Prompt object from {path}: {e}")

        return self.prompt_templates

    def get_filtered_prompts(self,
                             prompt_template_ids: Optional[List[str]] = None,
                             required_tags: Optional[List[str]] = None) -> Dict[str, PromptTemplate]:
        """
        Filters the currently loaded prompts (self.prompts) based on criteria.
        Does not modify the internal self.prompts state.

        Args:
            prompt_ids (Optional[List[str]]): If provided, filters for
                prompts whose IDs are in this list.
            required_tags (Optional[List[str]]): If provided, filters for
                prompts that contain ALL tags from this list.

        Returns:
            A new dictionary of {prompt_id: Prompt_object} matching the filters.
        """
        # Start with a copy of all loaded prompts
        filtered_prompt_templates = self.prompt_templates.copy()

        # 1. Filter by ID
        if prompt_template_ids:
            filtered_prompt_templates = {
                pid: p for pid, p in filtered_prompt_templates.items()
                if pid in prompt_template_ids
            }

        # 2. Filter by Tags
        if required_tags:
            filtered_prompt_templates = {
                pid: p for pid, p in filtered_prompt_templates.items()
                if all(req_tag in p.tags for req_tag in required_tags)
            }

        return filtered_prompt_templates

    def save_prompt(self, prompt_data: dict):
        if not self.folder:
            raise RuntimeError(
                "Cannot save prompt: PromptManager initialized in memory-only mode (no folder specified).")

        # compute hash and save
        prompt_data["id"] = self.hash_prompt(prompt_data)
        path = self.folder / f"{prompt_data['name']}.yml"
        with open(path, "w") as f:
            yaml.safe_dump(prompt_data, f, sort_keys=False)
        return path

    # def set_prefix_for_prompts(self,
    #                            assistant_prefix: str,
    #                            prompt_ids: Optional[List[str]] = None,
    #                            required_tags: Optional[List[str]] = None):
    #     """
    #     Modifies the 'assistant_prefix' attribute of loaded prompts in memory.

    #     This allows for dynamically setting experimental variables without
    #     changing the source YAML files.

    #     Args:
    #         assistant_prefix (str): The prefix to set (e.g., "Rating: " or "").
    #         prompt_ids (Optional[List[str]]): Filter for specific prompt IDs.
    #         required_tags (Optional[List[str]]): Filter for specific tags.
    #     """

    #     # 1. Find the prompts to modify using your existing filter logic
    #     # This returns a dict {pid: Prompt_Object}
    #     prompts_to_modify = self.get_filtered_prompts(
    #         prompt_ids, required_tags)

    #     # 2. Modify the attribute on the *actual* objects in self.prompts
    #     #    This works because get_filtered_prompts returns a dict of
    #     #    references to the original objects in self.prompts.
    #     for pid in prompts_to_modify.keys():
    #         if pid in self.prompt_templates:
    #             self.prompt_templates[pid].assistant_prefix = assistant_prefix
    #         else:
    #             # This should not happen, but it's a safe check
    #             print(
    #                 f"Warning: Prompt ID {pid} not found in main prompt list.")


def create_prompt_dict(
        prompt_text: str,
        name: str,
        description: str,
        token_constraints: list[str],
        tags: list[str],
        version: int = 1):
    """
    Dynamically creates a YAML-ready dictionary for a new prompt.
    """
    base = {
        "id": None,
        "name": name,
        "description": description,
        "token_constraints": token_constraints,
        "tags": tags or [],
        "version": version,
        "template": prompt_text.strip(),
    }
    return base


def create_chat_prompt_dict(
    name: str,
    description: str,
    template_chat: dict,
    token_constraints: list[str],
    tags: list[str] | None = None,
    dimension_name="holistic"
) -> dict:
    """
    Dynamically creates a YAML-ready dictionary for a new chat-style prompt.
    """
    prompt_definition = {
        "id": None,
        "name": name,
        "dimension_name": dimension_name,
        "description": description,
        "token_constraints": token_constraints,
        "tags": tags or [],
        "template_chat": template_chat
    }
    return prompt_definition


# def generate_prepared_prompt(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, input_data: dict, dimension_name: str="holistic", constrained_output: list[str] = [""]):
#     template_chat = {
#     "system": SYSTEM_PROMPT.strip(),
#     "user": USER_PROMPT_TEMPLATE.strip()}

#     chat_prompt_dict = create_chat_prompt_dict(
#         name="sandbox_temp",
#         dimension_name=dimension_name,
#         description="Auto-generated for sandbox use.",
#         template_chat=template_chat,
#         constrained_output=constrained_output,
#         tags=["sandbox"],
#         version = 1
#     )

#     prompt_template = PromptTemplate.from_dict(chat_prompt_dict)
#     rendered_prompt = prompt_template.render(input_data)

#     return prompt_template, rendered_prompt #TODO rewrite function for PromptSuite compatibility


def create_empty_prompt_template():
    template_chat = {
        "system": "",
        "user": ""}

    chat_prompt_dict = create_chat_prompt_dict(
        name="sandbox_temp",
        description="Auto-generated for sandbox use.",
        template_chat=template_chat,
        token_constraints=[""],
        tags=["sandbox"],
        dimension_name='holistic'
    )

    prompt_template = PromptTemplate.from_dict(chat_prompt_dict)
    return prompt_template

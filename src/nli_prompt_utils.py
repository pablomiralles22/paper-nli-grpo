import yaml
import re

class NLIPromptUtils:
    LABEL_TO_LABELID = {"entailment": 0, "neutral": 1, "contradiction": 2}
    LABELID_TO_LABEL = {v: k for k, v in LABEL_TO_LABELID.items()}

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = NLIPromptUtils("configs/prompts/deepseek.yaml")
        return cls._instance

    @classmethod
    def set_config_path(cls, config_path):
        cls._instance = NLIPromptUtils(config_path)

    def __init__(self, config_path):
        print(f"Loading NLI prompt utils with config: {config_path}")
        self.config_path = config_path
        with open(self.config_path) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.prompt_template = self.config["prompt"].strip()
        self.response_template = self.config["response"].strip()
        self.self_verify_template = self.config["self_verify"].strip()
        self.label_template = self.config["label_template"].strip()
        self.format_regex = self.config.get("format_regex")

    # ======================================================================================== #
    # Message building methods
    # ======================================================================================== #
    def get_system_prompt(self):
        if "system_prompt" in self.config:
            return [{"role": "system", "content": self.config["system_prompt"]}]
        return []

    def build_incomplete_example(self, premise, hypothesis):
        user_prompt = self.prompt_template.format(premise=premise, hypothesis=hypothesis)
        return [{"role": "user", "content": user_prompt}]

    def build_complete_example(self, premise, hypothesis, explanation, label):
        user_prompt = self.prompt_template.format(premise=premise, hypothesis=hypothesis)
        prediction = self.LABELID_TO_LABEL[label]
        assistant_response = self.response_template.format(explanation=explanation, prediction=prediction)
        
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]

    def build_self_verify_prompt_example(self, premise, hypothesis, explanation):
        user_prompt = self.prompt_template.format(premise=premise, hypothesis=hypothesis)
        assistant_response = self.self_verify_template.format(explanation=explanation)
        
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]

    # ======================================================================================== #
    # Extraction methods
    # ======================================================================================== #
    def extract_label_id(self, response):
        response = response.lower()
        for label in self.LABEL_TO_LABELID:
            label_str = self.label_template.format(label=label).lower()
            if label_str in response:
                return self.LABEL_TO_LABELID[label]
        return -1

    def extract_explanation(self, response):
        regex = self.label_template.format(label=".*")
        return re.sub(regex, '', response).strip()

    # ======================================================================================== #
    # Format checking methods
    # ======================================================================================== #
    def is_format_valid(self, completion):
        if self.format_regex is None:
            return self.extract_label_id(completion) != -1
        return bool(re.fullmatch(self.format_regex, completion.strip(), re.MULTILINE | re.DOTALL))
    
    # ======================================================================================== #
    # Auxiliary methods
    # ======================================================================================== #
    @classmethod
    def label_2_id(cls, label):
        return cls.LABEL_TO_LABELID[label]

    @classmethod
    def id_2_label(cls, label_id):
        return cls.LABELID_TO_LABEL[label_id]

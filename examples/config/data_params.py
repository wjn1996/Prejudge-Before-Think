BOOTSTRAPPING_PARAMS = {
    "gsm8k": {
        "data_name": "gsm8k",
        "data_path": {
            "train": "gsm8k_train.jsonl",
            "dev": None,
            "test": "gsm8k_test.jsonl",
        },
    },
    "gpqa_diamond": {
        "data_name": "gpqa_diamond",
        "data_path": {
            "train": None,
            "dev": None,
            "test": "GPQA_Diamond_test.jsonl"
        },
    },
    "gpqa_diamond_mcqa": {
        "data_name": "gpqa_diamond_mcqa",
        "data_path": {
            "train": None,
            "dev": None,
            "test": "GPQA_Diamond_MCQA_test.jsonl"
        },
    },
    "aqua": {
        "data_name": "aqua",
        "data_path": {
            "train": "aqua_rat_train.jsonl",
            "dev": "aqua_rat_dev.jsonl",
            "test": "aqua_rat_test.jsonl"
        },
    },
    "gsm8k": {
        "data_name": "gsm8k",
        "data_path": {
            "train": "gsm8k_train.jsonl",
            "dev": None,
            "test": "gsm8k_test.jsonl"
        },
    },
    "svamp": {
        "data_name": "svamp",
        "data_path": {
            "train": "svamp_train.jsonl",
            "dev": None,
            "test": "svamp_test.jsonl"
        },
    },
    "prm800k": {
        "data_name": "prm800k",
        "data_path": {
            "train": "PRM800K_train.jsonl",
            "dev": None,
            "test": "PRM800K_test.jsonl"
        },
    },
    "math_hard": {
        "data_name": "math_hard",
        "data_path": {
            "train": "MATH_hard_train.jsonl",
            "dev": None,
            "test": "MATH_hard_test.jsonl"
        },
    },
    "tabmwp": {
        "data_name": "tabmwp",
        "data_path": {
            "train": "tabmwp_train.jsonl",
            "dev": None,
            "test": None
        },
    },
    "aime": {
        "data_name": "aime",
        "data_path": {
            "train": "AIME_train.jsonl",
            "dev": None,
            "test": None
        },
    },
}

DISTILLATING_PARAMS = {
    "prejudge_critique": {
        "data_name": "prejudge_critique",
        "data_path": {
            "train": "data/math/prejudge_examples/prejudge_generation_examples_186944.jsonl",
            "dev": None,
            "test": None,
        }
    }
}
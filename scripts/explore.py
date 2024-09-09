import json
import os
import shutil
import subprocess
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def load_model_info(model_dir):
    with open(os.path.join(model_dir, "hyperparams.json"), "r") as f:
        hyperparams = json.load(f)

    return {
        "dir": os.path.basename(model_dir),
        "max_context_length": hyperparams["max_context_length"],
        "num_embeddings": hyperparams["num_embeddings"],
        "num_layers": hyperparams["num_layers"],
        "num_heads": hyperparams["num_heads"],
        "description": hyperparams.get("description", "No description available."),
        "recommended": hyperparams.get("recommended", False),
    }


def get_model_choices(base_dir):
    choices = []
    for model_dir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, model_dir)
        if os.path.isdir(full_path):
            model_info = load_model_info(full_path)
            choices.append(model_info)
    return sorted(choices, key=lambda x: (not x["recommended"], x["dir"]))


def is_command_available(command):
    return shutil.which(command) is not None


def select_model_fzf(choices, base_dir):
    try:
        preview_command = f"jq -r '.description' {base_dir}/${{+2}}/hyperparams.json | fold -s -w $(($(tput cols)/2-4))"
        fzf = subprocess.Popen(
            [
                "fzf",
                "--ansi",
                "--height=100%",
                "--layout=reverse",
                "--prompt=Welcome to ChessGPT ♟️! Select a model (the first option is recommended): ",
                "--preview",
                preview_command,
                "--preview-window=right:50%:wrap",
                "--with-nth=2..",  # Display from the second field onwards
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        fzf_input = "\n".join(
            f"{choice['recommended']}\t{choice['dir']}" for choice in choices
        ).encode()
        selected, _ = fzf.communicate(input=fzf_input)
        selected = selected.decode().strip()

        for choice in choices:
            if choice["dir"] in selected:
                return choice
    except FileNotFoundError:
        print("fzf not found. Falling back to simple CLI selection.")
        return select_model_cli(choices)


def select_model_cli(choices):
    print("Welcome to ChessGPT ♟️! Select a model:")
    for i, choice in enumerate(choices):
        recommended = "(Recommended) " if choice["recommended"] else ""
        print(f"{i + 1}. {recommended}{choice['dir']}")
        print(f"Description: {choice['description']}\n")

    while True:
        try:
            selection = (
                int(input("Enter the number of the model you want to use: ")) - 1
            )
            if 0 <= selection < len(choices):
                return choices[selection]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    base_dir = os.path.join(project_root, "trained_models")
    choices = get_model_choices(base_dir)

    if is_command_available("fzf") and is_command_available("jq"):
        selected_model = select_model_fzf(choices, base_dir)
    else:
        if not is_command_available("fzf"):
            print("fzf is not installed. Falling back to simple CLI selection.")
        elif not is_command_available("jq"):
            print("jq is not installed. Falling back to simple CLI selection.")
        selected_model = select_model_cli(choices)

    if selected_model:
        print(f"Selected model: {selected_model['dir']}")

        play_command = [
            "poetry",
            "run",
            "play",
            "--input-model-file",
            os.path.join(base_dir, selected_model["dir"], "model.pth"),
            "--input-tokenizer-file",
            os.path.join(base_dir, selected_model["dir"], "tokenizer.json"),
            "--max-context-length",
            str(selected_model["max_context_length"]),
            "--num-embeddings",
            str(selected_model["num_embeddings"]),
            "--num-layers",
            str(selected_model["num_layers"]),
            "--num-heads",
            str(selected_model["num_heads"]),
        ]

        print("Running play script with selected model...")
        subprocess.run(play_command)
    else:
        print("No model selected. Exiting.")


if __name__ == "__main__":
    main()

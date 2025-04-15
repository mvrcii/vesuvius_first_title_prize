import os
import platform
import subprocess
import sys
import threading
import time

ENV_NAME = "scroll5-title"
PYTHON_VERSION = "3.10.16"

# ANSI colors for terminals that support it
GREEN = "\033[92m" if sys.stdout.isatty() else ""
BLUE = "\033[94m" if sys.stdout.isatty() else ""
YELLOW = "\033[93m" if sys.stdout.isatty() else ""
RED = "\033[91m" if sys.stdout.isatty() else ""
CYAN = "\033[96m" if sys.stdout.isatty() else ""
BOLD = "\033[1m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""


class Spinner:
    """A spinner that shows while a command is running"""

    def __init__(self, step, total_steps, message):
        self.step_text = f"{BLUE}[{step}/{total_steps}]{RESET}"
        self.message = message
        self.spinning = False
        self.spinner_thread = None
        self.chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def spin(self):
        """Display the spinner animation"""
        i = 0
        while self.spinning:
            sys.stdout.write(f"\r{self.step_text} {self.chars[i % len(self.chars)]} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        """Start the spinner"""
        self.spinning = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self, success=True):
        """Stop the spinner and show success/failure"""
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()

        if success:
            sys.stdout.write(f"\r{self.step_text} {GREEN}✓{RESET} {self.message}\n")
        else:
            sys.stdout.write(f"\r{self.step_text} {RED}✗{RESET} {self.message}\n")
        sys.stdout.flush()


def run_command_with_spinner(command, step, total_steps, message, capture_output=True):
    """Run a command with a spinner showing progress"""
    spinner = Spinner(step, total_steps, message)
    spinner.start()

    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=True,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
        else:
            result = subprocess.run(command, shell=True, check=True)

        spinner.stop(success=True)
        return True
    except subprocess.CalledProcessError:
        spinner.stop(success=False)
        return False


def show_message_with_spinner(step, total_steps, message, status=True, pause=0.5):
    """Show a message with spinner animation but no command execution"""
    spinner = Spinner(step, total_steps, message)
    spinner.start()
    time.sleep(pause)  # Brief pause for visual consistency
    spinner.stop(success=status)
    return status


def environment_exists(env_name):
    """Check if a conda environment already exists"""
    try:
        output = subprocess.run(
            "conda env list",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        for line in output.stdout.splitlines():
            if line.strip().startswith(env_name + " ") or line.strip() == env_name:
                return True
        return False
    except subprocess.CalledProcessError:
        return False


def has_gpu():
    """Check if NVIDIA GPU is available"""
    if platform.system() == "Windows":
        try:
            subprocess.run("wmic path win32_VideoController get name | findstr NVIDIA",
                           shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
    else:
        try:
            subprocess.run("which nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1",
                           shell=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False


def main():
    print(f"\n{BLUE}╭─────────────────────────────╮{RESET}")
    print(f"{BLUE}│  {GREEN}PyTorch Environment Setup{BLUE}  │{RESET}")
    print(f"{BLUE}╰─────────────────────────────╯{RESET}\n")

    # Step 1: Create the conda environment (or use existing)
    if environment_exists(ENV_NAME):
        show_message_with_spinner(1, 4, f"Environment {GREEN}{ENV_NAME}{RESET} already exists, using it", status=True)
    else:
        run_command_with_spinner(
            f"conda create -y -q -n {ENV_NAME} python={PYTHON_VERSION}",
            1, 4, "Creating Conda Python environment"
        )

    # Step 2: Install PyTorch
    has_gpu_available = has_gpu()
    cuda_text = f"{GREEN if has_gpu_available else YELLOW}[CUDA: {'✓' if has_gpu_available else '✗'}]{RESET}"

    if has_gpu_available:
        run_command_with_spinner(
            f"conda run -n {ENV_NAME} pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126",
            2, 4, f"Installing PyTorch {cuda_text}"
        )
    else:
        run_command_with_spinner(
            f"conda run -n {ENV_NAME} pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            2, 4, f"Installing PyTorch {cuda_text}"
        )

    # Step 3: Install requirements.txt if it exists
    if os.path.isfile("requirements.txt"):
        run_command_with_spinner(
            f"conda run -n {ENV_NAME} pip install -q -r requirements.txt",
            3, 4, "Installing dependencies"
        )
    else:
        show_message_with_spinner(3, 4, "No requirements.txt found, skipping", status=True)

    run_command_with_spinner(
        f"conda run -n {ENV_NAME} pip install -e .",
        4, 4, "Installing phoenix package"
    )

    print(f"\n{GREEN}✅  Setup complete!{RESET}")
    print(f"\n{BLUE}Activate with:{RESET} {CYAN}{BOLD}conda activate {ENV_NAME}{RESET}")
    print()


if __name__ == "__main__":
    main()

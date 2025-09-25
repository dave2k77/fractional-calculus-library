# Environment Setup Guide for hpfracc Project

This guide explains how to set up automatic environment activation for the hpfracc project, ensuring that the correct conda environment is activated whenever you work on this project.

## 🚀 Quick Setup

### Option 1: Automatic Setup (Recommended)
Run the setup script to automatically configure your shell:

```bash
./setup_auto_activation.sh
```

This script will:
- Detect your shell type (bash, zsh, or fish)
- Configure automatic environment activation
- Create backups of your shell configuration files
- Set up project-specific environment variables

### Option 2: Manual Setup
If you prefer to configure manually, follow the sections below.

## 📋 Prerequisites

1. **Conda/Miniconda installed** on your system
2. **One of the supported environments** created:
   - `hpfracc-env` (recommended)
   - `fracnn` (from config/environment.yml)

## 🔧 Environment Creation

### Create hpfracc-env (Recommended)
```bash
conda create -n hpfracc-env python=3.11
conda activate hpfracc-env
pip install -r requirements.txt
pip install -r requirements_ml.txt
```

### Create from environment.yml
```bash
conda env create -f config/environment.yml
conda activate fracnn
```

## 🐚 Shell Configuration

### Bash/Zsh Setup
The setup script automatically adds this to your `.bashrc` or `.zshrc`:

```bash
# Auto-activation for hpfracc project
hpfracc_auto_activate() {
    if [[ "$PWD" == "/path/to/project"* ]] && [[ "$CONDA_DEFAULT_ENV" != "hpfracc-env" ]] && [[ "$CONDA_DEFAULT_ENV" != "fracnn" ]]; then
        if conda env list | grep -q "hpfracc-env"; then
            conda activate hpfracc-env
            echo "✅ Auto-activated hpfracc-env environment"
        elif conda env list | grep -q "fracnn"; then
            conda activate fracnn
            echo "✅ Auto-activated fracnn environment"
        fi
    elif [[ "$PWD" != "/path/to/project"* ]] && [[ "$CONDA_DEFAULT_ENV" == "hpfracc-env" ]] || [[ "$CONDA_DEFAULT_ENV" == "fracnn" ]]; then
        conda deactivate
        echo "🔄 Auto-deactivated hpfracc environment"
    fi
}

# Add to PROMPT_COMMAND for bash
if [[ "$SHELL" == *"bash"* ]]; then
    PROMPT_COMMAND="hpfracc_auto_activate; $PROMPT_COMMAND"
fi

# Add to precmd for zsh
if [[ "$SHELL" == *"zsh"* ]]; then
    precmd() {
        hpfracc_auto_activate
    }
fi
```

### Fish Shell Setup
For fish shell users, the setup script adds:

```fish
# Auto-activation for hpfracc project
function hpfracc_auto_activate
    if string match -q "/path/to/project*" (pwd)
        if conda env list | grep -q "hpfracc-env"
            conda activate hpfracc-env
            echo "✅ Auto-activated hpfracc-env environment"
        else if conda env list | grep -q "fracnn"
            conda activate fracnn
            echo "✅ Auto-activated fracnn environment"
        end
    else if string match -q "hpfracc-env" $CONDA_DEFAULT_ENV
        conda deactivate
        echo "🔄 Auto-deactivated hpfracc environment"
    else if string match -q "fracnn" $CONDA_DEFAULT_ENV
        conda deactivate
        echo "🔄 Auto-deactivated hpfracc environment"
    end
end

# Hook into fish_prompt
function fish_prompt
    hpfracc_auto_activate
    # Call the original fish_prompt if it exists
    if functions -q fish_prompt_original
        fish_prompt_original
    else
        echo -n "$ "
    end
end
```

## 🔄 How It Works

1. **Directory Detection**: The script monitors your current working directory
2. **Auto-Activation**: When you enter the project directory, it automatically activates the appropriate conda environment
3. **Auto-Deactivation**: When you leave the project directory, it deactivates the environment
4. **Smart Fallback**: Tries `hpfracc-env` first, then falls back to `fracnn` if available

## 📁 Project Structure

After setup, you'll have these new files:

```
fractional-calculus-library/
├── .envrc                    # direnv configuration (if using direnv)
├── .vscode/
│   ├── settings.json        # VS Code workspace settings
│   ├── extensions.json      # Recommended extensions
│   └── launch.json          # Debug configurations
├── tools/
│   ├── activate_env.sh      # Manual activation script
├── setup_auto_activation.sh # Auto-setup script
├── project.env              # Project environment variables
└── ENVIRONMENT_SETUP_GUIDE.md # This file
```

## 🛠️ Manual Activation

If you need to manually activate the environment:

```bash
source tools/activate_env.sh
```

## 🔍 Troubleshooting

### Environment Not Found
```bash
# Check available environments
conda env list

# Create missing environment
conda create -n hpfracc-env python=3.11
```

### Auto-activation Not Working
1. **Restart your terminal** after running the setup script
2. **Check shell configuration**:
   ```bash
   # For bash
   cat ~/.bashrc | grep hpfracc
   
   # For zsh
   cat ~/.zshrc | grep hpfracc
   ```
3. **Verify conda initialization**:
   ```bash
   conda init bash  # or zsh, fish
   ```

### VS Code Issues
1. **Select Python interpreter**: `Ctrl+Shift+P` → "Python: Select Interpreter"
2. **Choose conda environment**: Select `hpfracc-env` or `fracnn`
3. **Reload VS Code**: `Ctrl+Shift+P` → "Developer: Reload Window"

## 🧪 Testing the Setup

1. **Navigate to project directory**:
   ```bash
   cd /path/to/fractional-calculus-library
   ```

2. **Check environment activation**:
   ```bash
   echo $CONDA_DEFAULT_ENV
   python --version
   ```

3. **Run a quick test**:
   ```bash
   python -c "import hpfracc; print('Environment working!')"
   ```

## 📚 Additional Resources

- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [VS Code Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [direnv Documentation](https://direnv.net/) (if using direnv)

## 🔄 Updating Configuration

To update the auto-activation configuration:

1. **Remove old configuration** from your shell RC file
2. **Run setup script again**:
   ```bash
   ./setup_auto_activation.sh
   ```

## 📞 Support

If you encounter issues:

1. Check this guide first
2. Review the troubleshooting section
3. Check your shell configuration files
4. Verify conda environment status

---

**Note**: This setup creates backups of your shell configuration files before making changes. You can restore from these backups if needed.

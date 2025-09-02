#!/bin/bash
# Setup script for automatic environment activation
# This script configures your shell to automatically activate the hpfracc environment
# when you enter the project directory

echo "🚀 Setting up automatic environment activation for hpfracc project..."

# Get the current directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

# Detect shell type
SHELL_TYPE=""
if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_TYPE="bash"
    SHELL_RC="$HOME/.bashrc"
elif [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_TYPE="zsh"
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == *"fish"* ]]; then
    SHELL_TYPE="fish"
    SHELL_RC="$HOME/.config/fish/config.fish"
else
    echo "❌ Unsupported shell: $SHELL"
    echo "Supported shells: bash, zsh, fish"
    exit 1
fi

echo "📋 Detected shell: $SHELL_TYPE"

# Function to add auto-activation to bash/zsh
setup_bash_zsh() {
    local rc_file="$1"
    local project_dir="$2"
    local project_name="$3"
    
    # Check if already configured
    if grep -q "hpfracc.*auto.*activate" "$rc_file"; then
        echo "✅ Auto-activation already configured in $rc_file"
        return 0
    fi
    
    # Create backup
    cp "$rc_file" "${rc_file}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 Created backup: ${rc_file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Add auto-activation function
    cat >> "$rc_file" << EOF

# Auto-activation for hpfracc project
hpfracc_auto_activate() {
    if [[ "\$PWD" == "$project_dir"* ]] && [[ "\$CONDA_DEFAULT_ENV" != "hpfracc-env" ]] && [[ "\$CONDA_DEFAULT_ENV" != "fracnn" ]]; then
        if conda env list | grep -q "hpfracc-env"; then
            conda activate hpfracc-env
            echo "✅ Auto-activated hpfracc-env environment"
        elif conda env list | grep -q "fracnn"; then
            conda activate fracnn
            echo "✅ Auto-activated fracnn environment"
        fi
    elif [[ "\$PWD" != "$project_dir"* ]] && [[ "\$CONDA_DEFAULT_ENV" == "hpfracc-env" ]] || [[ "\$CONDA_DEFAULT_ENV" == "fracnn" ]]; then
        conda deactivate
        echo "🔄 Auto-deactivated hpfracc environment"
    fi
}

# Add to PROMPT_COMMAND for bash
if [[ "\$SHELL" == *"bash"* ]]; then
    PROMPT_COMMAND="hpfracc_auto_activate; \$PROMPT_COMMAND"
fi

# Add to precmd for zsh
if [[ "\$SHELL" == *"zsh"* ]]; then
    precmd() {
        hpfracc_auto_activate
    }
fi
EOF

    echo "✅ Added auto-activation to $rc_file"
}

# Function to setup fish shell
setup_fish() {
    local config_file="$1"
    local project_dir="$2"
    local project_name="$3"
    
    # Check if already configured
    if grep -q "hpfracc.*auto.*activate" "$config_file"; then
        echo "✅ Auto-activation already configured in $config_file"
        return 0
    fi
    
    # Create backup
    cp "$config_file" "${config_file}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 Created backup: ${config_file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Add auto-activation function
    cat >> "$config_file" << EOF

# Auto-activation for hpfracc project
function hpfracc_auto_activate
    if string match -q "$project_dir*" (pwd)
        if conda env list | grep -q "hpfracc-env"
            conda activate hpfracc-env
            echo "✅ Auto-activated hpfracc-env environment"
        else if conda env list | grep -q "fracnn"
            conda activate fracnn
            echo "✅ Auto-activated fracnn environment"
        end
    else if string match -q "hpfracc-env" \$CONDA_DEFAULT_ENV
        conda deactivate
        echo "🔄 Auto-deactivated hpfracc environment"
    else if string match -q "fracnn" \$CONDA_DEFAULT_ENV
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
        echo -n "\$ "
    end
end
EOF

    echo "✅ Added auto-activation to $config_file"
}

# Setup based on shell type
case "$SHELL_TYPE" in
    "bash"|"zsh")
        setup_bash_zsh "$SHELL_RC" "$PROJECT_DIR" "$PROJECT_NAME"
        ;;
    "fish")
        setup_fish "$SHELL_RC" "$PROJECT_DIR" "$PROJECT_NAME"
        ;;
esac

echo ""
echo "🎉 Setup complete! Here's what was configured:"
echo ""
echo "📁 Project directory: $PROJECT_DIR"
echo "🐚 Shell configuration: $SHELL_RC"
echo "🔄 Auto-activation: Enabled"
echo ""
echo "📋 To use the new configuration:"
echo "   1. Restart your terminal or run: source $SHELL_RC"
echo "   2. Navigate to the project directory: cd $PROJECT_DIR"
echo "   3. The environment will activate automatically"
echo ""
echo "🔧 Manual activation is still available:"
echo "   source activate_env.sh"
echo ""
echo "📚 For more information, see the README.md file"

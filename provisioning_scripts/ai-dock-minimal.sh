#!/bin/bash
PROVISIONING_SCRIPT_NAME="ai-dock-minimal"
echo "Pseudorandom '$PROVISIONING_SCRIPT_NAME' Provisioning Script is running..."

APT_PACKAGES=()
PIP_PACKAGES=(
    "diffusers"
)
NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/Pseudotools/Pseudocomfy"
    "https://github.com/Pseudotools/ComfyUI_IPAdapter_plus"
)
CHECKPOINT_MODELS=(
    "https://huggingface.co/pseudotools/pseudocomfy-models/resolve/main/checkpoints/sd_xl_base_1.0.safetensors"
)
UNET_MODELS=()
LORA_MODELS=()
VAE_MODELS=()
ESRGAN_MODELS=()
CONTROLNET_MODELS=(
    "https://huggingface.co/pseudotools/pseudocomfy-models/resolve/main/controlnet/control-lora-depth-rank128.safetensors"
)
IP_ADAPTER_MODELS=(
    "https://huggingface.co/pseudotools/pseudocomfy-models/resolve/main/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"
)
CLIP_VISION_MODELS=(
    "https://huggingface.co/pseudotools/pseudocomfy-models/resolve/main/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    "https://huggingface.co/pseudotools/pseudocomfy-models/resolve/main/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"
)
CLIP_VISION_FILENAMES=(
    "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###


function provisioning_start() {
    if [[ ! -d /opt/environments/python ]]; then 
        export MAMBA_BASE=true
    fi
    source /opt/ai-dock/etc/environment.sh
    source /opt/ai-dock/bin/venv-set.sh comfyui
    
    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages

    provisioning_create_extra_model_paths_yaml
    
    provisioning_get_models \
        "${STORAGE_PATH}/models/checkpoints" \
        "${CHECKPOINT_MODELS[@]}"
    provisioning_get_models \
        "${STORAGE_PATH}/models/unet" \
        "${UNET_MODELS[@]}"
    provisioning_get_models \
        "${STORAGE_PATH}/models/loras" \
        "${LORA_MODELS[@]}"
    provisioning_get_models \
        "${STORAGE_PATH}/models/controlnet" \
        "${CONTROLNET_MODELS[@]}"
    provisioning_get_models \
        "${STORAGE_PATH}/models/vae" \
        "${VAE_MODELS[@]}"
    provisioning_get_models \
        "${STORAGE_PATH}/models/esrgan" \
        "${ESRGAN_MODELS[@]}"
    
    ## KSTEINFE ADDED
    provisioning_get_ipadapter
    provisioning_get_clip_vision
    ## 
    
    provisioning_print_end
}

# ksteinfe
function provisioning_get_ipadapter() {
    provisioning_get_models \
        "${STORAGE_PATH}/models/ipadapter" \
        "${IP_ADAPTER_MODELS[@]}"
}



# ksteinfe
function provisioning_get_clip_vision() {
    dir="${STORAGE_PATH}/models/clip_vision"
    mkdir -p "$dir"
    
    for i in "${!CLIP_VISION_MODELS[@]}"; do
        url="${CLIP_VISION_MODELS[i]}"
        filename="${CLIP_VISION_FILENAMES[i]}"
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}/${filename}"
        printf "\n"
    done
}


# ksteinfe
function provisioning_create_extra_model_paths_yaml() {
    local model_root="${STORAGE_PATH}/models"
    local config_file="${WORKSPACE}/ComfyUI/extra_model_paths.yaml"

    # Define all the model subfolders you want
    local folders=(
        checkpoints
        clip
        clip_vision
        controlnet
        ipadapter
        loras
    )

    # Ensure each directory exists
    for folder in "${folders[@]}"; do
        mkdir -p "${model_root}/${folder}"
    done

    # Build YAML content into a variable
    local yaml_content="custom_models:"
    for folder in "${folders[@]}"; do
        yaml_content+="
  ${folder}: ${model_root}/${folder}"
    done

    # Write it to the file
    echo "$yaml_content" > "$config_file"

    echo "[✓] Created model directories and wrote $config_file"
}




function pip_install() {
    if [[ -z $MAMBA_BASE ]]; then
            "$COMFYUI_VENV_PIP" install --no-cache-dir "$@"
        else
            micromamba run -n comfyui pip install --no-cache-dir "$@"
        fi
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
            sudo $APT_INSTALL ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
            pip_install ${PIP_PACKAGES[@]}
    fi
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/opt/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                   pip_install -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                pip_install -r "${requirements}"
            fi
        fi
    done
}

function provisioning_get_default_workflow() {
    if [[ -n $DEFAULT_WORKFLOW ]]; then
        workflow_json=$(curl -s "$DEFAULT_WORKFLOW")
        if [[ -n $workflow_json ]]; then
            echo "export const defaultGraph = $workflow_json;" > /opt/ComfyUI/web/scripts/defaultGraph.js
        fi
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n     $PROVISIONING_SCRIPT_NAME      \n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
    if [[ $DISK_GB_ALLOCATED -lt $DISK_GB_REQUIRED ]]; then
        printf "WARNING: Your allocated disk size (%sGB) is below the recommended %sGB - Some models will not be downloaded\n" "$DISK_GB_ALLOCATED" "$DISK_GB_REQUIRED"
    fi
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Web UI will start now\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    url="https://huggingface.co/api/whoami-v2"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif 
        [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]];then
        wget --header="Authorization: Bearer $auth_token" -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

provisioning_start

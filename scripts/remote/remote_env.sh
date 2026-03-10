#!/usr/bin/env bash

set -euo pipefail

expand_remote_home_path() {
    local path="${1:-}"

    if [[ -z "${path}" ]]; then
        printf '%s/.cargo/env\n' "${HOME}"
    elif [[ "${path}" == '$HOME' ]]; then
        printf '%s\n' "${HOME}"
    elif [[ "${path}" == '$HOME/'* ]]; then
        printf '%s/%s\n' "${HOME}" "${path#\$HOME/}"
    elif [[ "${path}" == \~ ]]; then
        printf '%s\n' "${HOME}"
    elif [[ "${path}" == \~/* ]]; then
        printf '%s/%s\n' "${HOME}" "${path:2}"
    else
        printf '%s\n' "${path}"
    fi
}

load_remote_cargo_env() {
    local cargo_env_file="${1:-$HOME/.cargo/env}"
    local rustup_toolchain="${2:-}"

    cargo_env_file="$(expand_remote_home_path "${cargo_env_file}")"

    if [[ -f "${cargo_env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${cargo_env_file}"
    fi

    if [[ -n "${rustup_toolchain}" ]]; then
        export RUSTUP_TOOLCHAIN="${rustup_toolchain}"
    fi
}

# À sourcer : source gr-litevna/env.sh
# Rend les blocs LiteVNA visibles de GNU Radio Companion et de Python, sans installation.
_grl_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$_grl_dir:$(dirname "$_grl_dir")${PYTHONPATH:+:$PYTHONPATH}"
export GRC_BLOCKS_PATH="$_grl_dir/grc${GRC_BLOCKS_PATH:+:$GRC_BLOCKS_PATH}"
unset _grl_dir

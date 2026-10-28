# Keep this isolated package prefix ahead of other workspaces that also contain
# a package named "experiment".  ament_prepend_unique_value intentionally does
# not reorder an existing entry, so remove this prefix first and then prepend it.
# Package-level setup defines AMENT_CURRENT_PREFIX; colcon's workspace-level
# fast path invokes hooks directly and defines COLCON_CURRENT_PREFIX instead.
_neural_experiment_prefix="${COLCON_CURRENT_PREFIX:-${AMENT_CURRENT_PREFIX:-}}"
if [ -z "$_neural_experiment_prefix" ]; then
  return 0
fi
_neural_experiment_filtered=""
_neural_experiment_old_ifs="$IFS"
IFS=:
for _neural_experiment_entry in ${AMENT_PREFIX_PATH:-}; do
  if [ -n "$_neural_experiment_entry" ] && \
     [ "$_neural_experiment_entry" != "$_neural_experiment_prefix" ]; then
    if [ -n "$_neural_experiment_filtered" ]; then
      _neural_experiment_filtered="${_neural_experiment_filtered}:$_neural_experiment_entry"
    else
      _neural_experiment_filtered="$_neural_experiment_entry"
    fi
  fi
done
IFS="$_neural_experiment_old_ifs"

if [ -n "$_neural_experiment_filtered" ]; then
  export AMENT_PREFIX_PATH="${_neural_experiment_prefix}:${_neural_experiment_filtered}"
else
  export AMENT_PREFIX_PATH="$_neural_experiment_prefix"
fi

unset _neural_experiment_prefix
unset _neural_experiment_filtered
unset _neural_experiment_old_ifs
unset _neural_experiment_entry

#!/bin/bash

for f in *; do
  # if [[ "$f" == *"[HG]"* ]]; then
  if [[ "$f" == *'['\''HG'\'']'* ]]; then
    newname=$(echo "$f" | sed "s/^\['HG'\]_*/HG_/")
    echo "Would rename: $f → $newname"
    changes+=("$f:$newname")
  fi
done

if [[ ${#changes[@]} -eq 0 ]]; then
  echo "No files matched."
  exit 0
fi

read -p "Proceed with renaming? (Y/n) " response
response=${response,,}  # to lowercase

if [[ "$response" == "y" || "$response" == "" ]]; then
  for entry in "${changes[@]}"; do
    oldname="${entry%%:*}"
    newname="${entry##*:}"
    mv -- "$oldname" "$newname"
    echo "Renamed: $oldname → $newname"
  done
else
  echo "Aborted."
fi

# for f in *; do
#   if [[ "$f" == *'['\''HG'\'']'* ]]; then
#     # newname=$(echo "$f" | sed "s/^'['\''HG'\'']*'/HG/")
#     newname=$(echo "$f" | sed "s/^\['HG'\]_*/HG_/")
#     echo "Would rename: $f → $newname"
#   fi
# done

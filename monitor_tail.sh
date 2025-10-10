#!/bin/bash
if [ "$#" -eq 2 ]; then
    num_lines=$1
    file_name=$2
else
    num_lines=35
    file_name=$1
fi

escaped_file_name=$(printf "%q" "$file_name") # from chatgpt :o
watch -n 1 tail -n $num_lines "${escaped_file_name}"

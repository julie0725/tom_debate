#!/bin/bash
while IFS= read -r line; do
    echo "$line"
    # Section headers get longer pause
    if [[ "$line" == *"==="* ]] || [[ "$line" == *"###"* ]] || [[ "$line" == *"---"* ]]; then
        sleep 1.2
    elif [[ "$line" == "" ]]; then
        sleep 0.6
    else
        sleep 0.3
    fi
done < /mnt/c/Users/juyeo/Desktop/tom_debate/demo_output.txt

#!/bin/bash

if !command -v bc &> /dev/null; then
    echo "bc could not be found. Please install bc and try again."
    exit 1
fi

if !command -v shuf &> /dev/null; then
    echo "shuf could not be found. Please install shuf and try again."
    exit 1
fi

SRC_DIR="src"
BIN_DIR="bin"
SRC_FILE="fft.cu"
OUTPUT_FILE="fft"

mkdir -p "$BIN_DIR"

nvcc -O3 "$SRC_DIR/$SRC_FILE" -lcublas -UENABLE_DEBUG -o "$BIN_DIR/$OUTPUT_FILE"

echo -e "Compilation complete. Binary placed in $BIN_DIR/$OUTPUT_FILE\n"

generate_random_number() {
    local length=$1
    local num=""
    for ((i=0; i<length; i++)); do
        if [ $i -eq 0 ]; then
            num="${num}$(shuf -i 1-9 -n 1)"  # First digit cannot be zero
        else
            num="${num}$(shuf -i 0-9 -n 1)"
        fi
    done
    echo $num
}

for i in {1..5}
do
    if [ $i -lt 5 ]; then
        len1=$((RANDOM % (i * 5) + 1))
        len2=$((RANDOM % (i * 5) + 1))
    else
        len1=$((RANDOM % 49901 + 100))
        len2=$((RANDOM % 49901 + 100))
        echo "Multiplying two integers of $len1 and $len2 digits, respectively"
    fi
    
    num1=$(generate_random_number $len1)
    num2=$(generate_random_number $len2)
    
    bc_result=$(echo "$num1 * $num2" | bc | tr -d '\n' | tr -d ' ' | tr -d '\\')
    bc_result=$(echo "Product of $num1 and $num2 is $bc_result")
    cuda_result=$("$BIN_DIR/$OUTPUT_FILE" "$num1" "$num2")

    if [ "$bc_result" == "$cuda_result" ]; then
        echo $bc_result
    else
        echo "Mismatch: FFT output $cuda_result != Expected $bc_result"
        echo $bc_result
        echo $cuda_result
    fi

    echo ""
done
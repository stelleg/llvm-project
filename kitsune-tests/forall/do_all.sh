declare -a files=(

    # for loops
    basic_forall
    stdvector
    vector_iterator
    dual_vector_iterator

    # for range loops
    array_range
    vector_range

    # composite
    stdmap
)

# iterate over source files
for file in "${files[@]}"
do
    . cheatsheet.sh ${file}
    ./${file}
done
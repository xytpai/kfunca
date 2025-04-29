find ./src \( -name "*.h" -o -name "*.cpp" -o -name "*.cu" \) -exec clang-format -i {} +
find ./test \( -name "*.h" -o -name "*.cpp" -o -name "*.cu" \) -exec clang-format -i {} +

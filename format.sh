find ./src \( -name "*.h" -o -name "*.cpp" -o -name "*.cu" \) -exec clang-format -i {} +
find ./test/cpp \( -name "*.h" -o -name "*.cpp" -o -name "*.cu" \) -exec clang-format -i {} +

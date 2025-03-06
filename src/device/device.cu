#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "device.h"
#include "launcher.h"

void launcher_test() {
    auto l = Launcher::GetInstance();
    int count = l->device_count();
    std::cout << "device count: " << count << std::endl;
}

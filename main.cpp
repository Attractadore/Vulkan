#include <vulkan/vulkan.h>
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <stdexcept>
#include <iostream>
#include <optional>
#include <unordered_set>
#include <vector>

namespace util {
std::ostream& log = std::cout;
std::ostream& error_log = std::cerr;

#ifndef NDEBUG
    constexpr bool is_debug = true;
#else
    constexpr bool is_debug = false;
#endif

#define STRING(v) # v

const char* vulkanErrorString(VkResult res) {
    switch(res) {
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "extension not present";
    }
    error_log << "FIXME: Unknown VkResult\n";
    static std::string buffer;
    buffer = std::to_string(res);
    return buffer.c_str();
}

void error(VkResult res, const std::string& s) {
    if (res != VK_SUCCESS) {
        throw std::runtime_error(s + ": " + vulkanErrorString(res));
    }
}

template<typename I, typename U>
bool contains_if(I first, I last, U op) {
    return std::find_if(first, last, op) != last;
}
}

SDL_Window* createWindow(unsigned width, unsigned height) {
    return SDL_CreateWindow(
        "Vulkan tutorial",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        width, height,
        SDL_WINDOW_VULKAN
    );
}

bool shouldClose(SDL_Window* window) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_WINDOWEVENT and
            e.window.windowID == SDL_GetWindowID(window) and
            e.window.event == SDL_WINDOWEVENT_CLOSE) {
            return true;
        }
    }
    return false;
}

VkSurfaceKHR createSurface(SDL_Window* window, VkInstance instance) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    SDL_Vulkan_CreateSurface(window, instance, &surface);
    return surface;
}

namespace {
VKAPI_ATTR VkBool32 VKAPI_CALL debugErrorCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        util::error_log << "VULKAN ERROR: " << callback_data->pMessage << std::endl;
    }
    return VK_FALSE;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugInfoCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*) {
    util::error_log << "VULKAN INFO: " << callback_data->pMessage << std::endl;
    return VK_FALSE;
}
}

VkDebugUtilsMessengerCreateInfoEXT getDebugMessengerDefaultCreateInfo() {
    return {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugErrorCallback,
        .pUserData = nullptr,
    };
}

VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance) {
    auto f = (PFN_vkCreateDebugUtilsMessengerEXT) (
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")
    );
    auto err_str = "Failed to create Vulkan debug messenger";
    if (!f) {
        util::error(VK_ERROR_EXTENSION_NOT_PRESENT, err_str);
    }

    VkDebugUtilsMessengerCreateInfoEXT create_info = 
        getDebugMessengerDefaultCreateInfo();
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    f(instance, &create_info, nullptr, &messenger);

    return messenger;
}

void destroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger) {
    auto f = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, STRING(vkDestroyDebugUtilsMessengerEXT))
    );
    if (f) {
        f(instance, messenger, nullptr);
    }
}

template<typename I>
bool checkAvailableLayers(I first, I last) {
    unsigned num_layers = 0;
    vkEnumerateInstanceLayerProperties(&num_layers, nullptr);
    std::vector<VkLayerProperties> layers{num_layers};
    vkEnumerateInstanceLayerProperties(&num_layers, layers.data());
    return std::all_of(
        first, last,
        [&](const auto& layer_name) {
            return util::contains_if(
                layers.begin(), layers.end(),
                [&] (const auto& layer) { return std::strcmp(layer.layerName, layer_name) == 0; }
            );
        }
    );
}

VkInstance createInstance(SDL_Window* window) {
    const std::array<const char*, 1> validation_layers = {"VK_LAYER_KHRONOS_validation"};
    auto validation_layer_count =
    [&]() -> unsigned {
        if (util::is_debug) {
            for (const auto& l : validation_layers) {
                util::log << "Request layer " << l << "\n";
            }
            if(!checkAvailableLayers(validation_layers.begin(), validation_layers.end())) {
                util::error_log << "Not all requested layers are available!\n";
                return 0;
            }
            else {
                return validation_layers.size();
            }
        }
        else {
            return 0;
        }
    }();

    unsigned num_extensions = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &num_extensions, nullptr);
    std::vector<const char*> extensions{num_extensions};
    SDL_Vulkan_GetInstanceExtensions(window, &num_extensions, extensions.data());
    if (validation_layer_count) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    for (const auto& e: extensions) {
        util::log << "Require extension " << e << "\n";
    }

    VkDebugUtilsMessengerCreateInfoEXT debug_messenger_create_info = 
        getDebugMessengerDefaultCreateInfo();
    VkInstanceCreateInfo create_info {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = util::is_debug ? &debug_messenger_create_info : nullptr,
        .flags = 0,
        .pApplicationInfo = nullptr,
        .enabledLayerCount = validation_layer_count,
        .ppEnabledLayerNames = validation_layers.data(),
        .enabledExtensionCount = extensions.size(),
        .ppEnabledExtensionNames = extensions.data(),
    };
    VkInstance instance = VK_NULL_HANDLE;
    vkCreateInstance(&create_info, nullptr, &instance);

    return instance;
}

unsigned gpuScore(VkPhysicalDevice gpu) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    return props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
}

std::string gpuName(VkPhysicalDevice gpu) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    return props.deviceName;
}

VkPhysicalDevice selectGPU(VkInstance instance) {
    unsigned num_gpus = 0;
    vkEnumeratePhysicalDevices(instance, &num_gpus, nullptr);
    if (!num_gpus) {
        throw std::runtime_error("No GPUs with Vulkan support found");
    }
    std::vector<VkPhysicalDevice> gpus{num_gpus};
    vkEnumeratePhysicalDevices(instance, &num_gpus, gpus.data());
    util::log << "Available GPUs:\n";
    for (const auto& gpu: gpus) {
        util::log << gpuName(gpu) << "\n";
    }

    auto it = std::max_element(gpus.begin(), gpus.end(), [&](const auto& lhs, const auto& rhs) {return gpuScore(lhs) > gpuScore(rhs);});
    auto gpu = *it;
    util::log << "Select " << gpuName(gpu) << "\n";

    return gpu;
}

struct QueueFamilyIndices {
    std::optional<unsigned> graphics;
    std::optional<unsigned> present;
};

std::optional<unsigned> findGraphicsQueueFamily(VkPhysicalDevice gpu, const std::vector<VkQueueFamilyProperties>& queues) {
    for (unsigned i = 0; i < queues.size(); i++) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            util::log << "Found graphics queue at index " << i << "\n";
            return i;
        }
    }
    return {};
}

std::optional<unsigned> findPresentQueueFamily(
    VkPhysicalDevice gpu, VkSurfaceKHR surface, const std::vector<VkQueueFamilyProperties>& queues) {
    for (unsigned i = 0; i < queues.size(); i++) {
        VkBool32 has_present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &has_present);
        if (has_present) {
            util::log << "Found present queue at index " << i << "\n";
            return i;
        }
    }
    return {};
}

QueueFamilyIndices getGPUQueueFamilies(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    unsigned num_queues = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queues, nullptr);
    std::vector<VkQueueFamilyProperties> queues{num_queues};
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queues, queues.data());

    return {
        .graphics = findGraphicsQueueFamily(gpu, queues),
        .present = findPresentQueueFamily(gpu, surface, queues),
    };
}

VkDevice createDevice(VkPhysicalDevice gpu, const QueueFamilyIndices& queue_families) {
    float priority = 1.0f;
    std::unordered_set<unsigned> queue_family_indices =
        {queue_families.graphics.value(), queue_families.present.value()};
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    for (const auto& i : queue_family_indices) {
        VkDeviceQueueCreateInfo queue_create_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = i,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        queue_create_infos.push_back(queue_create_info);
    }
    VkDeviceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueCreateInfoCount = queue_create_infos.size(),
        .pQueueCreateInfos = queue_create_infos.data(),
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
        .pEnabledFeatures = nullptr,
    };
    VkDevice device = VK_NULL_HANDLE;
    vkCreateDevice(gpu, &create_info, nullptr, &device);
    return device;
}

struct Queues {
    VkQueue graphics;  
    VkQueue present;  
};

Queues getDeviceQueues(VkDevice device, const QueueFamilyIndices& queue_families) {
    Queues queues;
    vkGetDeviceQueue(device, queue_families.graphics.value(), 0, &queues.graphics);
    vkGetDeviceQueue(device, queue_families.present.value(), 0, &queues.present);
    return queues;
}

int main() {
    SDL_SetMainReady();
    SDL_Init(SDL_INIT_EVERYTHING);

    auto window = createWindow(1280, 720);
    auto instance = createInstance(window);
    auto surface = createSurface(window, instance);
    auto debug_messenger = 
        util::is_debug ? createDebugMessenger(instance) : VK_NULL_HANDLE;
    auto gpu = selectGPU(instance);
    auto queue_families = getGPUQueueFamilies(gpu, surface);
    auto device = createDevice(gpu, queue_families);
    auto queues = getDeviceQueues(device, queue_families);

    while(!shouldClose(window)) {
        SDL_UpdateWindowSurface(window);
    }

    vkDestroyDevice(device, nullptr);
    destroyDebugMessenger(instance, debug_messenger);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    SDL_DestroyWindow(window);

    SDL_Quit();
}

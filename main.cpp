#include <vulkan/vulkan.h>
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
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

#define STRING(v) #v

const char* vulkanErrorString(VkResult res) {
    switch (res) {
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "extension not present";
        default: {
            error_log << "FIXME: Unknown VkResult\n";
            static std::string buffer;
            buffer = std::to_string(res);
            return buffer.c_str();
        }
    }
}

void error(VkResult res, const std::string& s) {
    if (res != VK_SUCCESS) {
        throw std::runtime_error(s + ": " + vulkanErrorString(res));
    }
}

template <typename I, typename U>
bool contains_if(I first, I last, U op) {
    return std::find_if(first, last, op) != last;
}

std::vector<std::byte> loadBinaryFile(const std::string& path) {
    std::ifstream f{path, std::ios::binary};   
    if (!f) {
        return {};
    }
    auto size = std::filesystem::file_size(path);
    std::vector<std::byte> data{size};
    f.read(reinterpret_cast<char*>(data.data()), size);
    if (!f) {
        return {};
    }
    return data;
}
}  // namespace util

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
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        util::error_log << "VULKAN ERROR: " << callback_data->pMessage << std::endl;
    }
    return VK_FALSE;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugInfoCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*)
{
    util::error_log << "VULKAN INFO: " << callback_data->pMessage << std::endl;
    return VK_FALSE;
}
}  // namespace

VkDebugUtilsMessengerCreateInfoEXT getDebugMessengerDefaultCreateInfo() {
    return {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugErrorCallback,
        .pUserData = nullptr,
    };
}

VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance) {
    auto f = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, STRING(vkCreateDebugUtilsMessengerEXT))
    );
    if (!f) {
        util::error(
            VK_ERROR_EXTENSION_NOT_PRESENT,
            "Failed to create Vulkan debug messenger"
        );
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

template <typename S>
bool instanceLayersAvailable(const S& layers) {
    unsigned num_instance_layers = 0;
    vkEnumerateInstanceLayerProperties(&num_instance_layers, nullptr);
    std::vector<VkLayerProperties> instance_layers{num_instance_layers};
    vkEnumerateInstanceLayerProperties(&num_instance_layers, instance_layers.data());
    return std::all_of(
        std::begin(layers), std::end(layers),
        [&](const auto& layer_name) {
            return util::contains_if(
                instance_layers.begin(), instance_layers.end(),
                [&](const auto& instance_layer) {
                    return std::strcmp(instance_layer.layerName, layer_name) == 0;
                }
            );
        }
    );
}

VkInstance createInstance(SDL_Window* window) {
    const std::array<const char*, 1> validation_layers = {"VK_LAYER_KHRONOS_validation"};
    auto validation_layer_count =
    [&]() -> unsigned {
        if (util::is_debug) {
            for (const auto& l: validation_layers) {
                util::log << "Request layer " << l << "\n";
            }
            if (!instanceLayersAvailable(validation_layers)) {
                util::error_log << "Not all requested layers are available!\n";
                return 0;
            } else {
                return validation_layers.size();
            }
        } else {
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
        util::log << "Require instance extension " << e << "\n";
    }

    VkDebugUtilsMessengerCreateInfoEXT debug_messenger_create_info =
        getDebugMessengerDefaultCreateInfo();
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = util::is_debug ? &debug_messenger_create_info : nullptr,
        .flags = 0,
        .pApplicationInfo = nullptr,
        .enabledLayerCount = validation_layer_count,
        .ppEnabledLayerNames = validation_layers.data(),
        .enabledExtensionCount = static_cast<unsigned>(extensions.size()),
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

template <typename S>
bool gpuExtensionsSupported(VkPhysicalDevice gpu, const S& extensions) {
    unsigned num_gpu_extensions = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &num_gpu_extensions, nullptr);
    std::vector<VkExtensionProperties> gpu_extensions(num_gpu_extensions);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &num_gpu_extensions, gpu_extensions.data());

    return std::all_of(
        std::begin(extensions), std::begin(extensions),
        [&](const auto& extension) {
            return util::contains_if(
                gpu_extensions.begin(), gpu_extensions.end(),
                [&](const auto& gpu_extension) {
                    return std::strcmp(gpu_extension.extensionName, extension) == 0;
                }
            );
        }
    );
}

struct SwapchainInfo {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

SwapchainInfo getSwapchainInfo(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    SwapchainInfo info{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &info.capabilities);
    unsigned num_formats = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &num_formats, nullptr);
    info.formats.resize(num_formats);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &num_formats, info.formats.data());
    unsigned num_present_modes = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &num_present_modes, nullptr);
    info.present_modes.resize(num_present_modes);
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &num_present_modes, info.present_modes.data());
    return info;
}

bool gpuSwapchainSupported(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    auto info = getSwapchainInfo(gpu, surface);
    return info.formats.size() and info.present_modes.size();
}

struct QueueFamilyIndices {
    std::optional<unsigned> graphics;
    std::optional<unsigned> present;

    explicit operator bool() const { return graphics and present; }
};

std::optional<unsigned> findGraphicsQueueFamily(
    VkPhysicalDevice gpu,
    const std::vector<VkQueueFamilyProperties>& queues)
{
    for (unsigned i = 0; i < queues.size(); i++) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            return i;
        }
    }
    return {};
}

std::optional<unsigned> findPresentQueueFamily(
    VkPhysicalDevice gpu, VkSurfaceKHR surface,
    const std::vector<VkQueueFamilyProperties>& queues)
{
    for (unsigned i = 0; i < queues.size(); i++) {
        VkBool32 has_present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &has_present);
        if (has_present) {
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

VkPhysicalDevice selectGPU(VkInstance instance, VkSurfaceKHR surface) {
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

    std::array<const char*, 1> required_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    auto new_end = std::remove_if(
        gpus.begin(), gpus.end(),
        [&](const auto& gpu) {
            return !(gpuExtensionsSupported(gpu, required_extensions) and
                     gpuSwapchainSupported(gpu, surface) and
                     getGPUQueueFamilies(gpu, surface));
        }
    );
    gpus.resize(std::distance(gpus.begin(), new_end));
    if (gpus.empty()) {
        util::log << "No suitable GPUs available\n";
        return VK_NULL_HANDLE;
    }

    auto it = std::max_element(
        gpus.begin(), gpus.end(),
        [&](const auto& lhs, const auto& rhs) { return gpuScore(lhs) > gpuScore(rhs); }
    );
    auto gpu = *it;
    util::log << "Select " << gpuName(gpu) << "\n";

    return gpu;
}

VkDevice createDevice(VkPhysicalDevice gpu, const QueueFamilyIndices& queue_families) {
    float priority = 1.0f;
    std::unordered_set<unsigned> queue_family_indices =
        {queue_families.graphics.value(), queue_families.present.value()};
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    for (const auto& i: queue_family_indices) {
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
    std::array<const char*, 1> extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    for (const auto& e: extensions) {
        util::log << "Require device extension " << e << "\n";
    }
    VkDeviceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueCreateInfoCount = static_cast<unsigned>(queue_create_infos.size()),
        .pQueueCreateInfos = queue_create_infos.data(),
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<unsigned>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
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

VkSurfaceFormatKHR
selectSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f: formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB and
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.front();
}

VkPresentModeKHR
selectPresentMode(const std::vector<VkPresentModeKHR>& present_modes) {
    for (const auto& pm: present_modes) {
        if (pm == VK_PRESENT_MODE_MAILBOX_KHR) {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D selectSwapchainExtent(SDL_Window* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    VkExtent2D special_value = {
        .width = std::numeric_limits<decltype(special_value.width)>::max(),
        .height = std::numeric_limits<decltype(special_value.height)>::max(),
    };
    if (capabilities.currentExtent.width == special_value.width and
        capabilities.currentExtent.height == special_value.height) {
        int w = 0;
        int h = 0;
        SDL_Vulkan_GetDrawableSize(window, &w, &h);
        return {
            .width = std::clamp<unsigned>(
                w, capabilities.minImageExtent.width, capabilities.maxImageExtent.width
            ),
            .height = std::clamp<unsigned>(
                h, capabilities.minImageExtent.height, capabilities.maxImageExtent.height
            ),
        };
    } else {
        return capabilities.currentExtent;
    }
}

VkSwapchainKHR createSwapchain(
    SDL_Window* window, VkSurfaceKHR surface,
    VkPhysicalDevice gpu, VkDevice device,
    const QueueFamilyIndices& queue_families)
{
    auto info = getSwapchainInfo(gpu, surface);
    auto format = selectSurfaceFormat(info.formats);
    auto present_mode = selectPresentMode(info.present_modes);
    auto extent = selectSwapchainExtent(window, info.capabilities);
    unsigned image_count = info.capabilities.minImageCount + 1;
    if (info.capabilities.maxImageCount) {
        image_count = std::min(image_count, info.capabilities.maxImageCount);
    }

    bool exclusive =
        queue_families.graphics.value() == queue_families.present.value();
    std::array<unsigned, 2> queues =
        {queue_families.graphics.value(), queue_families.present.value()};

    VkSwapchainCreateInfoKHR create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode =
            exclusive ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount =
            exclusive ? 0 : static_cast<unsigned>(queues.size()),
        .pQueueFamilyIndices = exclusive ? nullptr : queues.data(),
        .preTransform = info.capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain);

    return swapchain;
}

std::vector<VkImage> getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain) {
    unsigned num_images = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &num_images, nullptr);
    std::vector<VkImage> images{num_images};
    vkGetSwapchainImagesKHR(device, swapchain, &num_images, images.data());
    return images;
}

std::vector<VkImageView> createImageViews(VkDevice device, VkFormat format, const std::vector<VkImage> images) {
    std::vector<VkImageView> views{images.size()};
    for (unsigned i = 0; i < images.size(); i++) {
        VkImageViewCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        vkCreateImageView(device, &create_info, nullptr, &views[i]);
    }
    return views;
}

VkPipelineLayout createPipelineLayout(VkDevice device) {
    VkPipelineLayoutCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &create_info, nullptr, &layout);
    return layout;
}

VkRenderPass createRenderPass(VkDevice device, VkFormat format) {
    VkAttachmentDescription attachment_description = {
        .flags = 0,
        .format = format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    VkAttachmentReference attachment_reference = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass = {
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment_reference,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };

    VkRenderPassCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &attachment_description,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr,
    };

    VkRenderPass render_pass = VK_NULL_HANDLE;
    vkCreateRenderPass(device, &create_info, nullptr, &render_pass);

    return render_pass;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<std::byte>& binary) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = binary.size(),
        .pCode = reinterpret_cast<decltype(create_info.pCode)>(binary.data()),
    };
    VkShaderModule shader_module;
    vkCreateShaderModule(device, &create_info, nullptr, &shader_module);
    return shader_module;
}

VkPipelineShaderStageCreateInfo getPipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, VkShaderModule shader_module)
{
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stage,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };
}

VkPipeline createPipeline(
    VkDevice device, VkExtent2D swapchain_extent,
    VkPipelineLayout layout, VkRenderPass render_pass) 
{
    constexpr auto vert_name = "shader.vert.spv";
    constexpr auto frag_name = "shader.frag.spv";
    auto vertBinary = util::loadBinaryFile(vert_name);
    if (vertBinary.empty()) {
        util::error_log << "Failed to load binary from " << vert_name << "\n";
        return VK_NULL_HANDLE;
    }
    auto fragBinary = util::loadBinaryFile(frag_name);
    if (fragBinary.empty()) {
        util::error_log << "Failed to load binary from " << frag_name << "\n";
        return VK_NULL_HANDLE;
    }

    auto vertModule = createShaderModule(device, vertBinary);
    auto fragModule = createShaderModule(device, fragBinary);

    std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {
        getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertModule),
        getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragModule),
    };

    VkPipelineVertexInputStateCreateInfo vertex_input_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = false,
    };

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = float(swapchain_extent.width),
        .height = float(swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor = {
        .offset = {.x = 0, .y = 0},
        .extent = swapchain_extent,
    };

    VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = false,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .blendEnable = false,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    VkGraphicsPipelineCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = shader_stages.size(),
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pTessellationState = nullptr,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState = &multisample_state,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &color_blend_state,
        .pDynamicState = nullptr,
        .layout = layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create_info, nullptr, &pipeline);

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);

    return pipeline;
};

int main() {
    SDL_SetMainReady();
    SDL_Init(SDL_INIT_EVERYTHING);

    auto window = createWindow(1280, 720);
    auto instance = createInstance(window);
    auto surface = createSurface(window, instance);
    auto debug_messenger =
        util::is_debug ? createDebugMessenger(instance) : VK_NULL_HANDLE;
    auto gpu = selectGPU(instance, surface);
    auto swapchain_info = getSwapchainInfo(gpu, surface);
    auto surface_format = selectSurfaceFormat(swapchain_info.formats).format;
    auto swapchain_extent = selectSwapchainExtent(window, swapchain_info.capabilities);
    auto queue_families = getGPUQueueFamilies(gpu, surface);
    auto device = createDevice(gpu, queue_families);
    auto queues = getDeviceQueues(device, queue_families);
    auto swapchain =
        createSwapchain(window, surface, gpu, device, queue_families);
    auto images = getSwapchainImages(device, swapchain);
    auto views = createImageViews(device, surface_format, images);
    auto pipeline_layout = createPipelineLayout(device);
    auto render_pass = createRenderPass(device, surface_format);
    auto pipeline = createPipeline(device, swapchain_extent, pipeline_layout, render_pass);

    while (!shouldClose(window)) {
        SDL_UpdateWindowSurface(window);
    }

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    for (auto& view: views) {
        vkDestroyImageView(device, view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);
    destroyDebugMessenger(instance, debug_messenger);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    SDL_DestroyWindow(window);

    SDL_Quit();
}

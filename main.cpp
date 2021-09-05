#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
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

VkResult createDebugMessenger(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger
) {
    auto f = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, STRING(vkCreateDebugUtilsMessengerEXT))
    );
    if (f) {
        return f(instance, pCreateInfo, pAllocator, pMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void destroyDebugMessenger(
    VkInstance instance,
    VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks* pAllocator
) {
    auto f = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, STRING(vkDestroyDebugUtilsMessengerEXT))
    );
    if (f) {
        f(instance, messenger, pAllocator);
    }
}

struct VulkanInstance {
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger;
};

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

std::vector<const char*> getInstanceLayers() {
    std::vector<const char*> layers;
    if constexpr (util::is_debug) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    return layers;
}

std::vector<const char*> getInstanceExtensions() {
    unsigned num_extensions = 0;
    const char** extensions_array = glfwGetRequiredInstanceExtensions(&num_extensions);
    std::vector<const char*> extensions{extensions_array, extensions_array + num_extensions};
    if constexpr (util::is_debug) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return extensions;
}

VulkanInstance createVulkanInstance() {
    VulkanInstance vi{};

    auto layers = getInstanceLayers();
    auto extensions = getInstanceExtensions();
    for (const auto& l: layers) {
        util::log << "Request layer " << l << "\n";
    }
    for (const auto& e: extensions) {
        util::log << "Require instance extension " << e << "\n";
    }

    VkDebugUtilsMessengerCreateInfoEXT messenger_create_info = {
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

    VkInstanceCreateInfo instance_create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = util::is_debug ? &messenger_create_info : nullptr,
        .flags = 0,
        .pApplicationInfo = nullptr,
        .enabledLayerCount =
            static_cast<decltype(instance_create_info.enabledLayerCount)>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount =
            static_cast<decltype(instance_create_info.enabledExtensionCount)>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    vkCreateInstance(&instance_create_info, nullptr, &vi.instance);
    if constexpr (util::is_debug) {
        createDebugMessenger(vi.instance, &messenger_create_info, nullptr, &vi.messenger);
    }

    return vi;
}

void destroyVulkanInstance(VulkanInstance& vi) {
    if constexpr (util::is_debug) {
        destroyDebugMessenger(vi.instance, vi.messenger, nullptr);
    }
    vkDestroyInstance(vi.instance, nullptr);
}

struct VulkanWindow {
    GLFWwindow* window;
    VkSurfaceKHR surface;
};

VulkanWindow createVulkanWindow(VkInstance instance, unsigned width, unsigned height) {
    VulkanWindow vw{};
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, false);
    vw.window = glfwCreateWindow(width, height, "Vulkan Tutorial", nullptr, nullptr);
    glfwCreateWindowSurface(instance, vw.window, nullptr, &vw.surface);
    return vw;
}

bool shouldClose(VulkanWindow& vw) {
    glfwPollEvents();
    return glfwWindowShouldClose(vw.window);
}

void destroyVulkanWindow(VkInstance instance, VulkanWindow& vw) {
    vkDestroySurfaceKHR(instance, vw.surface, nullptr);
    glfwDestroyWindow(vw.window);
}

std::vector<const char*> getGPUExtensions() {
    return {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
}

struct QueueFamilies {
    std::optional<unsigned> graphics;
    std::optional<unsigned> present;

    explicit operator bool() const {
        return graphics and present;
    }
};

std::optional<unsigned> getGPUGraphicsQueueFamily(
    VkPhysicalDevice gpu,
    const std::vector<VkQueueFamilyProperties>& queues
) {
    for (unsigned i = 0; i < queues.size(); i++) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            return i;
        }
    }
    return {};
}

std::optional<unsigned> getGPUPresentQueueFamily(
    VkPhysicalDevice gpu, VkSurfaceKHR surface,
    const std::vector<VkQueueFamilyProperties>& queues
) {
    for (unsigned i = 0; i < queues.size(); i++) {
        VkBool32 has_present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &has_present);
        if (has_present) {
            return i;
        }
    }
    return {};
}

QueueFamilies getGPUQueueFamilies(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    unsigned num_queues = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queues, nullptr);
    std::vector<VkQueueFamilyProperties> queues{num_queues};
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queues, queues.data());

    return {
        .graphics = getGPUGraphicsQueueFamily(gpu, queues),
        .present = getGPUPresentQueueFamily(gpu, surface, queues),
    };
}

struct SurfaceInfo {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

SurfaceInfo getGPUSurfaceInfo(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    SurfaceInfo info{};

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

bool gpuExtensionsSupported(VkPhysicalDevice gpu) {
    auto required = getGPUExtensions();

    unsigned num_supported = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &num_supported, nullptr);
    std::vector<VkExtensionProperties> supported(num_supported);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &num_supported, supported.data());

    return std::all_of(
        required.begin(), required.end(),
        [&](const auto& r) {
            return util::contains_if(
                supported.begin(), supported.end(),
                [&](const auto& s) {
                    return std::strcmp(s.extensionName, r) == 0;
                }
            );
        }
    );
}

bool gpuSurfaceSupported(VkPhysicalDevice gpu, const SurfaceInfo& surface_info) {
    return surface_info.formats.size() and surface_info.present_modes.size();
}

struct VulkanGPU {
    VkPhysicalDevice gpu;
    QueueFamilies queue_families;
    SurfaceInfo surface_info;
};

std::vector<VulkanGPU> getInstanceGPUs(VkInstance instance, VkSurfaceKHR surface) {
    unsigned num_gpus = 0;
    vkEnumeratePhysicalDevices(instance, &num_gpus, nullptr);
    std::vector<VkPhysicalDevice> gpus(num_gpus);
    vkEnumeratePhysicalDevices(instance, &num_gpus, gpus.data());

    std::vector<VulkanGPU> vgpus(gpus.size());
    for (unsigned i = 0; i < gpus.size(); i++) {
        vgpus[i] = {
            .gpu = gpus[i],
            .queue_families = getGPUQueueFamilies(gpus[i], surface),
            .surface_info = getGPUSurfaceInfo(gpus[i], surface),
        };
    }

    return vgpus;
}

unsigned gpuScore(VulkanGPU gpu) {
    if (gpuExtensionsSupported(gpu.gpu) and
        gpuSurfaceSupported(gpu.gpu, gpu.surface_info) and
        gpu.queue_families) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpu.gpu, &props);
        return (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 2 : 1;
    }
    return 0;
}

std::string gpuName(VkPhysicalDevice gpu) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    return props.deviceName;
}

VulkanGPU selectVulkanGPU(VkInstance instance, VkSurfaceKHR surface) {
    auto gpus = getInstanceGPUs(instance, surface);
    auto gpu = *std::max_element(
        gpus.begin(), gpus.end(),
        [&](const auto& lhs, const auto& rhs) {
            return gpuScore(lhs) < gpuScore(rhs);
        }
    );

    util::log << "Available GPUs:\n";
    for (const auto& gpu: gpus) {
        util::log << gpuName(gpu.gpu) << "\n";
    }
    util::log << "Select " << gpuName(gpu.gpu) << "\n";

    return gpu;
}

VkDevice createDevice(VkPhysicalDevice gpu, const QueueFamilies& queue_families) {
    float priority = 1.0f;
    std::vector<unsigned> queue_family_indices =
        {queue_families.graphics.value(), queue_families.present.value()};
    std::sort(queue_family_indices.begin(), queue_family_indices.end());
    {
        auto new_end = std::unique(queue_family_indices.begin(), queue_family_indices.end());
        auto new_size = std::distance(queue_family_indices.begin(), new_end);
        queue_family_indices.resize(new_size);
    }
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

Queues getDeviceQueues(VkDevice device, const QueueFamilies& queue_families) {
    Queues queues;
    vkGetDeviceQueue(device, queue_families.graphics.value(), 0, &queues.graphics);
    vkGetDeviceQueue(device, queue_families.present.value(), 0, &queues.present);
    return queues;
}

VkSurfaceFormatKHR selectSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f: formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB and
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.front();
}

VkPresentModeKHR selectPresentMode(const std::vector<VkPresentModeKHR>& present_modes) {
    for (const auto& pm: present_modes) {
        if (pm == VK_PRESENT_MODE_MAILBOX_KHR) {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D selectSwapchainExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    VkExtent2D special_value = {
        .width = std::numeric_limits<decltype(special_value.width)>::max(),
        .height = std::numeric_limits<decltype(special_value.height)>::max(),
    };
    if (capabilities.currentExtent.width == special_value.width and
        capabilities.currentExtent.height == special_value.height) {
        int w = 0;
        int h = 0;
        glfwGetFramebufferSize(window, &w, &h);
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
    GLFWwindow* window, VkSurfaceKHR surface,
    VkPhysicalDevice gpu, VkDevice device,
    const QueueFamilies& queue_families)
{
    auto info = getGPUSurfaceInfo(gpu, surface);
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

    VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };

    VkRenderPassCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &attachment_description,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
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
    VkRenderPass render_pass)
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

    auto layout = createPipelineLayout(device);

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

    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);

    return pipeline;
};

std::vector<VkFramebuffer> createSwapchainFramebuffers(
    VkDevice device, VkRenderPass render_pass,
    const std::vector<VkImageView> views, VkExtent2D swapchain_extent)
{
    std::vector<VkFramebuffer> framebuffers{views.size()};
    for (unsigned i = 0; i < views.size(); i++) {
        VkFramebufferCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &views[i],
            .width = swapchain_extent.width,
            .height = swapchain_extent.height,
            .layers = 1,
        };
        vkCreateFramebuffer(device, &create_info, nullptr, &framebuffers[i]);
    }
    return framebuffers;
}

VkCommandPool createCommandPool(VkDevice device, unsigned graphics_queue) {
    VkCommandPoolCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = graphics_queue,
    };
    VkCommandPool command_pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &create_info, nullptr, &command_pool);
    return command_pool;
}

std::vector<VkCommandBuffer> allocateCommandBuffers(
    VkDevice device, VkCommandPool command_pool, unsigned num_command_buffers)
{
    VkCommandBufferAllocateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = num_command_buffers,
    };
    std::vector<VkCommandBuffer> command_buffers{num_command_buffers};
    vkAllocateCommandBuffers(device, &create_info, command_buffers.data());
    return command_buffers;
}

void recordCommandBuffers(
    const std::vector<VkCommandBuffer>& command_buffers, VkRenderPass render_pass,
    const std::vector<VkFramebuffer> framebuffers, VkExtent2D swapchain_extent,
    VkPipeline pipeline)
{
    for (unsigned i = 0; i < command_buffers.size(); i++) {
        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pInheritanceInfo = nullptr,
        };
        vkBeginCommandBuffer(command_buffers[i], &begin_info);

        VkClearValue clear_color = {
            .color = {.float32{0.0f, 0.0f, 0.0f, 1.0f}}
        };
        VkRenderPassBeginInfo render_pass_begin_info = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = render_pass,
            .framebuffer = framebuffers[i],
            .renderArea = {
                .offset = {.x = 0, .y = 0},
                .extent = swapchain_extent,
            },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };
        vkCmdBeginRenderPass(command_buffers[i], &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdDraw(command_buffers[i], 3, 1, 0, 0);
        vkCmdEndRenderPass(command_buffers[i]);

        vkEndCommandBuffer(command_buffers[i]);
    }
}

VkSemaphore createSemaphore(VkDevice device) {
    VkSemaphoreCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    VkSemaphore semaphore = VK_NULL_HANDLE;
    vkCreateSemaphore(device, &create_info, nullptr, &semaphore);
    return semaphore;
}

std::vector<VkSemaphore> createSemaphores(VkDevice device, unsigned num_semaphores) {
    std::vector<VkSemaphore> semaphores{num_semaphores};
    std::generate(semaphores.begin(), semaphores.end(), [&]() { return createSemaphore(device); });
    return semaphores;
}

void destroySemaphores(VkDevice device, std::vector<VkSemaphore>& semaphores) {
    for (auto& semaphore: semaphores) {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    semaphores.clear();
}

VkFence createFence(VkDevice device) {
    VkFenceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    VkFence fence;
    vkCreateFence(device, &create_info, nullptr, &fence);
    return fence;
};

void destroyFences(VkDevice device, std::vector<VkFence>& fences) {
    for (auto& fence: fences) {
        vkDestroyFence(device, fence, nullptr);
    }
    fences.clear();
}

std::vector<VkFence> createFences(VkDevice device, unsigned num_fences) {
    std::vector<VkFence> fences{num_fences};
    std::generate(fences.begin(), fences.end(), [&]() { return createFence(device); } );
    return fences;
};

void submitDraw(
    VkQueue queue, VkCommandBuffer command_buffer,
    VkSemaphore wait_semaphore, VkSemaphore signal_semaphore,
    VkFence signal_fence)
{
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphore,
    };
    vkQueueSubmit(queue, 1, &submit_info, signal_fence);
}

void submitPresent(VkQueue queue, VkSwapchainKHR swapchain, unsigned image, VkSemaphore wait_semaphore) {
    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image,
        .pResults = nullptr,
    };
    vkQueuePresentKHR(queue, &present_info);
}

void mainLoop(
    VulkanWindow vw, VkDevice device,
    VkSwapchainKHR swapchain, const Queues& queues,
    const std::vector<VkCommandBuffer>& command_buffers)
{
    auto num_swapchain_images = command_buffers.size();
    auto acquire_semaphores = createSemaphores(device, num_swapchain_images);
    auto render_semaphores = createSemaphores(device, num_swapchain_images);
    // Fences are used so that no more frames are being rendered at the same
    // time than there are swapchain images.
    auto frame_fences = createFences(device, num_swapchain_images);
    // The swapchain might return an image whose index differs from the current
    // frame's. In this case waiting on the current frame's fence will not be
    // sufficient, as the image might still be accessed by another frame's commands.
    // Store the fence of the last frame that rendered to an image to wait
    // for all accesses to be done.
    std::vector<VkFence> image_fences{num_swapchain_images, VK_NULL_HANDLE};

    unsigned frame = 0;
    while (!shouldClose(vw)) {
        constexpr auto uint64_t_max = std::numeric_limits<uint64_t>::max();

        auto& acquire_semaphore = acquire_semaphores[frame];
        auto& render_semaphore = render_semaphores[frame];
        auto& frame_fence = frame_fences[frame];

        unsigned image = 0;
        vkAcquireNextImageKHR(device, swapchain, uint64_t_max, acquire_semaphore, VK_NULL_HANDLE, &image);

        auto& image_fence = image_fences[image];
        auto& command_buffer = command_buffers[image];

        if (image_fence != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &image_fence, true, uint64_t_max);
        }
        image_fence = frame_fence;
        vkWaitForFences(device, 1, &frame_fence, true, uint64_t_max);
        vkResetFences(device, 1, &frame_fence);

        submitDraw(queues.graphics, command_buffer, acquire_semaphore, render_semaphore, frame_fence);
        submitPresent(queues.present, swapchain, image, render_semaphore);
        glfwSwapBuffers(vw.window);

        frame = (frame + 1) % num_swapchain_images;
    }
    vkDeviceWaitIdle(device);

    destroyFences(device, frame_fences);
    destroySemaphores(device, render_semaphores);
    destroySemaphores(device, acquire_semaphores);
}

void run() {
    auto vi = createVulkanInstance();
    auto vw = createVulkanWindow(vi.instance, 1280, 720);
    auto gpu = selectVulkanGPU(vi.instance, vw.surface);

    auto device = createDevice(gpu.gpu, gpu.queue_families);
    auto queues = getDeviceQueues(device, gpu.queue_families);

    auto swapchain_info = getGPUSurfaceInfo(gpu.gpu, vw.surface);
    auto surface_format = selectSurfaceFormat(swapchain_info.formats).format;
    auto swapchain_extent = selectSwapchainExtent(vw.window, swapchain_info.capabilities);
    auto swapchain =
        createSwapchain(vw.window, vw.surface, gpu.gpu, device, gpu.queue_families);
    auto images = getSwapchainImages(device, swapchain);

    auto render_pass = createRenderPass(device, surface_format);
    auto pipeline = createPipeline(device, swapchain_extent, render_pass);

    auto views = createImageViews(device, surface_format, images);
    auto framebuffers = createSwapchainFramebuffers(device, render_pass, views, swapchain_extent);

    auto command_pool = createCommandPool(device, gpu.queue_families.graphics.value());
    auto command_buffers = allocateCommandBuffers(device, command_pool, framebuffers.size());
    recordCommandBuffers(command_buffers, render_pass, framebuffers, swapchain_extent, pipeline);
    
    mainLoop(vw, device, swapchain, queues, command_buffers);

    vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());
    vkDestroyCommandPool(device, command_pool, nullptr);

    for (auto& framebuffer: framebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    for (auto& view: views) {
        vkDestroyImageView(device, view, nullptr);
    }

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);

    vkDestroySwapchainKHR(device, swapchain, nullptr);

    vkDestroyDevice(device, nullptr);

    destroyVulkanWindow(vi.instance, vw);
    destroyVulkanInstance(vi);
}

int main() {
    glfwInit();
    run();
    glfwTerminate();
}

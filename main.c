#define _POSIX_C_SOURCE 200112L

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iso646.h>

#include <sys/stat.h>

#ifdef NDEBUG
#define DEBUG (false)
#else
#define DEBUG (true)
#endif

#define STRING(v) #v

#define ARRAY_SIZE(arr) (sizeof((arr)) / sizeof(*(arr)))

#define CHECK_RETURN(res) if ((res) != VK_SUCCESS) { return res; }

FILE* getLog() {
    return stdout;
}

FILE* getErrorLog() {
    return stderr;
}

const char* vulkanErrorString(VkResult res) {
    switch (res) {
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "extension not present";
        default: {
            fprintf(getErrorLog(), "FIXME: Unknown VkResult %d\n", res);
            enum {buffer_size = 32};
            static char buffer[buffer_size];
            snprintf(buffer, buffer_size, "%d", res);
            return buffer;
        }
    }
}

typedef enum {
    FR_SUCCESS = 0,
    FR_OPEN_FAILED,
    FR_SIZE_FAILED,
    FR_MALLOC_FAILED,
    FR_READ_FAILED,
} FResult;

FResult getFileSize(FILE* file, size_t* size_ptr) {
    struct stat statbuf;
    if (fstat(fileno(file), &statbuf) or
        !S_ISREG(statbuf.st_mode)) {
        return FR_SIZE_FAILED;
    }
    *size_ptr = statbuf.st_size;
    return FR_SUCCESS;
}

FResult loadBinaryFileFP(FILE* fp, char** buffer_ptr, size_t* size_ptr) {
    size_t size = 0;
    FResult res = getFileSize(fp, &size);
    if (res) {
        return res;
    }

    char* buffer = calloc(size, sizeof(char));
    if (!buffer) {
        return FR_MALLOC_FAILED;
    }

    size_t num_read = fread(buffer, 1, size, fp);
    if (num_read != size or ferror(fp)) {
        free(buffer);
        return FR_READ_FAILED;
    }

    *buffer_ptr = buffer;
    *size_ptr = size;

    return FR_SUCCESS;
}

FResult loadBinaryFile(const char* path, char** buffer_ptr, size_t* size_ptr) {
    FILE* fp = fopen(path, "r");
    if (fp) {
        FResult res = loadBinaryFileFP(fp, buffer_ptr, size_ptr);
        fclose(fp);
        return res;
    }
    else {
        return FR_OPEN_FAILED;
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugErrorCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* p)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(getErrorLog(), "VULKAN ERROR: %s \n", callback_data->pMessage);
    }
    return VK_FALSE;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugInfoCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* p)
{
    fprintf(getErrorLog(), "VULKAN INFO: %s \n", callback_data->pMessage);
    return VK_FALSE;
}

VkResult createDebugMessenger(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger
) {
    PFN_vkCreateDebugUtilsMessengerEXT f = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, STRING(vkCreateDebugUtilsMessengerEXT));
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
    PFN_vkDestroyDebugUtilsMessengerEXT f = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, STRING(vkDestroyDebugUtilsMessengerEXT));
    if (f) {
        f(instance, messenger, pAllocator);
    }
}

typedef struct {
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger;
} VulkanInstance;

const char** getInstanceLayers(unsigned* num_layers_ptr) {
#if DEBUG
    static const char* required_layers[] = {
        "VK_LAYER_KHRONOS_validation",
    };
    *num_layers_ptr = ARRAY_SIZE(required_layers);
    return required_layers;
#else
    *num_layers_ptr = 0;
    return NULL;
#endif
}

const char** getInstanceExtensions(unsigned* num_extensions_ptr) {
#if DEBUG
    static const char* required_extensions[] = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    enum {
        num_required_extensions = ARRAY_SIZE(required_extensions)
    };
#else
    const char** required_extensions = NULL;
    enum {
        num_required_extensions = 0
    };
#endif

    // TODO: using a static buffer here feels like a hack.
    // But the number of windowing system extensions for GLFW
    // is fixed and small (currently 2).
    enum {buffer_elems = 16};
    static const char* extensions[buffer_elems];

    static_assert(
        num_required_extensions <= buffer_elems,
        "Extensions buffer is too small for required extensions"
    );
    memcpy(extensions, required_extensions, sizeof(*required_extensions) * num_required_extensions);

    unsigned num_display_extensions = 0;
    const char** display_extensions = glfwGetRequiredInstanceExtensions(&num_display_extensions);
    unsigned num_extensions = num_display_extensions + num_required_extensions;
    assert(
        num_extensions <= buffer_elems &&
        "Extensions buffer is too small for display extensions"
    );
    memcpy(
        extensions + num_required_extensions, display_extensions,
        num_display_extensions * sizeof(*display_extensions)
    );

    *num_extensions_ptr = num_display_extensions + num_required_extensions;

    return extensions;
}

VkResult createVulkanInstance(VulkanInstance* vi) {
    unsigned num_layers = 0;
    const char** layers = getInstanceLayers(&num_layers);
    unsigned num_extensions = 0;
    const char** extensions = getInstanceExtensions(&num_extensions);
    for (unsigned i = 0; i < num_layers; i++) {
        fprintf(getLog(), "Request layer %s\n", layers[i]);
    }
    for (unsigned i = 0; i < num_extensions; i++) {
        fprintf(getLog(), "Require instance extension %s\n", extensions[i]);
    }

    VkDebugUtilsMessengerCreateInfoEXT messenger_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = NULL,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugErrorCallback,
        .pUserData = NULL,
    };

    VkInstanceCreateInfo instance_create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = DEBUG ? &messenger_create_info : NULL,
        .flags = 0,
        .pApplicationInfo = NULL,
        .enabledLayerCount = num_layers,
        .ppEnabledLayerNames = layers,
        .enabledExtensionCount = num_extensions,
        .ppEnabledExtensionNames = extensions,
    };

    VkResult res = vkCreateInstance(&instance_create_info, NULL, &vi->instance);
    CHECK_RETURN(res);
#if DEBUG
    res = createDebugMessenger(vi->instance, &messenger_create_info, NULL, &vi->messenger);
    CHECK_RETURN(res);
#endif

    return VK_SUCCESS;
}

void destroyVulkanInstance(VulkanInstance* vi) {
#if DEBUG
    destroyDebugMessenger(vi->instance, vi->messenger, NULL);
#endif
    vkDestroyInstance(vi->instance, NULL);
}

typedef struct {
    GLFWwindow* window;
    VkSurfaceKHR surface;
} VulkanWindow;

VkResult createVulkanWindow(VkInstance instance, unsigned width, unsigned height, VulkanWindow* vw) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, false);
    vw->window = glfwCreateWindow(width, height, "Vulkan Tutorial", NULL, NULL);
    if (!vw->window) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    VkResult res = glfwCreateWindowSurface(instance, vw->window, NULL, &vw->surface);
    CHECK_RETURN(res);
    return res;
}

bool shouldClose(GLFWwindow* window) {
    glfwPollEvents();
    return glfwWindowShouldClose(window);
}

void destroyVulkanWindow(VkInstance instance, VulkanWindow* vw) {
    vkDestroySurfaceKHR(instance, vw->surface, NULL);
    glfwDestroyWindow(vw->window);
}

const char** getGPUExtensions(unsigned* num_extensions_ptr) {
    static const char* required_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    *num_extensions_ptr = ARRAY_SIZE(required_extensions);
    return required_extensions;
}

typedef struct{
    unsigned graphics;
    unsigned present;
    bool has_graphics;
    bool has_present;
} QueueFamilies;

bool getGPUGraphicsQueueFamily(
    const VkQueueFamilyProperties* queues,
    unsigned num_queues, unsigned* graphics_ptr
) {
    for (unsigned i = 0; i < num_queues; i++) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            *graphics_ptr = i;
            return true;
        }
    }
    return false;
}

bool getGPUPresentQueueFamily(
    VkPhysicalDevice gpu, VkSurfaceKHR surface,
    unsigned num_queues, unsigned* present_ptr
) {
    for (unsigned i = 0; i < num_queues; i++) {
        VkBool32 has_present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &has_present);
        if (has_present) {
            *present_ptr = i;
            return true;
        }
    }
    return false;
}

VkResult getGPUQueueFamilies(VkPhysicalDevice gpu, VkSurfaceKHR surface, QueueFamilies* queue_families) {
    unsigned num_queues = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queues, NULL);
    VkQueueFamilyProperties* queues = calloc(num_queues, sizeof(*queues));
    if (!queues) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queues, queues);

    queue_families->has_graphics = getGPUGraphicsQueueFamily(queues, num_queues, &queue_families->graphics);
    queue_families->has_present = getGPUPresentQueueFamily(gpu, surface, num_queues, &queue_families->present);

    free(queues);

    return VK_SUCCESS;
}

typedef struct {
    VkSurfaceCapabilitiesKHR capabilities;
    VkSurfaceFormatKHR* formats;
    VkPresentModeKHR* present_modes;
    unsigned num_formats;
    unsigned num_present_modes;
} SurfaceInfo;

void freeSurfaceInfo(SurfaceInfo* info) {
    free(info->formats);
    free(info->present_modes);
}

VkResult getGPUSurfaceInfoImpl(
    VkPhysicalDevice gpu, VkSurfaceKHR surface, SurfaceInfo* info
) {
    VkResult res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &info->capabilities);
    CHECK_RETURN(res);

    res = vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &info->num_formats, NULL);
    CHECK_RETURN(res);
    info->formats = calloc(info->num_formats, sizeof(*info->formats));
    if (!info->formats) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    res = vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &info->num_formats, info->formats);
    CHECK_RETURN(res);

    res = vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &info->num_present_modes, NULL);
    CHECK_RETURN(res);
    info->present_modes = calloc(info->num_present_modes, sizeof(*info->present_modes));
    if (!info->present_modes) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    res = vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &info->num_present_modes, info->present_modes);
    CHECK_RETURN(res);

    return VK_SUCCESS;
}

VkResult getGPUSurfaceInfo(VkPhysicalDevice gpu, VkSurfaceKHR surface, SurfaceInfo* info) {
    info->formats = NULL;
    info->present_modes = NULL;
    VkResult res = getGPUSurfaceInfoImpl(gpu, surface, info);
    if (res != VK_SUCCESS) {
        freeSurfaceInfo(info);
    }
    return res;
}

bool extensionSupported(
    VkExtensionProperties* supported, unsigned num_supported,
    const char* extension
) {
    for (unsigned i = 0; i < num_supported; i++) {
        if (strcmp(supported[i].extensionName, extension) == 0) {
            return true;
        }
    }
    return false;
}

bool extensionsSupported(
    VkExtensionProperties* supported, unsigned num_supported,
    const char** required, unsigned num_required
) {
    for (unsigned i = 0; i < num_required; i++) {
        if (!extensionSupported(supported, num_supported, required[i])) {
            return false;
        }
    }
    return true;
}

bool gpuExtensionsSupported(VkPhysicalDevice gpu) {
    unsigned num_required = 0;
    const char** required = getGPUExtensions(&num_required);

    unsigned num_supported = 0;
    if (vkEnumerateDeviceExtensionProperties(gpu, NULL, &num_supported, NULL) != VK_SUCCESS) {
        return false;
    }
    VkExtensionProperties* supported = calloc(num_supported, sizeof(*supported));
    if (!supported) { return false; }
    if (vkEnumerateDeviceExtensionProperties(gpu, NULL, &num_supported, supported) != VK_SUCCESS) {
        return false;
    }

    bool all = extensionsSupported(supported, num_supported, required, num_required);
    free(supported);

    return all;
}

bool gpuSurfaceSupported(const SurfaceInfo* surface_info) {
    return surface_info->num_formats and surface_info->num_present_modes;
}

typedef struct {
    VkPhysicalDevice gpu;
    QueueFamilies queue_families;
    SurfaceInfo surface_info;
} VulkanGPU;

void freeVulkanGPU(VulkanGPU* gpu) {
    freeSurfaceInfo(&gpu->surface_info);
}

VkResult getInstanceGPUsImpl(
    VkInstance instance, VkSurfaceKHR surface,
    VkPhysicalDevice** gpus_ptr,
    VulkanGPU** vgpus_ptr, unsigned* num_vgpus_ptr
) {
    unsigned num_gpus = 0;
    VkResult res = vkEnumeratePhysicalDevices(instance, &num_gpus, NULL);
    CHECK_RETURN(res);
    VkPhysicalDevice* gpus = calloc(num_gpus, sizeof(*gpus));
    if (!gpus) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    *gpus_ptr = gpus;
    res = vkEnumeratePhysicalDevices(instance, &num_gpus, gpus);
    CHECK_RETURN(res);

    VulkanGPU* vgpus = calloc(num_gpus, sizeof(*vgpus));
    if (!vgpus) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    *vgpus_ptr = vgpus;
    *num_vgpus_ptr = num_gpus;

    for (unsigned i = 0; i < num_gpus; i++) {
        vgpus[i].gpu = gpus[i];
        res = getGPUQueueFamilies(vgpus[i].gpu, surface, &vgpus[i].queue_families);
        CHECK_RETURN(res);
        res = getGPUSurfaceInfo(vgpus[i].gpu, surface, &vgpus[i].surface_info);
        CHECK_RETURN(res);
    }

    return VK_SUCCESS;
}

VkResult getInstanceGPUs(VkInstance instance, VkSurfaceKHR surface, VulkanGPU** vgpus_ptr, unsigned* num_vgpus_ptr) {
    VkPhysicalDevice* gpus = NULL;
    *vgpus_ptr = NULL;
    VkResult res = getInstanceGPUsImpl(instance, surface, &gpus, vgpus_ptr, num_vgpus_ptr);
    free(gpus);
    if (res != VK_SUCCESS) {
        free(*vgpus_ptr);
        return res;
    }
    return VK_SUCCESS;
}

unsigned gpuScore(VulkanGPU gpu) {
    if (gpuExtensionsSupported(gpu.gpu) and
        gpuSurfaceSupported(&gpu.surface_info) and
        gpu.queue_families.has_graphics and
        gpu.queue_families.has_present) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpu.gpu, &props);
        return (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 2 : 1;
    }
    return 0;
}

VulkanGPU* gpuWithHighestScore(VulkanGPU* gpus, unsigned num_gpus) {
    unsigned max_i = 0;
    for (unsigned i = 1; i < num_gpus; i++) {
        if (gpuScore(gpus[max_i]) < gpuScore(gpus[i])) {
            max_i = i;
        }
    }
    return &gpus[max_i];
}

const char* gpuName(VkPhysicalDevice gpu) {
    static VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    return props.deviceName;
}

VkResult selectVulkanGPU(VkInstance instance, VkSurfaceKHR surface, VulkanGPU* gpu) {
    VulkanGPU* gpus = NULL;
    unsigned num_gpus = 0;
    VkResult res = getInstanceGPUs(instance, surface, &gpus, &num_gpus);
    CHECK_RETURN(res);

    VulkanGPU* max_gpu_ptr = gpuWithHighestScore(gpus, num_gpus);
    *gpu = *max_gpu_ptr;

    fprintf(getLog(), "Available GPUs:\n");
    for (unsigned i = 0; i < num_gpus; i++) {
        fprintf(getLog(), "%s\n", gpuName(gpus[i].gpu));
    }
    fprintf(getLog(), "Select %s\n", gpuName(gpu->gpu));

    for (unsigned i = 0; i < num_gpus; i++) {
        if (&gpus[i] != max_gpu_ptr) {
            freeVulkanGPU(&gpus[i]);
        }
    }
    free(gpus);

    return VK_SUCCESS;
}

typedef struct {
    VkQueue graphics;
    VkQueue present;
} Queues;

typedef struct {
    VkDevice device;
    Queues queues;
} VulkanDevice;

unsigned* getQueueFamilyIndices(
    const QueueFamilies* queue_families,
    unsigned* num_queue_family_indices_ptr
) {
    if (!queue_families->has_graphics or !queue_families->has_present) {
        return NULL;
    }

    static unsigned queue_family_indices[2];
    queue_family_indices[0] = queue_families->graphics;
    unsigned n = 1;
    if (queue_families->graphics != queue_families->present) {
        queue_family_indices[1] = queue_families->present;
        n++;
    }
    *num_queue_family_indices_ptr = n;

    return queue_family_indices;
}

VkResult createVulkanDevice(VulkanGPU* gpu, VulkanDevice* vd) {
    unsigned num_queue_family_indices = 0;
    unsigned* queue_family_indices = getQueueFamilyIndices(&gpu->queue_families, &num_queue_family_indices);
    if (!queue_family_indices) { return VK_ERROR_FEATURE_NOT_PRESENT; }

    VkDeviceQueueCreateInfo* queue_create_infos = calloc(num_queue_family_indices, sizeof(*queue_create_infos));
    if (!queue_create_infos) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    float priority = 1.0f;
    for (unsigned i = 0; i < num_queue_family_indices; i++) {
        queue_create_infos[i] = (VkDeviceQueueCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .queueFamilyIndex = queue_family_indices[i],
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
    }

    unsigned num_extensions = 0;
    const char** extensions = getGPUExtensions(&num_extensions);
    for (unsigned i = 0; i < num_extensions; i++) {
        fprintf(getLog(), "Require device extension %s\n", extensions[i]);
    }

    VkDeviceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueCreateInfoCount = num_queue_family_indices,
        .pQueueCreateInfos = queue_create_infos,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = num_extensions,
        .ppEnabledExtensionNames = extensions,
        .pEnabledFeatures = NULL,
    };
    VkResult res = vkCreateDevice(gpu->gpu, &create_info, NULL, &vd->device);
    free(queue_create_infos);
    CHECK_RETURN(res);

    vkGetDeviceQueue(vd->device, gpu->queue_families.graphics, 0, &vd->queues.graphics);
    vkGetDeviceQueue(vd->device, gpu->queue_families.present, 0, &vd->queues.present);

    return VK_SUCCESS;
}

void destroyVulkanDevice(VulkanDevice* vd) {
    vkDestroyDevice(vd->device, NULL);
}

VkSurfaceFormatKHR selectSurfaceFormat(const VkSurfaceFormatKHR* formats, unsigned num_formats) {
    for (unsigned i = 0; i < num_formats; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB and
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return formats[i];
        }
    }
    return formats[0];
}

VkPresentModeKHR selectPresentMode(const VkPresentModeKHR* present_modes, unsigned num_present_modes) {
    for (unsigned i = 0; i < num_present_modes; i++) {
        if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

unsigned min(unsigned a, unsigned b) {
    return (a < b) ? a : b;
}

unsigned max(unsigned a, unsigned b) {
    return (a > b) ? a : b;
}

unsigned clamp(unsigned x, unsigned a, unsigned b) {
    x = min(x, a);
    x = max(x, b);
    return x;
}

VkExtent2D selectSwapchainExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR* capabilities) {
    VkExtent2D special_value = {
        .width = UINT32_MAX,
        .height = UINT32_MAX,
    };
    if (capabilities->currentExtent.width == special_value.width and
        capabilities->currentExtent.height == special_value.height) {
        int w = 0;
        int h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        return (VkExtent2D) {
            .width = clamp(
                w, capabilities->minImageExtent.width, capabilities->maxImageExtent.width
            ),
            .height = clamp(
                h, capabilities->minImageExtent.height, capabilities->maxImageExtent.height
            ),
        };
    } else {
        return capabilities->currentExtent;
    }
}

unsigned selectSwapchainImageCount(const VkSurfaceCapabilitiesKHR* capabilities) {
    unsigned min_images = capabilities->minImageCount;
    unsigned max_images = capabilities->maxImageCount;
    unsigned image_count = min_images + 1;
    if (max_images) {
        image_count = min(image_count, max_images);
    }
    return image_count;
}

VkResult getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, VkImage** images_ptr, unsigned* num_images_ptr) {
    VkResult res = VK_SUCCESS;

    unsigned num_images = 0;
    res = vkGetSwapchainImagesKHR(device, swapchain, &num_images, NULL);
    CHECK_RETURN(res);
    VkImage* images = calloc(num_images, sizeof(*images));
    if (!images) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    res = vkGetSwapchainImagesKHR(device, swapchain, &num_images, images);
    if (res != VK_SUCCESS) {
        free(images);
        return res;
    }

    *images_ptr = images;
    *num_images_ptr = num_images;

    return VK_SUCCESS;
}

typedef struct {
    VkSwapchainKHR swapchain;
    VkSurfaceFormatKHR format;
    VkPresentModeKHR present_mode;
    VkExtent2D extent;
    VkImage* images;
    unsigned num_images;
} VulkanSwapchain;

VkResult createVulkanSwapchain(
    VkDevice device, const VulkanWindow* vw,
    const VulkanGPU* gpu, VulkanSwapchain* swapchain
) {
    swapchain->format =
        selectSurfaceFormat(gpu->surface_info.formats, gpu->surface_info.num_formats);
    swapchain->present_mode =
        selectPresentMode(gpu->surface_info.present_modes, gpu->surface_info.num_present_modes);
    swapchain->extent = selectSwapchainExtent(vw->window, &gpu->surface_info.capabilities);
    unsigned image_count = selectSwapchainImageCount(&gpu->surface_info.capabilities);

    bool exclusive =
        gpu->queue_families.graphics == gpu->queue_families.present;
    unsigned queues[] = {
        gpu->queue_families.graphics, gpu->queue_families.present
    };

    VkSwapchainCreateInfoKHR create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .surface = vw->surface,
        .minImageCount = image_count,
        .imageFormat = swapchain->format.format,
        .imageColorSpace = swapchain->format.colorSpace,
        .imageExtent = swapchain->extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode =
            exclusive ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount =
            exclusive ? 0 : ARRAY_SIZE(queues),
        .pQueueFamilyIndices = exclusive ? NULL : queues,
        .preTransform = gpu->surface_info.capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = swapchain->present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };
    VkResult res = VK_SUCCESS;
    res = vkCreateSwapchainKHR(device, &create_info, NULL, &swapchain->swapchain);
    CHECK_RETURN(res);

    res = getSwapchainImages(device, swapchain->swapchain, &swapchain->images, &swapchain->num_images);
    if (res != VK_SUCCESS) {
        vkDestroySwapchainKHR(device, swapchain->swapchain, NULL);
        return res;
    }

    return res;
}

void destroyVulkanSwapchain(VkDevice device, VulkanSwapchain* vs) {
    free(vs->images);
    vkDestroySwapchainKHR(device, vs->swapchain, NULL);
}

#if 0
std::vector<VkImageView> createImageViews(VkDevice device, VkFormat format, const std::vector<VkImage> images) {
    std::vector<VkImageView> views{images.size()};
    for (unsigned i = 0; i < images.size(); i++) {
        VkImageViewCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = NULL,
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
        vkCreateImageView(device, &create_info, NULL, &views[i]);
    }
    return views;
}

void destroyImageViews(VkDevice dev, std::vector<VkImageView>& views) {
    for (auto& view: views) {
        vkDestroyImageView(dev, view, NULL);
    }
}

VkPipelineLayout createPipelineLayout(VkDevice device) {
    VkPipelineLayoutCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = NULL,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = NULL,
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &create_info, NULL, &layout);
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
        .pInputAttachments = NULL,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment_reference,
        .pResolveAttachments = NULL,
        .pDepthStencilAttachment = NULL,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = NULL,
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
        .pNext = NULL,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &attachment_description,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    VkRenderPass render_pass = VK_NULL_HANDLE;
    vkCreateRenderPass(device, &create_info, NULL, &render_pass);

    return render_pass;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<std::byte>& binary) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = binary.size(),
        .pCode = reinterpret_cast<decltype(create_info.pCode)>(binary.data()),
    };
    VkShaderModule shader_module;
    vkCreateShaderModule(device, &create_info, NULL, &shader_module);
    return shader_module;
}

VkPipelineShaderStageCreateInfo getPipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, VkShaderModule shader_module)
{
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = stage,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = NULL,
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
        .pNext = NULL,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = NULL,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = NULL,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = NULL,
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
        .pNext = NULL,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = NULL,
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
        .pNext = NULL,
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
        .pNext = NULL,
        .flags = 0,
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    auto layout = createPipelineLayout(device);

    VkGraphicsPipelineCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stageCount = shader_stages.size(),
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pTessellationState = NULL,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState = &multisample_state,
        .pDepthStencilState = NULL,
        .pColorBlendState = &color_blend_state,
        .pDynamicState = NULL,
        .layout = layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create_info, NULL, &pipeline);

    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyShaderModule(device, vertModule, NULL);
    vkDestroyShaderModule(device, fragModule, NULL);

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
            .pNext = NULL,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &views[i],
            .width = swapchain_extent.width,
            .height = swapchain_extent.height,
            .layers = 1,
        };
        vkCreateFramebuffer(device, &create_info, NULL, &framebuffers[i]);
    }
    return framebuffers;
}

void destroyFramebuffers(VkDevice dev, std::vector<VkFramebuffer>& fbs) {
    for (auto& fb: fbs) {
        vkDestroyFramebuffer(dev, fb, NULL);
    }
}

VkCommandPool createCommandPool(VkDevice device, unsigned graphics_queue) {
    VkCommandPoolCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueFamilyIndex = graphics_queue,
    };
    VkCommandPool command_pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &create_info, NULL, &command_pool);
    return command_pool;
}

std::vector<VkCommandBuffer> allocateCommandBuffers(
    VkDevice device, VkCommandPool command_pool, unsigned num_command_buffers)
{
    VkCommandBufferAllocateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
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
            .pNext = NULL,
            .flags = 0,
            .pInheritanceInfo = NULL,
        };
        vkBeginCommandBuffer(command_buffers[i], &begin_info);

        VkClearValue clear_color = {
            .color = {.float32{0.0f, 0.0f, 0.0f, 1.0f}}
        };
        VkRenderPassBeginInfo render_pass_begin_info = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = NULL,
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
        .pNext = NULL,
        .flags = 0,
    };
    VkSemaphore semaphore = VK_NULL_HANDLE;
    vkCreateSemaphore(device, &create_info, NULL, &semaphore);
    return semaphore;
}

std::vector<VkSemaphore> createSemaphores(VkDevice device, unsigned num_semaphores) {
    std::vector<VkSemaphore> semaphores{num_semaphores};
    std::generate(semaphores.begin(), semaphores.end(), [&]() { return createSemaphore(device); });
    return semaphores;
}

void destroySemaphores(VkDevice device, std::vector<VkSemaphore>& semaphores) {
    for (auto& semaphore: semaphores) {
        vkDestroySemaphore(device, semaphore, NULL);
    }
    semaphores.clear();
}

VkFence createFence(VkDevice device) {
    VkFenceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = NULL,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    VkFence fence;
    vkCreateFence(device, &create_info, NULL, &fence);
    return fence;
};

void destroyFences(VkDevice device, std::vector<VkFence>& fences) {
    for (auto& fence: fences) {
        vkDestroyFence(device, fence, NULL);
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
        .pNext = NULL,
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
        .pNext = NULL,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image,
        .pResults = NULL,
    };
    vkQueuePresentKHR(queue, &present_info);
}

void mainLoop(
    GLFWwindow* window, VulkanDevice vd,
    VkSwapchainKHR swapchain,
    const std::vector<VkCommandBuffer>& command_buffers)
{
    auto num_swapchain_images = command_buffers.size();
    auto acquire_semaphores = createSemaphores(vd.device, num_swapchain_images);
    auto render_semaphores = createSemaphores(vd.device, num_swapchain_images);
    // Fences are used so that no more frames are being rendered at the same
    // time than there are swapchain images.
    auto frame_fences = createFences(vd.device, num_swapchain_images);
    // The swapchain might return an image whose index differs from the current
    // frame's. In this case waiting on the current frame's fence will not be
    // sufficient, as the image might still be accessed by another frame's commands.
    // Store the fence of the last frame that rendered to an image to wait
    // for all accesses to be done.
    std::vector<VkFence> image_fences{num_swapchain_images, VK_NULL_HANDLE};

    unsigned frame = 0;
    while (!shouldClose(window)) {
        constexpr auto uint64_t_max = std::numeric_limits<uint64_t>::max();

        auto& acquire_semaphore = acquire_semaphores[frame];
        auto& render_semaphore = render_semaphores[frame];
        auto& frame_fence = frame_fences[frame];

        unsigned image = 0;
        vkAcquireNextImageKHR(vd.device, swapchain, uint64_t_max, acquire_semaphore, VK_NULL_HANDLE, &image);

        auto& image_fence = image_fences[image];
        auto& command_buffer = command_buffers[image];

        if (image_fence != VK_NULL_HANDLE) {
            vkWaitForFences(vd.device, 1, &image_fence, true, uint64_t_max);
        }
        image_fence = frame_fence;
        vkWaitForFences(vd.device, 1, &frame_fence, true, uint64_t_max);
        vkResetFences(vd.device, 1, &frame_fence);

        submitDraw(vd.queues.graphics, command_buffer, acquire_semaphore, render_semaphore, frame_fence);
        submitPresent(vd.queues.present, swapchain, image, render_semaphore);
        glfwSwapBuffers(window);

        frame = (frame + 1) % num_swapchain_images;
    }
    vkDeviceWaitIdle(vd.device);

    destroyFences(vd.device, frame_fences);
    destroySemaphores(vd.device, render_semaphores);
    destroySemaphores(vd.device, acquire_semaphores);
}
#endif

void run() {
    VkResult res = VK_SUCCESS;

    VulkanInstance vi;
    res = createVulkanInstance(&vi);
    if (res != VK_SUCCESS) {
        fprintf(getErrorLog(), "Failed to create Vulkan instance: %s\n", vulkanErrorString(res));
        return;
    }

    VulkanWindow vw;
    res = createVulkanWindow(vi.instance, 1280, 720, &vw);
    if (res != VK_SUCCESS) {
        fprintf(getErrorLog(), "Failed to create Vulkan window: %s\n", vulkanErrorString(res));
        return;
    }

    VulkanGPU gpu;
    res = selectVulkanGPU(vi.instance, vw.surface, &gpu);
    if (res != VK_SUCCESS) {
        fprintf(getErrorLog(), "Failed to select Vulkan GPU: %s\n", vulkanErrorString(res));
        return;
    }
    
    VulkanDevice vd;
    res = createVulkanDevice(&gpu, &vd);
    if (res != VK_SUCCESS) {
        fprintf(getErrorLog(), "Failed to create Vulkan device: %s\n", vulkanErrorString(res));
        return;
    }

    VulkanSwapchain swapchain;
    res = createVulkanSwapchain(vd.device, &vw, &gpu, &swapchain);
    if (res != VK_SUCCESS) {
        fprintf(getErrorLog(), "Failed to create Vulkan swapchain: %s\n", vulkanErrorString(res));
        return;
    }
#if 0

    auto render_pass = createRenderPass(device.device, swapchain.format.format);
    auto pipeline = createPipeline(device.device, swapchain.extent, render_pass);

    auto views = createImageViews(device.device, swapchain.format.format, swapchain.images);
    auto framebuffers = createSwapchainFramebuffers(device.device, render_pass, views, swapchain.extent);

    auto command_pool = createCommandPool(device.device, gpu.queue_families.graphics.value());
    auto command_buffers = allocateCommandBuffers(device.device, command_pool, framebuffers.size());
    recordCommandBuffers(command_buffers, render_pass, framebuffers, swapchain.extent, pipeline);
    
    mainLoop(vw.window, device, swapchain.swapchain, command_buffers);

    vkFreeCommandBuffers(device.device, command_pool, command_buffers.size(), command_buffers.data());
    vkDestroyCommandPool(device.device, command_pool, NULL);

    destroyFramebuffers(device.device, framebuffers);
    destroyImageViews(device.device, views);

    vkDestroyPipeline(device.device, pipeline, NULL);
    vkDestroyRenderPass(device.device, render_pass, NULL);

#endif
    destroyVulkanSwapchain(vd.device, &swapchain);
    destroyVulkanDevice(&vd);
    freeVulkanGPU(&gpu);
    destroyVulkanWindow(vi.instance, &vw);
    destroyVulkanInstance(&vi);
}

int main() {
    glfwInit();
    run();
    glfwTerminate();
}
